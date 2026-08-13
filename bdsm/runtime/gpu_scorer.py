#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from pipeline_security import kafka_ssl_config, safe_pickle_loads
from score_layout import HEAD_ORDER, to_score_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gpu_scorer")

PINNED_BACKBONE_SHA256 = "8ccf35dcdc7b6e67ab899b911ae32b991725ffa0ac1497eef2405994a37ebcab"
PINNED_HEAD_REGISTRY_HASH = "1bb89df6b629a615"
PINNED_RUN_NAME = "bdsm_v9_h8_mlp512x256"
DEFAULT_BOOTSTRAP = "localhost:9092"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kafka-bootstrap", default=DEFAULT_BOOTSTRAP)
    p.add_argument("--kafka-username", default="kafka-user")
    p.add_argument("--kafka-password", required=True)
    p.add_argument("--input-topic", default="abuse_ready_batches")
    p.add_argument("--output-topic", default="abuse_scored_results")
    p.add_argument("--kafka-group", default="bdsm-gpu-scorer")
    p.add_argument("--backbone-dir", required=True)
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--model-version", default=PINNED_RUN_NAME)
    p.add_argument("--health-port", type=int, default=8081)
    return p.parse_args()


def _verify_pinned_weights(args) -> tuple[dict, dict, list[str]]:
    manifest_path = os.path.join(args.backbone_dir, "MANIFEST.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    npz_path = os.path.join(args.backbone_dir, "backbone.npz")
    actual = _file_sha256(npz_path)
    if actual != PINNED_BACKBONE_SHA256 or manifest.get("file_sha256") != PINNED_BACKBONE_SHA256:
        raise RuntimeError(
            f"backbone hash mismatch: file={actual} manifest={manifest.get('file_sha256')} "
            f"pinned={PINNED_BACKBONE_SHA256} — refusing to start"
        )

    with open(os.path.join(args.head_checkpoint, "config.json")) as f:
        head_cfg = json.load(f)
    if head_cfg.get("head_registry_hash") != PINNED_HEAD_REGISTRY_HASH:
        raise RuntimeError(
            f"head registry hash {head_cfg.get('head_registry_hash')} != "
            f"{PINNED_HEAD_REGISTRY_HASH} — refusing to start"
        )
    if head_cfg.get("backbone_sha256") != PINNED_BACKBONE_SHA256:
        raise RuntimeError(
            "head checkpoint was trained against a different backbone — refusing to start"
        )
    head_names = list(head_cfg["head_names"])
    unknown = [n for n in head_names if n not in HEAD_ORDER]
    if unknown:
        raise RuntimeError(f"head names {unknown} are not in HEAD_ORDER — refusing to start")
    log.info(
        f"weights pinned OK: backbone sha {actual[:8]}…, "
        f"head registry {PINNED_HEAD_REGISTRY_HASH}, heads {head_names}"
    )
    return manifest, head_cfg, head_names


def _start_health_server(port: int, metrics: dict, config_view: dict):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") == "/config":
                body = json.dumps(config_view, indent=2).encode()
            else:
                body = "".join(f"{k}={v}\n" for k, v in sorted(metrics.items())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    srv = HTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _normalize_batch(batch: dict) -> dict:
    import feature_norm

    out = dict(batch)
    aseq = batch.get("action_seq", batch)
    out["action_seq"] = feature_norm.normalize_aseq(aseq)
    return out


def main() -> int:
    args = _parse_args()
    manifest, head_cfg, head_names = _verify_pinned_weights(args)

    import jax
    import jax.numpy as jnp
    import load_backbone
    import model as bdsm_model
    import zstandard as zstd
    from confluent_kafka import Consumer, KafkaError, Producer
    from task_heads import head_logits, load_head_params

    cfg = manifest["source_config"]
    params = load_backbone.load_params(
        os.path.join(args.backbone_dir, "backbone.npz"),
        os.path.join(args.backbone_dir, "MANIFEST.json"),
    )
    bc = bdsm_model.BackboneConfig.from_checkpoint_config(cfg)
    encode_fn = bdsm_model.build_encode_fn(bc)
    head_params = {k: jnp.asarray(v) for k, v in load_head_params(args.head_checkpoint).items()}

    @jax.jit
    def score_batch(p, hp, b):
        _, cls_raw, _ = encode_fn.apply(p, jax.random.PRNGKey(0), b)
        return jax.nn.sigmoid(head_logits(hp, cls_raw))

    kafka_common = {
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        **kafka_ssl_config(),
        "bootstrap.servers": args.kafka_bootstrap,
        "sasl.username": args.kafka_username,
        "sasl.password": args.kafka_password,
    }
    consumer = Consumer(
        {
            **kafka_common,
            "group.id": args.kafka_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "fetch.message.max.bytes": 10 * 1024 * 1024,
        }
    )
    consumer.subscribe([args.input_topic])
    producer = Producer(
        {
            **kafka_common,
            "message.max.bytes": 4 * 1024 * 1024,
            "linger.ms": 0,
            "acks": "1",
        }
    )
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=1)

    metrics = {
        "batches": 0,
        "users": 0,
        "skipped_batches": 0,
        "produce_errors": 0,
        "start_ts": int(time.time()),
    }
    _start_health_server(
        args.health_port,
        metrics,
        {
            "model_version": args.model_version,
            "input_topic": args.input_topic,
            "output_topic": args.output_topic,
            "backbone_sha256": PINNED_BACKBONE_SHA256,
            "head_registry_hash": PINNED_HEAD_REGISTRY_HASH,
            "head_names": head_names,
        },
    )

    def _on_delivery(err, _msg):
        if err is not None:
            metrics["produce_errors"] += 1

    t0_all = time.time()
    while True:
        msg = consumer.poll(0.1)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                log.warning(f"Kafka error: {msg.error()}")
            continue

        payload = safe_pickle_loads(dctx.decompress(msg.value()))
        batch, uids, B = payload["batch"], payload["uids"], payload["B"]
        score_ids = payload.get("score_ids")
        action_histograms = payload.get("action_histograms")
        behavioral_summaries = payload.get("behavioral_summaries")
        priority_lane = payload.get("priority_lane", False)

        try:
            batch_n = _normalize_batch(batch)
            probs8 = np.array(score_batch(params, head_params, jax.device_put(batch_n)))[:B]
            scores8 = np.stack([to_score_row(probs8[i], head_names) for i in range(B)])
            result = cctx.compress(
                pickle.dumps(
                    {
                        "uids": uids,
                        "scores": scores8.astype(np.float32),
                        "score_ids": score_ids,
                        "action_histograms": action_histograms,
                        "behavioral_summaries": behavioral_summaries,
                        "priority_lane": priority_lane,
                        "model_version": args.model_version,
                    },
                    protocol=5,
                )
            )
            producer.produce(args.output_topic, value=result, on_delivery=_on_delivery)
            producer.poll(0)
        except (KeyError, TypeError, ValueError) as e:
            if metrics["skipped_batches"] == 0:
                log.warning(f"Skipping incompatible batch ({type(e).__name__}: {e})")
            metrics["skipped_batches"] += 1
            continue

        metrics["batches"] += 1
        metrics["users"] += B
        if metrics["batches"] % 10 == 0:
            elapsed = time.time() - t0_all
            log.info(
                f"[GPU] {B} users | total {metrics['users']:,} "
                f"({metrics['users'] / max(elapsed, 1e-6):.0f}/sec) | "
                f"batches {metrics['batches']} | skipped {metrics['skipped_batches']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
