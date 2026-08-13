#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import queue
import threading
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from pipeline_security import kafka_ssl_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("prefetcher")

import sys as _sys

_sys.path.insert(0, os.environ.get("ABUSE_TRACE_LIB", ""))
try:
    import abuse_trace as _abuse_trace

    _trace = _abuse_trace.get_emitter()
except Exception as _e:
    log.warning(f"trace emitter unavailable, continuing without trace: {_e}")

    class _NoopTrace:
        enabled = False

        def emit(self, **kwargs):
            pass

        def is_active(self, uid):
            return False

        def is_active_many(self, uids):
            return set()

        def mark_active(self, uid):
            pass

        def get_metrics(self):
            return {"enabled": False}

    _trace = _NoopTrace()

import heads

N_MODEL_LABELS = heads.N_HEADS


def _pad_batch(batch: dict, batch_size: int) -> dict:
    import numpy as np

    def _n(obj):
        if isinstance(obj, np.ndarray):
            return obj.shape[0]
        if isinstance(obj, dict):
            for v in obj.values():
                n = _n(v)
                if n is not None:
                    return n
        return None

    n = _n(batch)
    if n is None or n >= batch_size:
        return batch

    def _pad(obj):
        if isinstance(obj, np.ndarray):
            return np.pad(obj, [(0, batch_size - n)] + [(0, 0)] * (obj.ndim - 1))
        if isinstance(obj, dict):
            return {k: _pad(v) for k, v in obj.items()}
        return obj

    return _pad(batch)


def serialize_batch(
    batch_dict: dict,
    user_ids: list[int],
    batch_size: int,
    score_ids: list[str] | None = None,
    action_histograms: list | None = None,
    behavioral_summaries: list | None = None,
    priority_lane: bool = False,
) -> bytes:
    import zstandard as zstd

    payload = {
        "B": batch_size,
        "uids": user_ids,
        "score_ids": score_ids,
        "batch": batch_dict,
        "action_histograms": action_histograms,
        "behavioral_summaries": behavioral_summaries,
        "priority_lane": priority_lane,
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    cctx = zstd.ZstdCompressor(level=1)
    return cctx.compress(raw)


def prefetcher_loop(
    redis_host: str,
    redis_port: int,
    redis_key: str,
    mh_endpoint: str,
    kafka_bootstrap: str,
    kafka_username: str,
    kafka_password: str,
    kafka_topic: str,
    batch_size: int,
    seq_len: int,
    n_action_types: int,
    metrics: dict,
    priority_lane: bool = False,
    seq_cache_enabled: bool = True,
    seq_cache_host: str = "localhost",
    seq_cache_port: int = 6379,
    seq_cache_ttl_secs: int = 4 * 3600,
    seq_cache_read_through: bool = True,
    realtime_cache_enabled: bool = False,
    realtime_cache_host: str = "localhost",
    realtime_cache_port: int = 6382,
    realtime_headroom_ms: int = 60_000,
    raw_seq_dir: str = os.environ.get("FREEZE_RAW_DIR", ""),
    features_dir: str = os.environ.get("FREEZE_FEATURES_DIR", ""),
):
    import redis as redis_lib
    from confluent_kafka import Producer

    r = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)
    r.ping()
    log.info(f"Connected to Redis at {redis_host}:{redis_port}")

    from manhattan_scorer import ManhattanArrowReader

    mh_pool = []
    for i in range(8):
        try:
            mh_pool.append(ManhattanArrowReader(endpoint=mh_endpoint))
        except Exception:
            if i == 0:
                raise
            break
    log.info(f"Manhattan connection pool: {len(mh_pool)} readers")

    from sequence_cache import merge_realtime_into_sequence as _rt_merge

    _caches: dict = {"seq": None, "rt": None}

    def _connect_cache(kind: str, make, deadline_s: float = 300.0, retry_s: float = 5.0):
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            try:
                _caches[kind] = make()
                return
            except Exception as _e:
                last_err = _e
                time.sleep(retry_s)
        log.warning(
            f"{kind} cache connect failed for {deadline_s:.0f}s "
            f"({last_err!r}); {'STEP 1' if kind == 'seq' else 'STEP 2'} "
            "DISABLED for this pod"
        )

    if seq_cache_enabled:
        from sequence_cache import SequenceCacheClient

        threading.Thread(
            target=_connect_cache,
            daemon=True,
            name="seq_cache_connect",
            args=("seq", lambda: SequenceCacheClient(host=seq_cache_host, port=seq_cache_port)),
        ).start()
    else:
        log.info("sequence cache DISABLED by flag (STEP 1 off)")

    if realtime_cache_enabled:
        from sequence_cache import RealtimeCacheClient

        threading.Thread(
            target=_connect_cache,
            daemon=True,
            name="rt_cache_connect",
            args=(
                "rt",
                lambda: RealtimeCacheClient(host=realtime_cache_host, port=realtime_cache_port),
            ),
        ).start()
    else:
        log.info("realtime cache DISABLED (STEP 2 off; --realtime-cache-enabled to turn on)")

    try:
        import abuse_v3_features as _rust

        _has_rust = True
        log.info("Using RUST extract_batch (fast path)")
    except ImportError:
        _has_rust = False
        log.info("Rust module not available, using Python fallback")

    producer_conf = {
        "bootstrap.servers": kafka_bootstrap,
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": kafka_username,
        "sasl.password": kafka_password,
        **kafka_ssl_config(),
        "message.max.bytes": 10 * 1024 * 1024,
        "compression.type": "none",
        "linger.ms": 0,
        "acks": "1",
        "queue.buffering.max.messages": 100,
    }
    producer = Producer(producer_conf)
    log.info(f"Kafka producer ready → {kafka_topic}")

    scoring_work_redis = None
    scoring_work_redis_key = os.environ.get("SCORING_WORK_REDIS_KEY", "abuse:scoring_work")
    classifier_shadow_enabled = os.environ.get("CLASSIFIER_SHADOW_ENABLED", "0") == "1"
    classifier_shadow_redis_key = os.environ.get(
        "CLASSIFIER_SHADOW_REDIS_KEY", "abuse:scoring_classifier_shadow_queue"
    )
    scoring_work_redis_enabled = os.environ.get("SCORING_WORK_REDIS_ENABLED", "0") == "1"
    scoring_work_redis_host = os.environ.get("SCORING_WORK_REDIS_HOST", "abuse-scoring-work-redis")
    scoring_work_metrics = {"rpush_ok": 0, "rpush_err": 0, "rpush_oom": 0}
    if scoring_work_redis_enabled:
        try:
            scoring_work_redis = redis_lib.Redis(
                host=scoring_work_redis_host,
                port=int(os.environ.get("SCORING_WORK_REDIS_PORT", "6379")),
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
                decode_responses=False,
            )
            scoring_work_redis.ping()
            log.info(
                f"SCORING_WORK_REDIS dual-write ENABLED → "
                f"{scoring_work_redis_host}:6379 key={scoring_work_redis_key}"
            )
        except Exception as _e:
            log.warning(
                f"SCORING_WORK_REDIS connect failed: {_e}; dual-write DISABLED for this pod"
            )
            scoring_work_redis = None
    else:
        log.info(
            "SCORING_WORK_REDIS dual-write DISABLED (set SCORING_WORK_REDIS_ENABLED=1 to enable)"
        )

    fetch_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="mh_fetch")
    _pool_idx = [0]

    def _fetch_mh(uid: int) -> bytes | None:
        for attempt in range(3):
            reader = mh_pool[(_pool_idx[0] + attempt) % len(mh_pool)]
            _pool_idx[0] += 1
            try:
                raw = reader.get_raw_bytes(uid)
                if raw and len(raw) > 100:
                    return raw
            except Exception:
                pass
        return None

    def _fetch(uid: int):
        raw = None
        cache_hit = False
        seq_cache = _caches["seq"]
        rt_cache = _caches["rt"]
        if seq_cache is not None:
            try:
                raw = seq_cache.get(uid)
                if raw and len(raw) > 100:
                    cache_hit = True
                    metrics["cache_hits"] = metrics.get("cache_hits", 0) + 1
                else:
                    raw = None
                    metrics["cache_misses"] = metrics.get("cache_misses", 0) + 1
            except Exception:
                raw = None
                metrics["cache_errors"] = metrics.get("cache_errors", 0) + 1

        if raw is None:
            raw = _fetch_mh(uid)
            if raw is not None and seq_cache is not None and seq_cache_read_through:
                seq_cache.set(uid, raw, seq_cache_ttl_secs)

        if raw is not None and rt_cache is not None:
            try:
                raw, n_merged = _rt_merge(raw, rt_cache, uid, headroom_ms=realtime_headroom_ms)
                if n_merged:
                    metrics["rt_merged_actions"] = metrics.get("rt_merged_actions", 0) + n_merged
                    metrics["rt_merged_users"] = metrics.get("rt_merged_users", 0) + 1
            except Exception:
                metrics["rt_errors"] = metrics.get("rt_errors", 0) + 1

        return (uid, raw, cache_hit)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard as _zstd_seq

    _seq_cctx = _zstd_seq.ZstdCompressor(level=3)
    _parquet_queue: queue.Queue = queue.Queue(maxsize=200)

    def _parquet_writer():
        import datetime

        raw_buf, feat_buf = [], []
        FLUSH_SIZE = 5000

        while True:
            try:
                item = _parquet_queue.get(timeout=10.0)
            except queue.Empty:
                if raw_buf:
                    _flush_parquet(raw_buf, feat_buf, raw_seq_dir, features_dir)
                    raw_buf.clear()
                    feat_buf.clear()
                continue

            uid, raw_bytes, feature_bytes, score_id = item
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

            raw_buf.append(
                {
                    "score_id": score_id,
                    "user_id": uid,
                    "score_ts": ts,
                    "sequence_bytes": _seq_cctx.compress(raw_bytes),
                    "sequence_format": "arrow_ipc",
                    "compressed_bytes": len(raw_bytes),
                }
            )
            feat_buf.append(
                {
                    "score_id": score_id,
                    "user_id": uid,
                    "score_ts": ts,
                    "feature_bytes": feature_bytes,
                }
            )

            if len(raw_buf) >= FLUSH_SIZE:
                _flush_parquet(raw_buf, feat_buf, raw_seq_dir, features_dir)
                raw_buf.clear()
                feat_buf.clear()
                metrics["parquet_flushes"] = metrics.get("parquet_flushes", 0) + 1

    def _flush_parquet(raw_buf, feat_buf, raw_dir, feat_dir):
        import datetime
        import os
        import uuid as _u

        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        hour_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H")
        file_id = str(_u.uuid4())[:8]

        rdir = os.path.join(raw_dir, date_str, hour_str)
        os.makedirs(rdir, exist_ok=True)
        tbl = pa.Table.from_pylist(raw_buf)
        pq.write_table(tbl, os.path.join(rdir, f"seq_{file_id}.parquet"), compression="zstd")

        fdir = os.path.join(feat_dir, date_str, hour_str)
        os.makedirs(fdir, exist_ok=True)
        tbl = pa.Table.from_pylist(feat_buf)
        pq.write_table(tbl, os.path.join(fdir, f"feat_{file_id}.parquet"), compression="zstd")

        metrics["parquet_rows"] = metrics.get("parquet_rows", 0) + len(raw_buf)

    freeze_enabled = os.environ.get("FREEZE_ENABLED", "0") == "1"
    if freeze_enabled:
        _pq_thread = threading.Thread(target=_parquet_writer, daemon=True, name="parquet_writer")
        _pq_thread.start()
        log.info(f"Parquet writer started → {raw_seq_dir} + {features_dir}")
    else:
        log.info(
            "Parquet freeze DISABLED (set FREEZE_ENABLED=1 to re-enable raw-sequence freezing)"
        )

    log.info(f"Starting prefetcher loop (batch_size={batch_size}, seq_len={seq_len})")
    batches_produced = 0

    while True:
        FILL_TIMEOUT_S = 0.2
        deadline = time.time() + FILL_TIMEOUT_S
        entries = []
        redis_err = None
        while True:
            n_remain = batch_size - len(entries)
            if n_remain <= 0:
                break
            try:
                more = r.zpopmin(redis_key, count=n_remain)
            except Exception as e:
                redis_err = e
                break
            if more:
                entries.extend(more)
                if len(entries) >= batch_size:
                    break
            else:
                if time.time() >= deadline:
                    break
                time.sleep(0.005)
                continue
            if time.time() >= deadline:
                break
        if redis_err is not None:
            log.warning(f"Redis ZPOPMIN failed: {redis_err}")
            time.sleep(1)
            try:
                r = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)
                r.ping()
                log.info("Redis reconnected")
            except Exception:
                pass
            continue

        if not entries:
            time.sleep(0.5)
            continue

        def _split_member(member_str: str):
            if ":" in member_str:
                a, b = member_str.split(":", 1)
                return int(a), int(b)
            return int(member_str), 0

        parsed = [_split_member(m) for m, _ in entries]
        trigger_by_uid: dict[int, int] = {uid: tid for uid, tid in parsed if tid}

        if classifier_shadow_enabled and scoring_work_redis is not None:
            try:
                shadow_payloads = []
                for uid, tid in parsed:
                    if tid:
                        shadow_payloads.append(f"{uid}:{tid}")
                    else:
                        shadow_payloads.append(str(uid))
                if shadow_payloads:
                    scoring_work_redis.rpush(classifier_shadow_redis_key, *shadow_payloads)
            except Exception as _cse:
                _cnt = scoring_work_metrics.setdefault("classifier_shadow_err", 0) + 1
                scoring_work_metrics["classifier_shadow_err"] = _cnt
                if _cnt % 100 == 1:
                    log.warning(f"classifier_shadow rpush err: {_cse!r} (count={_cnt})")

        raw_uid_score = [
            (uid, float(score)) for (uid, _tid), (_member, score) in zip(parsed, entries)
        ]
        raw_uids = [u for (u, _s) in raw_uid_score]
        score_by_uid = dict(raw_uid_score)

        if _trace.enabled:
            traced_uids_in_pop = _trace.is_active_many(raw_uids)
            for uid in traced_uids_in_pop:
                _trace.emit(
                    stage="prefetcher_pop",
                    uid=uid,
                    decision="popped",
                    correlator=score_by_uid[uid],
                )
        else:
            traced_uids_in_pop = set()

        try:
            pipe = r.pipeline(transaction=False)
            for uid in raw_uids:
                pipe.get(f"scored:{uid}")
            gate_results = pipe.execute()
            uids = [
                uid
                for uid, scored in zip(raw_uids, gate_results)
                if scored is None or scored == "leased"
            ]
            metrics["gate_skips"] = metrics.get("gate_skips", 0) + (len(raw_uids) - len(uids))
        except Exception:
            uids = raw_uids

        if not uids:
            continue

        t0 = time.time()

        futures = [fetch_pool.submit(_fetch, uid) for uid in uids]
        user_rows, uid_list = [], []
        traced_fetch_results = []
        for f in futures:
            uid, raw, cache_hit = f.result()
            n_bytes = len(raw) if raw else 0
            if raw and len(raw) > 100:
                user_rows.append((uid, raw))
                uid_list.append(uid)
                if uid in traced_uids_in_pop:
                    traced_fetch_results.append((uid, "cache_hit" if cache_hit else "ok", n_bytes))
            else:
                metrics["fetch_misses"] += 1
                if uid in traced_uids_in_pop:
                    traced_fetch_results.append((uid, "lt_100_bytes_drop", n_bytes))

        if _trace.enabled and traced_fetch_results:
            for uid, decision, n_bytes in traced_fetch_results:
                _trace.emit(
                    stage="prefetcher_mh_fetch",
                    uid=uid,
                    decision=decision,
                    correlator=score_by_uid.get(uid),
                    detail={"bytes": n_bytes},
                )

        B = len(user_rows)
        if B == 0:
            continue

        fetch_ms = (time.time() - t0) * 1000

        t1 = time.time()
        N_CHUNKS = 4
        chunk_size = max(1, len(user_rows) // N_CHUNKS)
        chunks = [user_rows[i : i + chunk_size] for i in range(0, len(user_rows), chunk_size)]

        if _has_rust:
            from concurrent.futures import ThreadPoolExecutor as _TP

            with _TP(max_workers=N_CHUNKS, thread_name_prefix="extract") as pool:
                chunk_results = list(
                    pool.map(
                        lambda c: _rust.extract_batch(
                            c,
                            seq_len=seq_len,
                            n_action_types=n_action_types,
                            n_labels=N_MODEL_LABELS,
                        ),
                        chunks,
                    )
                )
            import numpy as _np

            def _cat(arrays):
                return _np.concatenate(arrays, axis=0)

            first = chunk_results[0]
            merged = {
                "user_ids_raw": _cat([r["user_ids_raw"] for r in chunk_results]),
                "tweet_ids_raw": _cat([r["tweet_ids_raw"] for r in chunk_results]),
                "author_ids_raw": _cat([r["author_ids_raw"] for r in chunk_results]),
                "action_seq": {
                    k: _cat([r["action_seq"][k] for r in chunk_results])
                    for k in first["action_seq"]
                },
                "user_features": {
                    k: _cat([r["user_features"][k] for r in chunk_results])
                    for k in first["user_features"]
                },
                "label_vector": _cat([r["label_vector"] for r in chunk_results]),
                "label_mask": _cat([r["label_mask"] for r in chunk_results]),
            }
        else:
            from manhattan_arrow_adapter import arrow_bytes_to_batch

            single_batches = []
            for u, raw in user_rows:
                try:
                    b, _ = arrow_bytes_to_batch(
                        [(u, raw)], seq_len=seq_len, n_action_types=n_action_types
                    )
                    single_batches.append(b)
                except Exception:
                    pass
            if not single_batches:
                continue
            from manhattan_scorer import _merge_single_user_batches

            merged = _merge_single_user_batches(single_batches, seq_len)
            B = len(single_batches)
            uid_list = uid_list[:B]

        import numpy as _np

        try:
            from proto_gen.recsys_pb2 import ActionName as _AN

            _PROTO_IDX_TO_NAME = {v: k for k, v in _AN.items() if v > 0}
        except ImportError:
            _PROTO_IDX_TO_NAME = {}

        action_types_arr = merged["action_seq"]["action_types"]
        padding_mask_arr = merged["action_seq"]["padding_mask"]

        _NAME_TO_PROTO_IDX = {v: k for k, v in _PROTO_IDX_TO_NAME.items()}
        _app_id_arr = merged["action_seq"].get("client_app_id")
        _dom_client_masks = {}
        if _app_id_arr is not None:
            _pm_bool = padding_mask_arr.astype(bool)
            for _key, _cname in (
                ("create", "SERVER_TWEET_CREATE"),
                ("reply", "SERVER_TWEET_REPLY"),
            ):
                _cidx = _NAME_TO_PROTO_IDX.get(_cname)
                if _cidx is not None and _cidx < action_types_arr.shape[2]:
                    _dom_client_masks[_key] = _pm_bool & action_types_arr[:, :, _cidx].astype(bool)

        batch_histograms = []
        batch_behavioral_summaries = []
        for i in range(B):
            mask = padding_mask_arr[i].astype(bool)
            n_valid = int(mask.sum())
            col_sums = action_types_arr[i][mask].sum(axis=0)
            counts = []
            for proto_idx in range(len(col_sums)):
                if col_sums[proto_idx] > 0:
                    name = _PROTO_IDX_TO_NAME.get(proto_idx)
                    if name:
                        counts.append((name, int(col_sums[proto_idx])))
            batch_histograms.append(counts)

            summary = {}
            aseq = merged["action_seq"]

            try:
                valid_actions = action_types_arr[i][mask]
                dominant_idxs = valid_actions.argmax(axis=1)
                time_deltas = None
                if "continuous_actions" in aseq and aseq["continuous_actions"].shape[-1] >= 1:
                    time_deltas = aseq["continuous_actions"][i][mask][:, 0]
                tl_parts = []
                if n_valid > 0:
                    cum_time = 0.0
                    cur_name = _PROTO_IDX_TO_NAME.get(int(dominant_idxs[0]), f"A{dominant_idxs[0]}")
                    cur_count = 1
                    cur_time = 0.0
                    for j in range(1, n_valid):
                        if time_deltas is not None:
                            cum_time += float(time_deltas[j])
                        act_name = _PROTO_IDX_TO_NAME.get(
                            int(dominant_idxs[j]), f"A{dominant_idxs[j]}"
                        )
                        if act_name == cur_name:
                            cur_count += 1
                        else:
                            tl_parts.append(f"T+{int(cur_time)}s:{cur_name}x{cur_count}")
                            cur_name = act_name
                            cur_count = 1
                            cur_time = cum_time
                    tl_parts.append(f"T+{int(cur_time)}s:{cur_name}x{cur_count}")
                summary["decoded_timeline"] = " -> ".join(tl_parts)[:5000]
            except Exception:
                summary["decoded_timeline"] = ""

            try:
                _PLATFORM_NAMES = {0: "unknown", 1: "iOS", 2: "Android", 3: "Web", 4: "API"}
                plat_arr = aseq["client_platform"][i][mask]
                plat_counts = {}
                for p in plat_arr:
                    pname = _PLATFORM_NAMES.get(int(p), f"p{int(p)}")
                    plat_counts[pname] = plat_counts.get(pname, 0) + 1
                plat_sorted = sorted(plat_counts.items(), key=lambda x: -x[1])
                summary["platform_distribution"] = ",".join(
                    f"{name}:{cnt * 100 // n_valid}%" for name, cnt in plat_sorted if cnt > 0
                )
            except Exception:
                summary["platform_distribution"] = ""

            try:
                trans_arr = aseq["action_transition"][i][mask]
                bigram_counts = {}
                for j in range(1, n_valid):
                    a_prev = _PROTO_IDX_TO_NAME.get(
                        int(dominant_idxs[j - 1]), f"A{dominant_idxs[j - 1]}"
                    )
                    a_curr = _PROTO_IDX_TO_NAME.get(int(dominant_idxs[j]), f"A{dominant_idxs[j]}")
                    bigram = f"{a_prev}->{a_curr}"
                    bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
                top_bigrams = sorted(bigram_counts.items(), key=lambda x: -x[1])[:10]
                summary["top_action_transitions"] = ",".join(
                    f"{bg}:{cnt}" for bg, cnt in top_bigrams
                )
            except Exception:
                summary["top_action_transitions"] = ""

            try:
                ewi = aseq["engagement_without_impression"][i][mask]
                ewi_count = int(ewi.sum())
                summary["engagement_without_impression"] = ewi_count
                summary["engagement_without_impression_pct"] = (
                    round(ewi_count / n_valid * 100, 1) if n_valid else 0
                )
            except Exception:
                summary["engagement_without_impression"] = 0
                summary["engagement_without_impression_pct"] = 0.0

            try:
                if time_deltas is not None and len(time_deltas) > 1:
                    valid_deltas = time_deltas[time_deltas > 0]
                    if len(valid_deltas) > 0:
                        summary["median_gap_sec"] = round(float(_np.median(valid_deltas)), 1)
                        summary["time_span_sec"] = round(float(_np.sum(time_deltas)), 0)
                    else:
                        summary["median_gap_sec"] = 0
                        summary["time_span_sec"] = 0
                else:
                    summary["median_gap_sec"] = 0
                    summary["time_span_sec"] = 0
            except Exception:
                summary["median_gap_sec"] = 0
                summary["time_span_sec"] = 0

            total_acts = sum(c for _, c in counts) if counts else 0
            server_acts = sum(c for n, c in counts if n.startswith("SERVER_"))
            summary["server_pct"] = round(server_acts / total_acts * 100, 1) if total_acts else 0
            summary["dwell_count"] = sum(c for n, c in counts if "DWELLED" in n and "NOT" not in n)
            summary["render_count"] = sum(
                c for n, c in counts if "NOT_DWELLED" in n or "RENDER" in n
            )

            try:
                for _key, _amask_b in _dom_client_masks.items():
                    _amask = _amask_b[i]
                    _n_act = int(_amask.sum())
                    summary[f"n_{_key}"] = _n_act
                    if _n_act == 0:
                        continue
                    _apps = _app_id_arr[i][_amask]
                    _apps = _apps[_apps > 0]
                    if _apps.size:
                        _vals, _cnts = _np.unique(_apps, return_counts=True)
                        _k = int(_cnts.argmax())
                        summary[f"dom_{_key}_app_id"] = int(_vals[_k])
                        summary[f"dom_{_key}_app_frac"] = round(float(_cnts[_k]) / _n_act, 4)
            except Exception:
                pass

            batch_behavioral_summaries.append(summary)
            if i == 0:
                log.info(
                    f"[BS-DEBUG] user={uid_list[0]} summary_keys={list(summary.keys())} timeline_len={len(summary.get('decoded_timeline', ''))} platform={summary.get('platform_distribution', '?')}"
                )

        merged.pop("user_ids_raw", None)
        merged.pop("tweet_ids_raw", None)
        merged.pop("author_ids_raw", None)

        batch_score_ids = [str(_uuid.uuid4()) for _ in range(B)]

        if freeze_enabled:
            for i in range(B):
                raw_bytes_i = user_rows[i][1]
                feat_bytes_i = pickle.dumps(
                    {
                        k: (
                            merged[k][i : i + 1] if hasattr(merged[k], "__getitem__") else merged[k]
                        )
                        for k in merged
                        if k not in ("action_seq", "user_features")
                    },
                    protocol=5,
                )
                try:
                    _parquet_queue.put_nowait(
                        (uid_list[i], raw_bytes_i, feat_bytes_i, batch_score_ids[i])
                    )
                except queue.Full:
                    metrics["parquet_drops"] = metrics.get("parquet_drops", 0) + 1

        if batch_size > B:
            merged = _pad_batch(merged, batch_size)

        extract_ms = (time.time() - t1) * 1000

        t2 = time.time()
        message_bytes = serialize_batch(
            merged,
            uid_list,
            B,
            score_ids=batch_score_ids,
            action_histograms=batch_histograms,
            behavioral_summaries=batch_behavioral_summaries,
            priority_lane=priority_lane,
        )
        serialize_ms = (time.time() - t2) * 1000

        def _delivery_report(err, msg):
            if err:
                log.warning(f"Kafka delivery failed: {err}")
                metrics["kafka_errors"] += 1
            else:
                metrics["kafka_delivered"] += 1

        try:
            producer.produce(
                kafka_topic,
                value=message_bytes,
                callback=_delivery_report,
            )
        except Exception as _kp_e:
            metrics["kafka_errors"] = metrics.get("kafka_errors", 0) + 1
            if metrics["kafka_errors"] % 1000 == 1:
                log.warning(
                    f"Kafka produce raised {_kp_e!r}; suppressing (count={metrics['kafka_errors']})"
                )
        producer.poll(0)

        if scoring_work_redis is not None:
            try:
                scoring_work_redis.rpush(scoring_work_redis_key, message_bytes)
                scoring_work_metrics["rpush_ok"] += 1
            except redis_lib.exceptions.OutOfMemoryError:
                scoring_work_metrics["rpush_oom"] += 1
                if scoring_work_metrics["rpush_oom"] % 100 == 1:
                    log.warning(
                        f"SCORING_WORK_REDIS OOM "
                        f"(count={scoring_work_metrics['rpush_oom']}); "
                        "GPU consumers behind"
                    )
            except Exception as _e:
                scoring_work_metrics["rpush_err"] += 1
                if scoring_work_metrics["rpush_err"] % 100 == 1:
                    log.warning(
                        f"SCORING_WORK_REDIS rpush error: {_e} "
                        f"(count={scoring_work_metrics['rpush_err']})"
                    )

        batches_produced += 1
        total_ms = (time.time() - t0) * 1000
        metrics["batches_produced"] += 1
        metrics["users_prepared"] += B

        if _trace.enabled and traced_uids_in_pop:
            batch_id = f"{kafka_topic}:{batches_produced}"
            for uid in uid_list:
                if uid in traced_uids_in_pop:
                    _trace.emit(
                        stage="prefetcher_batch_publish",
                        uid=uid,
                        decision="kafka_ok",
                        correlator=score_by_uid.get(uid),
                        batch_id=batch_id,
                        detail={"batch_size": B, "msg_bytes": len(message_bytes)},
                    )

        if batches_produced % 10 == 0:
            elapsed = time.time() - metrics["_start_time"]
            log.info(
                f"[Prefetcher] batch={batches_produced} | {B} users in {total_ms:.0f}ms "
                f"(fetch={fetch_ms:.0f} extract={extract_ms:.0f} serialize={serialize_ms:.0f}) | "
                f"{len(message_bytes) / 1024 / 1024:.1f}MB | "
                f"total: {metrics['users_prepared']:,} users ({metrics['users_prepared'] / elapsed:.0f}/sec) | "
                f"kafka_ok={metrics['kafka_delivered']} kafka_err={metrics['kafka_errors']} "
                f"redis_ok={scoring_work_metrics['rpush_ok']} "
                f"redis_oom={scoring_work_metrics['rpush_oom']} "
                f"redis_err={scoring_work_metrics['rpush_err']} "
                f"fetch_miss={metrics['fetch_misses']} "
                f"cache_hit={metrics.get('cache_hits', 0)} "
                f"cache_miss={metrics.get('cache_misses', 0)} "
                f"cache_err={metrics.get('cache_errors', 0)} "
                f"rt_users={metrics.get('rt_merged_users', 0)} "
                f"rt_actions={metrics.get('rt_merged_actions', 0)} "
                f"rt_err={metrics.get('rt_errors', 0)} "
                f"pq_rows={metrics.get('parquet_rows', 0)} pq_drops={metrics.get('parquet_drops', 0)}"
            )

        if batches_produced % 50 == 0:
            producer.flush(timeout=5.0)


def start_health_server(metrics: dict, port: int):
    deployment = os.environ.get("MH_METRICS_DEPLOYMENT", "batch-prefetcher")
    app_id = os.environ.get("MH_APP_ID", "your-app")
    dataset = os.environ.get("MH_DATASET", "your-dataset")
    cluster = os.environ.get("MH_CLUSTER", "your-cluster")

    def _render_prom() -> str:
        lbl = f'deployment="{deployment}",cluster="{cluster}",app_id="{app_id}",dataset="{dataset}"'
        out = [
            "# HELP safety_mh_get_total Manhattan Gets issued by this prefetcher, by result.",
            "# TYPE safety_mh_get_total counter",
            f'safety_mh_get_total{{{lbl},result="hit"}} {metrics.get("users_prepared", 0)}',
            f'safety_mh_get_total{{{lbl},result="miss"}} {metrics.get("fetch_misses", 0)}',
            "# HELP safety_prefetcher_batches_total GPU-ready batches produced.",
            "# TYPE safety_prefetcher_batches_total counter",
            f"safety_prefetcher_batches_total{{{lbl}}} {metrics.get('batches_produced', 0)}",
            "# HELP safety_prefetcher_kafka_delivered_total Kafka batches delivered.",
            "# TYPE safety_prefetcher_kafka_delivered_total counter",
            f"safety_prefetcher_kafka_delivered_total{{{lbl}}} {metrics.get('kafka_delivered', 0)}",
            "# HELP safety_seq_cache_get_total Historical sequence cache Gets, by result.",
            "# TYPE safety_seq_cache_get_total counter",
            f'safety_seq_cache_get_total{{{lbl},result="hit"}} {metrics.get("cache_hits", 0)}',
            f'safety_seq_cache_get_total{{{lbl},result="miss"}} {metrics.get("cache_misses", 0)}',
            f'safety_seq_cache_get_total{{{lbl},result="error"}} {metrics.get("cache_errors", 0)}',
            "# HELP safety_realtime_merge_total Realtime cache merge outcomes.",
            "# TYPE safety_realtime_merge_total counter",
            f'safety_realtime_merge_total{{{lbl},result="users_merged"}} {metrics.get("rt_merged_users", 0)}',
            f'safety_realtime_merge_total{{{lbl},result="actions_merged"}} {metrics.get("rt_merged_actions", 0)}',
            f'safety_realtime_merge_total{{{lbl},result="error"}} {metrics.get("rt_errors", 0)}',
        ]
        return "\n".join(out) + "\n"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/metrics":
                body = _render_prom().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                return
            body = (
                f"batches_produced={metrics['batches_produced']}\n"
                f"users_prepared={metrics['users_prepared']}\n"
                f"kafka_delivered={metrics['kafka_delivered']}\n"
                f"kafka_errors={metrics['kafka_errors']}\n"
                f"fetch_misses={metrics['fetch_misses']}\n"
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *_):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info(f"Health + Prometheus /metrics endpoint on :{port}")


def main():
    parser = argparse.ArgumentParser(description="Abuse batch prefetcher (CPU)")
    parser.add_argument("--redis-host", default="abuse-score-redis")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-key", default="abuse:score_queue")
    parser.add_argument("--mh-endpoint", default="localhost:8080")
    parser.add_argument(
        "--no-seq-cache",
        action="store_true",
        help="Disable the cache-first sequence lookup (STEP 1).",
    )
    parser.add_argument("--seq-cache-host", default=os.environ.get("SEQ_CACHE_HOST", "localhost"))
    parser.add_argument(
        "--seq-cache-port", type=int, default=int(os.environ.get("SEQ_CACHE_PORT", "6379"))
    )
    parser.add_argument(
        "--seq-cache-ttl-secs", type=int, default=4 * 3600, help="TTL for read-through cache fills."
    )
    parser.add_argument(
        "--no-seq-cache-read-through",
        action="store_true",
        help="Do not SETEX Manhattan results back into the cache.",
    )
    parser.add_argument(
        "--realtime-cache-enabled",
        action="store_true",
        default=os.environ.get("REALTIME_CACHE_ENABLED", "0") == "1",
        help="Merge realtime cache actions into each sequence (STEP 2).",
    )
    parser.add_argument(
        "--realtime-cache-host", default=os.environ.get("REALTIME_CACHE_HOST", "localhost")
    )
    parser.add_argument(
        "--realtime-cache-port",
        type=int,
        default=int(os.environ.get("REALTIME_CACHE_PORT", "6382")),
    )
    parser.add_argument(
        "--realtime-headroom-ms",
        type=int,
        default=60_000,
        help="Read-window headroom before the sequence watermark.",
    )
    parser.add_argument("--kafka-bootstrap", default="localhost:9092")
    parser.add_argument("--kafka-username", default="kafka-user")
    parser.add_argument("--kafka-password", required=True)
    parser.add_argument("--kafka-topic", default="abuse_ready_batches")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-action-types", type=int, default=256)
    parser.add_argument("--health-port", type=int, default=8081)
    parser.add_argument(
        "--priority-lane", action="store_true", help="Tag every output batch as priority_lane=True."
    )
    args = parser.parse_args()

    metrics = {
        "batches_produced": 0,
        "users_prepared": 0,
        "kafka_delivered": 0,
        "kafka_errors": 0,
        "fetch_misses": 0,
        "_start_time": time.time(),
    }

    start_health_server(metrics, args.health_port)

    prefetcher_loop(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_key=args.redis_key,
        mh_endpoint=args.mh_endpoint,
        kafka_bootstrap=args.kafka_bootstrap,
        kafka_username=args.kafka_username,
        kafka_password=args.kafka_password,
        kafka_topic=args.kafka_topic,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_action_types=args.n_action_types,
        metrics=metrics,
        priority_lane=args.priority_lane,
        seq_cache_enabled=not args.no_seq_cache,
        seq_cache_host=args.seq_cache_host,
        seq_cache_port=args.seq_cache_port,
        seq_cache_ttl_secs=args.seq_cache_ttl_secs,
        seq_cache_read_through=not args.no_seq_cache_read_through,
        realtime_cache_enabled=args.realtime_cache_enabled,
        realtime_cache_host=args.realtime_cache_host,
        realtime_cache_port=args.realtime_cache_port,
        realtime_headroom_ms=args.realtime_headroom_ms,
    )


if __name__ == "__main__":
    main()
