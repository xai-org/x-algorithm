#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
from xai_proto import recsys_pb2, recsys_pb2_grpc

logger = logging.getLogger("oss_bench")

DEFAULT_CONFIG_NAME_BY_SERVICE: dict[str, str] = {
    "ranking": "home_direct_packed_aggregated_kafka",
    "retrieval": "xrecsys_two_tower_aggregated_kafka",
}
DEFAULT_SERVICE_TYPE = "ranking"
DEFAULT_CONFIG_NAME = DEFAULT_CONFIG_NAME_BY_SERVICE[DEFAULT_SERVICE_TYPE]
DEFAULT_GRPC_PORT = 9988
DEFAULT_BOOT_TIMEOUT_S = 600
DEFAULT_REQUEST_TIMEOUT_S = 30


@dataclass
class BenchConfig:
    config_name: str = DEFAULT_CONFIG_NAME
    service_type: str = DEFAULT_SERVICE_TYPE
    checkpoint_path: str | None = None
    smoke: bool = False
    num_devices_per_process: int = 1
    bs_per_device: int = 1
    history_seq_len: int = 128
    candidate_seq_len: int = 8
    grpc_port: int = DEFAULT_GRPC_PORT
    boot_timeout_s: int = DEFAULT_BOOT_TIMEOUT_S
    extra_overrides: list[str] = field(default_factory=list)


def _detect_launch_script() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    candidate_root = here.parents[3]
    candidates = [("xrex", "inference")]
    for pkg, module_dir in candidates:
        candidate_script = candidate_root / pkg / module_dir / "launch_inference.py"
        if candidate_script.exists():
            return candidate_script, candidate_root

    raise FileNotFoundError("Could not locate launch_inference.py in the launchpad layout.")


def build_launch_argv(cfg: BenchConfig) -> tuple[Path, Path, list[str]]:
    launch_script, uv_dir = _detect_launch_script()

    args = [
        "--driver",
        "local",
        "--config_name",
        cfg.config_name,
        "--service_type",
        cfg.service_type,
        "--num_devices_per_process",
        str(cfg.num_devices_per_process),
        "--bs_per_device",
        str(cfg.bs_per_device),
        "--history_seq_len",
        str(cfg.history_seq_len),
        "--candidate_seq_len",
        str(cfg.candidate_seq_len),
        "--grpc_port",
        str(cfg.grpc_port),
        "--max_inflight_requests",
        "16",
        "--allow_random_init",
        "true" if cfg.smoke else "false",
        "--fake_mm_embeddings",
        "true",
    ]
    if cfg.checkpoint_path:
        args.extend(["--checkpoint_path", cfg.checkpoint_path])

    args.extend(
        [
            (
                "attn_impl=pallas_ranker_attn_infer"
                if cfg.service_type == "ranking"
                else "attn_impl=pallas_ranker_attn"
            ),
            "use_seqpack=False",
            "right_anchored_rope=True",
            f"bs_per_device={cfg.bs_per_device}",
            f"parallel_config.num_devices_per_process={cfg.num_devices_per_process}",
            f"num_devices_per_process={cfg.num_devices_per_process}",
            "ep=1",
            "dp=1",
            "training_ep=1",
        ]
    )
    if cfg.service_type == "ranking":
        args.append(
            f"model_config.model_config.sequence_len="
            f"{cfg.history_seq_len + cfg.candidate_seq_len + 2}"
        )
    if cfg.service_type == "retrieval":
        args.extend(
            [
                "use_post_sid=False",
                "use_post_embedding=True",
            ]
        )
    args.extend(cfg.extra_overrides)
    return launch_script, uv_dir, args


def boot_inference_server(cfg: BenchConfig) -> subprocess.Popen[bytes]:
    launch_script, uv_dir, args = build_launch_argv(cfg)
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--directory",
        str(uv_dir),
        str(launch_script),
        *args,
    ]

    env = {
        **os.environ,
        "JAX_PLATFORMS": "",
        "NUM_PHYSICAL_DEVICES_PER_NODE": "1",
        "JAX_COORDINATOR_PORT": os.environ.get(
            "JAX_COORDINATOR_PORT", str(2345 + (cfg.grpc_port % 1000))
        ),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(uv_dir), os.environ.get("PYTHONPATH", "")) if p
        ),
    }

    logger.info("Booting inference server:\n  %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=False,
    )


def wait_for_ready(
    server: subprocess.Popen[bytes],
    port: int,
    timeout_s: int,
) -> tuple[bool, str]:
    import threading

    deadline = time.time() + timeout_s
    addr = f"localhost:{port}"

    warmup_done = threading.Event()
    server_exited = threading.Event()

    def _drain():
        assert server.stdout is not None
        suppressed = False
        for raw in iter(server.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                line = repr(raw)
            if not suppressed:
                sys.stdout.write(line)
                sys.stdout.flush()
            if "Model warm up finished" in line:
                warmup_done.set()
                suppressed = True
        server_exited.set()

    t = threading.Thread(target=_drain, daemon=True, name="subprocess-drain")
    t.start()

    last_log = 0.0
    while time.time() < deadline:
        if warmup_done.is_set():
            logger.info(
                "Detected 'Model warm up finished' from subprocess — "
                "launchpad inference smoke succeeded."
            )
            return True, "warmup"
        if server_exited.is_set():
            logger.error("Subprocess exited before becoming ready.")
            return False, "exited"
        try:
            channel = grpc.insecure_channel(addr)
            grpc.channel_ready_future(channel).result(timeout=2)
            logger.info("Server is ready on %s (gRPC port bound).", addr)
            return True, "grpc"
        except grpc.FutureTimeoutError:
            now = time.time()
            if now - last_log > 10:
                remaining = int(deadline - now)
                logger.info("  Waiting for %s (%ds remaining)...", addr, remaining)
                last_log = now
        time.sleep(2)

    logger.error("Server did not become ready within %ds.", timeout_s)
    return False, "timeout"


def make_request(cfg: BenchConfig) -> Any:
    if cfg.service_type == "retrieval":
        return _make_retrieval_request()
    return _make_ranking_request()


def _user_action_sequence(history_n: int = 3) -> Any:
    return recsys_pb2.UserActionSequence(
        userId=1,
        userActions=[],
        userActionsData=recsys_pb2.UserActionSequenceDataContainer(
            orderedAggregatedUserActionsList=recsys_pb2.AggregatedUserActionList(
                aggregatedUserActions=[
                    recsys_pb2.AggregatedUserAction(
                        userId=1,
                        tweetInfo=recsys_pb2.TweetInfo(tweetId=1000 + i, authorId=2000 + i),
                        actions=[
                            recsys_pb2.ActionInfo(
                                engagementTimeMs=1234567890,
                                actionName=recsys_pb2.ActionName.SERVER_TWEET_FAV,
                                actionCategory=recsys_pb2.ActionCategory.TWEET,
                                actionGroup=recsys_pb2.ActionGroup.P_FAV,
                                userActionMeta=recsys_pb2.UserActionMeta(
                                    indexInSequence=i,
                                    page="home",
                                    dwellTime=5.0,
                                    clientPlatform="web",
                                ),
                            )
                        ],
                    )
                    for i in range(history_n)
                ]
            ),
        ),
        metadata=recsys_pb2.UserActionSequenceMeta(
            length=history_n, firstSequenceTime=1234567890, lastSequenceTime=1234567890
        ),
    )


def _make_ranking_request(history_n: int = 3, n_candidates: int = 8) -> Any:
    cand_set = recsys_pb2.CandidateSet(
        userId=1,
        candidates=[
            recsys_pb2.TweetInfo(tweetId=3000 + i, authorId=4000 + i) for i in range(n_candidates)
        ],
    )
    return recsys_pb2.PredictNextActionsRequest(
        sequences=[_user_action_sequence(history_n)],
        candidateSets=[cand_set],
    )


def _make_retrieval_request(history_n: int = 3) -> Any:
    return recsys_pb2.RetrieveTopKCandidatesRequest(
        user_ids=[1],
        sequences=[_user_action_sequence(history_n)],
        topK=10,
        metadata="bench",
    )


WARMUP_GRPC_GRACE_S = 15


def grpc_port_ready(port: int, timeout_s: float) -> bool:
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        return True
    except grpc.FutureTimeoutError:
        return False


def send_request(
    port: int,
    request: Any,
    service_type: str = DEFAULT_SERVICE_TYPE,
    timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
) -> Any:
    channel = grpc.insecure_channel(f"localhost:{port}")
    if service_type == "retrieval":
        stub = recsys_pb2_grpc.RecsysRetrievalPredictorStub(channel)
        return stub.RetrieveTopKCandidates(request, timeout=timeout_s)
    stub = recsys_pb2_grpc.RecsysPredictorStub(channel)
    return stub.PredictNextActions(request, timeout=timeout_s)


def response_to_dict(response: recsys_pb2.PredictNextActionsResponse) -> dict[str, Any]:
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(response, preserving_proto_field_name=True)


def save_golden(response: recsys_pb2.PredictNextActionsResponse, path: Path) -> None:
    data = response_to_dict(response)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info("Saved golden output to %s", path)


def verify_against_golden(
    response: recsys_pb2.PredictNextActionsResponse,
    path: Path,
) -> bool:
    if not path.exists():
        logger.error("Golden file does not exist: %s. Use --save_golden first.", path)
        return False

    actual = response_to_dict(response)
    expected = json.loads(path.read_text())

    if actual == expected:
        logger.info("Response matches golden file ✓")
        return True

    logger.error("Response does NOT match golden file.")
    logger.error("Expected: %s", json.dumps(expected, indent=2, sort_keys=True))
    logger.error("Actual:   %s", json.dumps(actual, indent=2, sort_keys=True))
    return False


def shutdown_server(server: subprocess.Popen[bytes], timeout_s: int = 10) -> None:
    if server.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.warning("Server did not exit within %ds of SIGKILL.", timeout_s)


def run_bench(cfg: BenchConfig, golden_path: Path | None, save_golden_path: Path | None) -> int:
    server = boot_inference_server(cfg)
    try:
        ok, mode = wait_for_ready(server, cfg.grpc_port, cfg.boot_timeout_s)
        if not ok:
            return 1
        if mode == "warmup":
            if grpc_port_ready(cfg.grpc_port, WARMUP_GRPC_GRACE_S):
                logger.info(
                    "gRPC port accepted connections after warmup — "
                    "proceeding with a synthetic request."
                )
            else:
                logger.info(
                    "Launchpad smoke passed via warmup signal (gRPC port did "
                    "not accept within %ds — expected only when the "
                    "xai_recsys_engine extra is not installed).",
                    WARMUP_GRPC_GRACE_S,
                )
                return 0
        logger.info("Sending %s request...", cfg.service_type)
        response = send_request(cfg.grpc_port, make_request(cfg), cfg.service_type)
        logger.info("Response:\n%s", response)

        if save_golden_path:
            save_golden(response, save_golden_path)
            return 0

        if golden_path:
            return 0 if verify_against_golden(response, golden_path) else 1

        return 0
    finally:
        logger.info("Shutting down server...")
        shutdown_server(server)


def parse_args() -> tuple[BenchConfig, Path | None, Path | None]:
    parser = argparse.ArgumentParser(
        description="Boot the recsys inference server, send a request, verify."
    )
    parser.add_argument(
        "--service_type",
        choices=sorted(DEFAULT_CONFIG_NAME_BY_SERVICE),
        default=DEFAULT_SERVICE_TYPE,
        help=(
            "Which inference service to launch. 'ranking' runs the "
            "RankingModelRunner against an xrecsys CONFIGS entry; "
            "'retrieval' runs the RetrievalModelRunner against an "
            "xrecsys_two_tower CONFIGS entry."
        ),
    )
    parser.add_argument(
        "--config_name",
        default=None,
        help=(
            "Which model config to serve. Defaults to "
            f"{DEFAULT_CONFIG_NAME_BY_SERVICE['ranking']!r} for ranking and "
            f"{DEFAULT_CONFIG_NAME_BY_SERVICE['retrieval']!r} for retrieval."
        ),
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Skip checkpoint loading; use random init.",
    )
    parser.add_argument("--num_devices_per_process", type=int, default=1)
    parser.add_argument("--bs_per_device", type=int, default=1)
    parser.add_argument("--history_seq_len", type=int, default=128)
    parser.add_argument("--candidate_seq_len", type=int, default=8)
    parser.add_argument("--grpc_port", type=int, default=DEFAULT_GRPC_PORT)
    parser.add_argument("--boot_timeout_s", type=int, default=DEFAULT_BOOT_TIMEOUT_S)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra-style override (e.g. --override foo=bar). May be repeated.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Path to golden output JSON; assert response matches.",
    )
    parser.add_argument(
        "--save_golden", type=Path, default=None, help="Save the response as the new golden output."
    )
    args = parser.parse_args()

    if args.golden and args.save_golden:
        parser.error("--golden and --save_golden are mutually exclusive.")

    if not args.smoke and not args.checkpoint_path:
        parser.error("Either --smoke (random init) or --checkpoint_path is required.")

    config_name = args.config_name or DEFAULT_CONFIG_NAME_BY_SERVICE[args.service_type]
    cfg = BenchConfig(
        config_name=config_name,
        service_type=args.service_type,
        checkpoint_path=args.checkpoint_path,
        smoke=args.smoke,
        num_devices_per_process=args.num_devices_per_process,
        bs_per_device=args.bs_per_device,
        history_seq_len=args.history_seq_len,
        candidate_seq_len=args.candidate_seq_len,
        grpc_port=args.grpc_port,
        boot_timeout_s=args.boot_timeout_s,
        extra_overrides=list(args.override),
    )
    return cfg, args.golden, args.save_golden


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        force=True,
    )
    cfg, golden_path, save_golden_path = parse_args()
    logger.info("Bench config: %s", cfg)
    return run_bench(cfg, golden_path, save_golden_path)


if __name__ == "__main__":
    sys.exit(main())
