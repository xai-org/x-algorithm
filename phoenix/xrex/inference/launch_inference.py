# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import argparse
import importlib
import importlib.util
import logging
import os
import random
import shutil
import uuid

from xrex.configs.xrecsys import CONFIGS as model_cfgs
from xrex.configs.xrecsys_two_tower import CONFIGS as model_cfgs_two_tower
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.driver.config_factory import build_eval_configs
from xrex.driver.run import pin_visible_devices, run_driver_train
from xrex.inference import service_registry
from xrex.inference.model_runner import RankingModelRunner, RetrievalModelRunner
from xrex.models.recsys_model import RecsysAggregatedModelConfig
from xrex.models.recsys_two_tower_model import RecsysTwoTowerModelConfig
from xrex.utils import cluster

for _family in ("xrex.inference.gen_recs_services",):
    if importlib.util.find_spec(_family) is not None:
        importlib.import_module(_family)


logger = logging.getLogger(__name__)

log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger()
logger.setLevel(log_level)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y", "t"):
        return True
    if lowered in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'. Use true/false.")


def parse_bs_per_device_buckets(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--bs_per_device_buckets must list at least one size")
    sizes = set()
    for p in parts:
        try:
            n = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid bucket size '{p}' (must be an integer)")
        if n <= 0:
            raise argparse.ArgumentTypeError(f"Bucket size must be positive, got {n}")
        sizes.add(n)
    return tuple(sorted(sizes))


def _atomic_link_publish(final_path: str, val: bytes, pod: str) -> bool:
    if os.path.exists(final_path):
        return False
    d, base = os.path.split(final_path)
    tmp = os.path.join(d, f"{base}.tmp.{pod}.{os.getpid()}.{uuid.uuid4().hex}")
    created = False
    try:
        with open(tmp, "wb") as f:
            f.write(val)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, final_path)
            created = True
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return created


def _seed_local_from_nfs(local_dir: str, nfs_dir: str, pod: str) -> int:
    if not os.path.isdir(nfs_dir):
        return 0
    try:
        names = os.listdir(nfs_dir)
    except OSError:
        return 0
    seeded = 0
    for name in names:
        if ".tmp." in name:
            continue
        src = os.path.join(nfs_dir, name)
        dst = os.path.join(local_dir, name)
        if os.path.exists(dst):
            continue
        tmp = os.path.join(local_dir, f"{name}.tmp.{pod}.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            shutil.copyfile(src, tmp)
            try:
                os.link(tmp, dst)
                seeded += 1
            except FileExistsError:
                pass
        except OSError as e:
            logger.debug("JAX cache: seeding %s from NFS failed: %s", name, e)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return seeded


def _install_local_first_lru_cache_put(nfs_dir: str | None) -> None:
    from jax._src.lru_cache import _CACHE_SUFFIX, LRUCache

    if getattr(LRUCache.put, "_xai_local_first", False):
        return

    original_put = LRUCache.put
    pod = cluster.get_pod_name() or "unknownpod"

    def _local_first_put(self, key: str, val: bytes) -> None:
        if not key:
            raise ValueError("key cannot be empty")
        if self.eviction_enabled:
            original_put(self, key, val)
            return

        local_final = os.path.join(os.fspath(self.path), f"{key}{_CACHE_SUFFIX}")
        _atomic_link_publish(local_final, val, pod)

        if nfs_dir is not None:
            try:
                _atomic_link_publish(os.path.join(nfs_dir, f"{key}{_CACHE_SUFFIX}"), val, pod)
            except OSError as e:
                logger.debug("JAX cache: NFS mirror of %r failed: %s", key, e)

    _local_first_put._xai_local_first = True
    LRUCache.put = _local_first_put
    logger.info(
        "JAX persistent cache: installed local-first LRUCache.put (%s).",
        "local + NFS mirror" if nfs_dir is not None else "local-only, NFS unavailable",
    )


def _maybe_configure_jax_persistent_cache(args: argparse.Namespace) -> None:
    nfs_dir = (args.jax_compilation_cache_dir or "").strip()
    deployment_name = (args.deployment_name or "").strip()
    if not nfs_dir:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        logger.info(
            "JAX persistent compilation cache disabled "
            "(--jax_compilation_cache_dir is empty). "
            "Cold start will pay the full JIT compile cost."
        )
        return

    pod = cluster.get_pod_name() or "unknownpod"

    import jax

    local_dir = os.environ.get("JAX_COMPILATION_CACHE_LOCAL_DIR", "").strip()
    if not local_dir:
        local_dir = "/tmp/jax-cache/0"
    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError as e:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        logger.warning(
            "JAX persistent compilation cache: local dir %s is unusable (%s); "
            "falling back to the trainer's default per-process /tmp cache.",
            local_dir,
            e,
        )
        return

    nfs_ok = False
    probe_path = os.path.join(nfs_dir, f".write_probe_{os.getpid()}")
    try:
        os.makedirs(nfs_dir, exist_ok=True)
        with open(probe_path, "w") as f:
            f.write("ok")
        os.unlink(probe_path)
        nfs_ok = True
    except OSError as e:
        logger.warning(
            "JAX persistent compilation cache: shared NFS dir %s is unusable "
            "(%s); running with a local-only cache at %s.",
            nfs_dir,
            e,
            local_dir,
        )

    if nfs_ok:
        seeded = _seed_local_from_nfs(local_dir, nfs_dir, pod)
        logger.info(
            "JAX persistent compilation cache: seeded %d entries from NFS %s into local %s.",
            seeded,
            nfs_dir,
            local_dir,
        )

    os.environ["JAX_COMPILATION_CACHE_DIR"] = local_dir
    jax.config.update("jax_compilation_cache_dir", local_dir)
    min_compile_time_secs = int(args.jax_persistent_cache_min_compile_time_secs)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)

    _install_local_first_lru_cache_put(nfs_dir if nfs_ok else None)

    locked_keys = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
    )
    original_update = jax.config.update

    def _locked_update(name, value):
        if name in locked_keys:
            logger.info(
                "JAX persistent cache: ignoring later jax.config.update(%r, %r) -- "
                "key is locked by --jax_compilation_cache_dir.",
                name,
                value,
            )
            return None
        return original_update(name, value)

    jax.config.update = _locked_update

    logger.info(
        "JAX persistent compilation cache enabled: local=%s nfs=%s (%s) "
        "deployment=%s min_compile_time_secs=%d (config locked).",
        local_dir,
        nfs_dir,
        "mirrored" if nfs_ok else "unavailable, local-only",
        deployment_name or "<unset>",
        min_compile_time_secs,
    )


def run(
    args: argparse.Namespace,
):
    _maybe_configure_jax_persistent_cache(args)

    if args.allow_random_init:
        os.environ["DEBUG_ALLOW_RANDOM_INIT"] = "1"
    for apply_service_env in service_registry.SERVICE_ENV_HOOKS:
        apply_service_env(args.service_type)

    all_model_cfgs = {
        **model_cfgs,
        **model_cfgs_two_tower,
        **service_registry.EXTRA_MODEL_CFGS,
    }
    if args.config_name not in all_model_cfgs:
        raise KeyError(f"Unknown config_name: {args.config_name}")
    model = all_model_cfgs[args.config_name]
    model.checkpoint_config.no_loading = "in_out_embed.embeddings"

    BUCKET_SERVICE_TYPES = ("ranking", "retrieval")
    if args.service_type in BUCKET_SERVICE_TYPES:
        bs_per_device_buckets = (
            args.bs_per_device_buckets
            if args.bs_per_device_buckets is not None
            else tuple(range(1, args.bs_per_device + 1))
        )
    else:
        if args.bs_per_device_buckets is not None and len(args.bs_per_device_buckets) > 1:
            logger.warning(
                "--bs_per_device_buckets is only supported for --service_type in %s; "
                "ignoring %s for service_type=%s and using a single bucket "
                "(bs_per_device=%d).",
                BUCKET_SERVICE_TYPES,
                list(args.bs_per_device_buckets),
                args.service_type,
                args.bs_per_device,
            )
        bs_per_device_buckets = (args.bs_per_device,)
    max_bs_per_device = max(bs_per_device_buckets)

    model.bs_per_device = max_bs_per_device
    model.init_opt_state = False

    assert isinstance(
        model.model_config,
        (
            RecsysAggregatedModelConfig,
            RecsysTwoTowerModelConfig,
            *service_registry.EXTRA_CONFIG_TYPES,
        ),
    )
    model.model_config.embed_memory_kind = "unpinned_host"

    if args.restart_period is not None:
        restart_jitter = random.random() * args.max_restart_jitter
        restart_period = args.restart_period + restart_jitter
    else:
        restart_period = None

    if args.service_type == "ranking":
        model_runner_cls = RankingModelRunner
    elif args.service_type == "retrieval":
        model_runner_cls = RetrievalModelRunner
    elif args.service_type in service_registry.EXTRA_RUNNERS:
        model_runner_cls = service_registry.EXTRA_RUNNERS[args.service_type]
    else:
        raise ValueError(f"Invalid service type: {args.service_type}")

    for select_runner_cls in service_registry.RUNNER_CLASS_HOOKS:
        model_runner_cls = select_runner_cls(model_runner_cls, args)

    if args.service_type in service_registry.RETRIEVAL_DATASET_SERVICE_TYPES:
        assert args.retrieval_dataset_types, (
            f"--retrieval_dataset_types must be provided when running {args.service_type} inference"
        )

    num_devices_per_process = args.num_devices_per_process
    runner = model_runner_cls.from_trainer(
        t=model,
        inference_batch_size=max_bs_per_device * num_devices_per_process,
        history_seq_len=args.history_seq_len,
        candidate_seq_len=args.candidate_seq_len,
        copy_url=args.copy_url,
        grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
        readiness_port=args.readiness_port,
        max_inflight_requests=args.max_inflight_requests,
        restart_period=restart_period,
        graceful_restart_timeout=args.graceful_restart_timeout,
        max_inflight_backend_requests=args.max_inflight_backend_requests,
        large_k=args.large_k,
        debug_log_dir=args.debug_log_dir,
        jax_profile=args.jax_profile_path is not None,
        jax_profile_path=args.jax_profile_path,
        retrieval_dataset_types=tuple(
            RetrievalDataset[name] for name in args.retrieval_dataset_types
        ),
    )
    is_storage_path = False
    for resolve_checkpoint in service_registry.CHECKPOINT_RESOLVERS:
        is_storage_path = resolve_checkpoint(runner, args.checkpoint_path) or is_storage_path
    if not is_storage_path:
        runner.load = args.checkpoint_path
    runner.inference_batch_size_buckets = tuple(
        b * num_devices_per_process for b in bs_per_device_buckets
    )
    runner.channel_size = args.channel_size
    runner.enqueue_timeout_ms = args.enqueue_timeout_ms
    runner.queue_max_staleness_ms = args.queue_max_staleness_ms
    runner.enable_deadline_admission = args.enable_deadline_admission
    runner.deadline_margin_ms = args.deadline_margin_ms
    runner.deadline_post_ms = args.deadline_post_ms
    runner.deadline_fallback_budget_ms = args.deadline_fallback_budget_ms
    runner.service_time_ewma_alpha = args.service_time_ewma_alpha
    runner.use_pipelining = args.use_pipelining
    runner.use_pinned_d2h = args.use_pinned_d2h
    runner.pinned_d2h_num_buffers = args.pinned_d2h_num_buffers
    runner.embedding_gather_threads = args.embedding_gather_threads
    if hasattr(runner, "enable_bloom_filter"):
        runner.enable_bloom_filter = args.enable_bloom_filter
        runner.enable_topic_filter = args.enable_topic_filter
    runner.fake_mm_embeddings = args.fake_mm_embeddings
    runner.enable_async_response_compression = args.enable_async_response_compression
    runner.enable_hotswap = args.enable_hotswap
    runner.hotswap_download_rate_limit_gbps = args.hotswap_download_rate_limit_gib_per_s
    runner.hotswap_download_max_concurrent = args.hotswap_download_max_concurrent
    runner.hotswap_stage_rate_limit_gbps = args.hotswap_stage_rate_limit_gib_per_s
    runner.hotswap_stage_chunk_mib = args.hotswap_stage_chunk_mib
    if args.worker_id is not None:
        runner.worker_id = args.worker_id
    if args.num_workers is not None:
        runner.num_workers = args.num_workers
    runner.log_rotate = args.log_rotate
    runner.log_rotate_max_bytes = args.log_rotate_max_bytes
    runner.log_rotate_backup_count = args.log_rotate_backup_count
    if args.sid_endpoint is not None:
        runner.sid_endpoint = args.sid_endpoint
    if hasattr(runner, "beam_width") and args.beam_width != 1:
        runner.beam_width = args.beam_width
    if hasattr(runner, "decode_levels") and args.decode_levels != 0:
        runner.decode_levels = args.decode_levels
    if hasattr(runner, "num_generate") and args.num_generate > 1:
        runner.num_generate = args.num_generate
    runner.checkpoint_config.no_opt_state = True

    base_overrides = [
        f"num_devices_per_process={num_devices_per_process}",
        "no_opt_state=True",
        "verify_checksums=True",
        f"ep={num_devices_per_process}",
        "dp=1",
        "num_negatives_per_example=0",
        "num_global_negatives_per_example=0",
        f"history_seq_len={args.history_seq_len}",
        f"candidate_seq_len={args.candidate_seq_len}",
    ]
    if args.checkpoint_path and not is_storage_path:
        base_overrides.insert(0, f"load={args.checkpoint_path}")
    overrides = base_overrides + args.overrides
    if is_storage_path:
        for append_storage_overrides in service_registry.STORAGE_OVERRIDE_HOOKS:
            append_storage_overrides(args.checkpoint_path, overrides)
    driver_config, run_config = build_eval_configs(
        run_config=runner,
        overrides=overrides,
    )
    logger.info(f"{driver_config=}, {run_config=}")
    run_driver_train(driver_config, run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--history_seq_len", type=int, default=1023)
    parser.add_argument("--candidate_seq_len", type=int, default=1400)
    parser.add_argument(
        "--bs_per_device",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--bs_per_device_buckets",
        type=parse_bs_per_device_buckets,
        default=None,
        help=(
            "Comma-separated per-device batch sizes to pre-compile as buckets, "
            "e.g. '1,2,4,8,16,32'. One forward executable is compiled per bucket "
            "and each request is padded up to the smallest bucket that fits "
            "(requests above the largest bucket clamp to it). The largest bucket "
            "sizes the model buffers and the engine's max batch. Ranking & "
            "retrieval only: for those service types, unset defaults to one "
            "bucket per size from 1 to --bs_per_device; gen_recs ignores "
            "this and runs a single --bs_per_device batch."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="xrecsys_aggregated_kafka",
        help="which model config to serve",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
    )
    parser.add_argument(
        "--large_k",
        type=int,
        default=10000,
        help="Maximum number of top candidates to retrieve (default: 10000)",
    )
    parser.add_argument(
        "--copy_url",
        type=str,
        default="",
        help="address of a weight source, typically trainer",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=9988,
        help="main port for communication with clients",
    )
    parser.add_argument(
        "--metrics_port",
        type=int,
        default=9090,
        help="port for metrics",
    )
    parser.add_argument(
        "--readiness_port",
        type=int,
        default=None,
        help="port for readiness/shutdown HTTP server (default: disabled)",
    )
    parser.add_argument(
        "--service_type",
        type=str,
        choices=["ranking", "retrieval", *service_registry.EXTRA_RUNNERS],
        default="ranking",
        help="Type of service to launch: 'ranking' for PredictNextActions, "
        + "/".join(f"'{s}'" for s in ["retrieval", *service_registry.EXTRA_RUNNERS])
        + " for RetrieveTopKCandidates",
    )
    parser.add_argument(
        "--retrieval_dataset_types",
        type=str,
        nargs="+",
        choices=[ds.name for ds in RetrievalDataset if ds.path is not None],
        default=[RetrievalDataset.HOME.name],
        help="Dataset types to use for retrieval (default: HOME). Can specify multiple.",
    )
    parser.add_argument(
        "--max_inflight_requests",
        type=int,
        default=2560,
    )
    parser.add_argument(
        "--channel_size",
        type=int,
        default=2048,
        help="Size of the request queue channel (default: 2048)",
    )
    parser.add_argument(
        "--enqueue_timeout_ms",
        type=int,
        default=1000,
        help="Timeout in ms for enqueueing requests (default: 1000)",
    )
    parser.add_argument(
        "--queue_max_staleness_ms",
        type=int,
        default=1200,
        help=(
            "Max time (ms) an item may sit in the engine request queue before "
            "it is dropped at dequeue with deadline_exceeded. Default 1200, "
            "aligned with --deadline_fallback_budget_ms: admission predicts a "
            "1200ms violation and sheds at the front door; this drop is the "
            "realized-age backstop at dequeue. 0 disables the drop. Set to a "
            "value somewhat shorter than the gRPC client deadline where "
            "known. See crates/serving/xai-recsys-engine for the "
            "queueing-theory motivation."
        ),
    )
    parser.add_argument(
        "--enable_deadline_admission",
        type=str2bool,
        default=False,
        help="Reject a request at the gRPC front door when its predicted completion "
        "time (queue backlog / measured per-batch service rate) exceeds the caller's "
        "grpc-timeout, so it can fail fast and retry a healthy replica. Gated; off by default.",
    )
    parser.add_argument(
        "--deadline_margin_ms",
        type=float,
        default=30.0,
        help="Safety margin added to predicted completion time (default: 30ms).",
    )
    parser.add_argument(
        "--deadline_post_ms",
        type=float,
        default=2.0,
        help="Fixed post-compute reply/serialize cost not in the service-time sample "
        "(default: 2ms; the pipeline completion lag is modeled separately as "
        "pipeline_depth service periods).",
    )
    parser.add_argument(
        "--deadline_fallback_budget_ms",
        type=float,
        default=1200.0,
        help="Fallback deadline budget when a request carries no grpc-timeout — the "
        "max time a request may sit in the queue before it is stale. Upstream "
        "callers typically send no grpc-timeout, so this is the operative budget. "
        "0 disables the fallback (default: 1200). Distinct from "
        "--queue_max_staleness_ms (the dequeue-time stale drop); set them "
        "consistently when both are in use.",
    )
    parser.add_argument(
        "--service_time_ewma_alpha",
        type=float,
        default=0.2,
        help="EWMA smoothing factor for the per-batch service-time estimate (default: 0.2).",
    )
    parser.add_argument(
        "--max_inflight_backend_requests",
        type=int,
        default=2,
        help="max number of inflight requests to backend",
    )
    parser.add_argument(
        "--restart_period",
        type=int,
        default=None,
        help="Restart period in seconds. If not specified, never restart.",
    )
    parser.add_argument(
        "--graceful_restart_timeout",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--driver",
        type=str,
        choices=("local",),
        default="local",
    )
    parser.add_argument(
        "--max_restart_jitter",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--num_devices_per_process",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--debug_log_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--jax_profile_path",
        type=str,
        default=None,
        help="Enable JAX profiling and specify output directory path",
    )
    parser.add_argument(
        "--allow_random_init",
        type=str2bool,
        default=False,
        help="Allow random initialization for inference (sets DEBUG_ALLOW_RANDOM_INIT=1).",
    )
    parser.add_argument(
        "--fake_mm_embeddings",
        type=str2bool,
        default=False,
        help=(
            "Skip the real object-store multimodal-embeddings fetch and leave the pinned host buffer "
            "as zeros. Useful when profiling inference runtime without needing real MM data."
        ),
    )
    parser.add_argument(
        "--use_pipelining",
        type=str2bool,
        default=True,
        help="Enable the compute/reply pipeline (default: True).",
    )
    parser.add_argument(
        "--use_pinned_d2h",
        type=str2bool,
        default=False,
        help="Use CUDA pinned host memory for D2H transfer (~50 GB/s vs JAX's ~3 GB/s). "
        "Saves ~29ms/inference.",
    )
    parser.add_argument(
        "--pinned_d2h_num_buffers",
        type=int,
        default=3,
        help="Number of pinned host buffers per output shape when --use_pinned_d2h is enabled. "
        "Only needs to cover the Python pipeline depth (not Rust reply threads) since "
        "reply_request() copies into heap memory before donating to Rust.",
    )
    parser.add_argument(
        "--embedding_gather_threads",
        type=int,
        default=16,
        help="Threads for the Rust embedding gather.",
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=1,
        help="Number of embeddings to generate autoregressively (gen_recs only, default: 1)",
    )
    parser.add_argument(
        "--enable_bloom_filter",
        type=str2bool,
        default=False,
        help="Enable bloom-filter eligible-post mask for retrieval. "
        "When True, the JIT compiles with a [B, M] mask at startup so "
        "bloom-filter requests incur no recompilation. Only needed for "
        "deployments that pass eligible-post bloom filters in requests.",
    )
    parser.add_argument(
        "--enable_topic_filter",
        type=str2bool,
        default=False,
        help="Enable topic-based filtering for retrieval. "
        "When True, loads topic bitmap parquet files at startup and on "
        "checkpoint reload, enabling per-request topic filtering via "
        "topic_entity_ids in the gRPC request.",
    )
    parser.add_argument(
        "--enable_async_response_compression",
        type=str2bool,
        default=False,
        help="Enable async gRPC response compression (zstd, off the tokio executor).",
    )
    parser.add_argument(
        "--enable_hotswap",
        type=str2bool,
        default=False,
        help="Enable zero-downtime hot-swap model reloading. "
        "Uses double-buffered emb_table (host memory) and GPU-resident "
        "dual-copy dense weights with instant pointer swap. Falls back "
        "to H2D-copy swap (~100-500ms) if GPU memory is insufficient "
        "for two param copies. The server never marks itself not-ready "
        "during reloads.",
    )
    parser.add_argument(
        "--hotswap_stage_rate_limit_gib_per_s",
        type=float,
        default=None,
        help="Cap the live-swap host->GPU staging bandwidth in GiB/s so staging "
        "does not starve per-request embedding H2D. None = unlimited / single "
        "device_put per leaf (default). Only used by the live swap path.",
    )
    parser.add_argument(
        "--hotswap_stage_chunk_mib",
        type=int,
        default=32,
        help="Max bytes (MiB) per host->GPU staging transfer when "
        "--hotswap_stage_rate_limit_gib_per_s is set; larger param leaves are "
        "split along axis 0. Default 32.",
    )
    parser.add_argument(
        "--hotswap_download_rate_limit_gib_per_s",
        type=float,
        default=None,
        help="Limit hotswap background download bandwidth in GiB/s "
        "(gibibytes per second, not gigabits). Downloads run in batches "
        "of --hotswap_download_max_concurrent channels with sleeps between "
        "batches to stay under the target rate. Reduces memory bandwidth "
        "contention with the serving path at the cost of slower checkpoint "
        "reloads. None = unlimited (default).",
    )
    parser.add_argument(
        "--hotswap_download_max_concurrent",
        type=int,
        default=None,
        help="Maximum number of parallel download channels per batch when "
        "rate limiting is active. Controls peak memory write bandwidth "
        "(more channels = higher burst, fewer = smoother). "
        "Default: half of available channels when rate limit is set.",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=None,
        help="Worker index in multi-worker mode (0 = leader). None = single-process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Total number of workers in multi-worker mode. None = single-process.",
    )
    parser.add_argument(
        "--log_rotate",
        type=str2bool,
        default=True,
        help="Enable log rotation for rank log files (default: True).",
    )
    parser.add_argument(
        "--log_rotate_max_bytes",
        type=int,
        default=3 * 1024 * 1024 * 1024,
        help="Max bytes per rotated log file (default: 3GB).",
    )
    parser.add_argument(
        "--log_rotate_backup_count",
        type=int,
        default=7,
        help="Number of rotated backup files to keep (default: 7, ~21GB total ≈ 3 days at inference log rates).",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=1,
        help=(
            "Parallel beam width for SID retrieval decode (default 1). "
            "Each beam SID maps to ≤1 corpus post; reply filled to large_k "
            "in beam-score order. Set at server boot — restart to change. "
            "Only consumed when --service_type=sid_retrieval."
        ),
    )
    parser.add_argument(
        "--decode_levels",
        type=int,
        default=0,
        help=(
            "SID retrieval: stop the autoregressive beam search after the "
            "first N levels. 0 (default) = full decode (cfg.sid_num_levels, "
            "typically 6). Smaller N speeds up beam decode roughly linearly "
            "but makes each beam SID a prefix matching many corpus posts."
        ),
    )
    parser.add_argument(
        "--sid_endpoint",
        type=str,
        default=None,
        help=(
            "gRPC endpoint of the SID service for SID-aware retrieval inference. "
            "Required when the model uses use_post_sid=True. The endpoint must match "
            "the model's trained sid_codebook_size."
        ),
    )
    parser.add_argument(
        "--jax_compilation_cache_dir",
        type=str,
        default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        help=(
            "Directory used as the JAX persistent compilation cache. When "
            "set, XLA-compiled forward-pass executables are written here on "
            "first compile and reused across pod restarts that share the "
            "same JAX/XLA version, model graph, sharding, and device "
            "topology. Falls back to the JAX_COMPILATION_CACHE_DIR env var. "
            "Empty (default) disables the cache -- behaviour is identical "
            "to today."
        ),
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default=os.environ.get("DEPLOYMENT_NAME", ""),
        help=(
            "Human-readable deployment identifier (e.g. "
            "'recsys-inference-experiment-1'). Logged for debuggability and "
            "used to namespace "
            "--jax_compilation_cache_dir across deployments. Falls back to the "
            "DEPLOYMENT_NAME env var."
        ),
    )
    parser.add_argument(
        "--jax_persistent_cache_min_compile_time_secs",
        type=int,
        default=int(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", 0)),
        help=(
            "Only cache JIT executables whose XLA compile takes at least "
            "this many seconds. Default 0 caches every JIT, maximizing "
            "cold-start coverage (the cache lives on local disk + NFS and is "
            "small relative to the model). Raise it to skip trivially-cheap "
            "compiles. Has no effect when --jax_compilation_cache_dir is empty."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "key=value arguments to override in the config. "
            "Can specify leaf arguments such as 'sequence_len=2048' or "
            "partially-qualified paths 'model.model_type=200m'."
        ),
    )

    args = parser.parse_args()
    pin_visible_devices(args.num_devices_per_process)
    run(args=args)
