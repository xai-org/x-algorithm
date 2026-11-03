# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import os
import socket
from pathlib import Path

from xrex import settings
from xrex.configs import config_registry
from xrex.data.grpc_recsys import PhoenixGrpcDataset
from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.rust_kafka_recsys import RustKafkaDataset
from xrex.data.rust_parquet_recsys import RustParquetDataset
from xrex.data.streaming.kafkadispatcherloader import PhoenixKafkaDispatcherDataset
from xrex.data.streaming.kafkaloader import (
    PhoenixKafkaDataset,
    cluster_sasl_username,
    ensure_sasl_password_env,
    resolve_internal_bootstrap,
)
from xrex.utils import cluster as cluster_identity

PAD_TOKEN = 0

_KAFKA_CLUSTER = settings.KAFKA_CLUSTER

_OTEL_ENDPOINT = settings.OTEL_ENDPOINT
_RUST_BOOTSTRAP = settings.KAFKA_RUST_BOOTSTRAP
_DISPATCHER_HOST_TEMPLATE = settings.KAFKA_DISPATCHER_HOST_TEMPLATE
_GRPC_DATALOADER_ADDRESS = settings.GRPC_DATALOADER_ADDRESS


def kafka_kwargs(cluster: str = _KAFKA_CLUSTER) -> dict:
    ensure_sasl_password_env(cluster)
    return {
        "bootstrap_servers": resolve_internal_bootstrap(cluster),
        "sasl_plain_username": cluster_sasl_username(cluster),
        "otel_endpoint": _OTEL_ENDPOINT,
    }


def phoenix_kafka_kwargs() -> dict:
    return kafka_kwargs(_KAFKA_CLUSTER)


SID_GLOBAL_IDS_PLACEHOLDER = "/path/to/post_sid_global_ids.parquet"
_SID_GLOBAL_IDS_PLACEHOLDER = Path(SID_GLOBAL_IDS_PLACEHOLDER)
SID_GLOBAL_IDS_SNAPSHOT = (
    Path(settings.SID_GLOBAL_IDS_SNAPSHOT)
    if settings.SID_GLOBAL_IDS_SNAPSHOT
    else _SID_GLOBAL_IDS_PLACEHOLDER
)
SID_VIDEO_GLOBAL_IDS_PLACEHOLDER = "/path/to/post_sid_video_global_ids.parquet"
_SID_VIDEO_GLOBAL_IDS_PLACEHOLDER = Path(SID_VIDEO_GLOBAL_IDS_PLACEHOLDER)
SID_VIDEO_GLOBAL_IDS_SNAPSHOT = (
    Path(settings.SID_VIDEO_GLOBAL_IDS_SNAPSHOT)
    if settings.SID_VIDEO_GLOBAL_IDS_SNAPSHOT
    else _SID_VIDEO_GLOBAL_IDS_PLACEHOLDER
)

_RANKING_DUMP_PATH = settings.RANKING_DUMP_PATH
_RETRIEVAL_DUMP_PATH = settings.RETRIEVAL_DUMP_PATH


GROUP_IDS: dict[str, str] = {
    "xrecsys_seqpack": "user_action_sequence_xrecsys",
    "xrecsys_search": "user_action_sequence_xrecsys",
    "home_direct_packed": "user_action_sequence_home_direct_packed",
    "home_direct_packed_gb300": "user_action_sequence_home_direct_packed",
    "home_direct_packed_nano": "user_action_sequence_home_direct_packed_nano",
    "immersive_page_rl": "user_action_sequence_immersive_page_rl",
    "immersive_page_rl_cread": "user_action_sequence_immersive_page_rl_cread",
    "xrecsys_two_tower": "user_action_sequence_xrecsys_two_tower",
    "xrecsys_two_tower_combined": "user_action_sequence_xrecsys_two_tower_combined",
    "xrecsys_two_tower_combined_gb300": "user_action_sequence_xrecsys_two_tower_combined",
    "xrecsys_two_tower_home_only": "user_action_sequence_xrecsys_two_tower_home_only",
    "xrecsys_two_tower_immersive_only": "user_action_sequence_xrecsys_two_tower_immersive_only",
    "xrecsys_two_tower_immersive_only_gb300": "user_action_sequence_xrecsys_two_tower_immersive_only",
    "xrecsys_two_tower_nano": "user_action_sequence_xrecsys_two_tower_nano",
    "reply_two_tower_v2_main": "reply_two_tower_v2_main",
}


def _off_cluster_suffix() -> str:
    user = (os.environ.get("USER") or os.environ.get("LOGNAME") or "oss").strip()
    host = (cluster_identity.get_hostname() or socket.gethostname() or "localhost").strip()
    return f"{user}_local_{host}"


def job_suffix() -> str:
    user = cluster_identity.get_user()
    job = cluster_identity.get_job_name()
    cluster = cluster_identity.get_cluster()

    if not (user or job or cluster):
        return _off_cluster_suffix()

    if not user:
        raise ValueError("XAI_USER environment variable is required")
    if not job:
        raise ValueError("XAI_JOB_NAME environment variable is required")
    if not cluster:
        raise ValueError("XAI_CLUSTER environment variable is required")

    return f"{user}_{job}_{cluster}"


def _group_id(config_name: str) -> str:
    return f"{GROUP_IDS[config_name]}_{job_suffix()}"


def resolve_global_ids_file_path(global_ids_file_path: Path | None) -> Path | None:
    if global_ids_file_path == _SID_GLOBAL_IDS_PLACEHOLDER:
        return SID_GLOBAL_IDS_SNAPSHOT
    if global_ids_file_path == _SID_VIDEO_GLOBAL_IDS_PLACEHOLDER:
        return SID_VIDEO_GLOBAL_IDS_SNAPSHOT
    return global_ids_file_path


def _page_records(mparams) -> bool:
    page_rl = mparams.get("page_rl")
    return page_rl is not None and page_rl.enabled


def _ranking_aggregated_kafka(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    return PhoenixKafkaDataset(
        **phoenix_kafka_kwargs(),
        topic_name=settings.RANKING_KAFKA_TOPIC,
        group_id=_group_id(config_name) + "-direct-numpy",
        max_queue_size=100,
        hash_table=hash_table,
        path=None,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        output_vocab_size=mparams["output_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        pad_token=PAD_TOKEN,
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        search_query_embedding_dim=mparams.get("search_query_embedding_dim", 0),
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        compute_post_unexplored_label=mparams.get("compute_post_unexplored_label", False),
        enable_stale_post=mparams.get("enable_stale_post", False),
        exclude_required_columns=mparams.get("exclude_required_columns", ""),
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
        search_negative_clear_word_match=mparams.get("search_negative_clear_word_match", False),
        read_ads_engagement_counts=mparams.get("read_ads_engagement_counts", False),
        page_records=_page_records(mparams),
    )


def _ranking_rust_kafka(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    ensure_sasl_password_env(_KAFKA_CLUSTER)
    return RustKafkaDataset(
        topic_name=settings.RANKING_RUST_KAFKA_TOPIC,
        bootstrap_servers=_RUST_BOOTSTRAP,
        sasl_plain_username=cluster_sasl_username(_KAFKA_CLUSTER),
        otel_endpoint=_OTEL_ENDPOINT,
        group_id=_group_id(config_name) + "-rust-kafka",
        hash_table=hash_table,
        path=None,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        output_vocab_size=mparams["output_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        pad_token=PAD_TOKEN,
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        search_query_embedding_dim=mparams.get("search_query_embedding_dim", 0),
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        compute_post_unexplored_label=mparams.get("compute_post_unexplored_label", False),
        enable_stale_post=mparams.get("enable_stale_post", False),
        exclude_required_columns=mparams.get("exclude_required_columns", ""),
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
        max_redistribution_bytes=mparams.get("max_redistribution_bytes", 256 << 20),
        read_ads_engagement_counts=mparams.get("read_ads_engagement_counts", False),
        page_records=_page_records(mparams),
    )


def _ranking_kafka_dispatcher(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    return PhoenixKafkaDispatcherDataset(
        **phoenix_kafka_kwargs(),
        topic_name=settings.RANKING_DISPATCHER_KAFKA_TOPIC,
        group_id=_group_id(config_name) + "-dispatcher",
        max_queue_size=100,
        hash_table=hash_table,
        path=None,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        output_vocab_size=mparams["output_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        pad_token=PAD_TOKEN,
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        grpc_host_template=_DISPATCHER_HOST_TEMPLATE,
        grpc_port=50051,
        grpc_timeout=60.0,
        poll_interval=1.0,
        fetch_batch_size=192,
        reset_to_latest=False,
        seek_to_timestamp_ms=None,
        seek_to_offset=None,
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        compute_post_unexplored_label=mparams.get("compute_post_unexplored_label", False),
        enable_stale_post=mparams.get("enable_stale_post", False),
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
        page_records=_page_records(mparams),
    )


def _ranking_offline_kafka_dump(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    del config_name
    return PhoenixDataset(
        hash_table=hash_table,
        path=_RANKING_DUMP_PATH,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        num_kafka_partitions=1024,
        output_vocab_size=mparams["output_vocab_size"],
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        compute_post_unexplored_label=mparams.get("compute_post_unexplored_label", False),
        enable_stale_post=mparams.get("enable_stale_post", False),
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
        page_records=_page_records(mparams),
    )


def _ranking_rust_parquet(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    del config_name
    return RustParquetDataset(
        hash_table=hash_table,
        path=_RANKING_DUMP_PATH,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        num_kafka_partitions=1024,
        output_vocab_size=mparams["output_vocab_size"],
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        compute_post_unexplored_label=mparams.get("compute_post_unexplored_label", False),
        enable_stale_post=mparams.get("enable_stale_post", False),
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
        page_records=_page_records(mparams),
    )


def _ranking_grpc_recsys(mparams, hash_table, use_post_sid, sid_num_levels, config_name):
    del config_name
    if mparams.get("enable_stale_post", False):
        raise ValueError(
            "enable_stale_post is unsupported on grpc_recsys, which does not transport per-post feature arrays (int64/bool/categorical/float)"
        )
    return PhoenixGrpcDataset(
        hash_table=hash_table,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        server_address=_GRPC_DATALOADER_ADDRESS,
        input_vocab_size=mparams["input_vocab_size"],
        pad_token=PAD_TOKEN,
        output_vocab_size=mparams["output_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
        num_data_loaders=8,
        num_kafka_partitions=1024,
        multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        ads_head_masking=mparams.get("ads_head_masking", False),
        ads_pixel_head_masking=mparams.get("ads_pixel_head_masking", False),
        ads_late_window_heads=mparams.get("ads_late_window_heads", False),
    )


RANKING_DATASET_FACTORIES = {
    "aggregated_kafka": _ranking_aggregated_kafka,
    "rust_kafka": _ranking_rust_kafka,
    "kafka_dispatcher": _ranking_kafka_dispatcher,
    "offline_kafka_dump": _ranking_offline_kafka_dump,
    "rust_parquet": _ranking_rust_parquet,
    "grpc_recsys": _ranking_grpc_recsys,
}

RANKING_EXTRA_DATASET_TYPES = [
    "offline_kafka_dump",
    "rust_parquet",
    "rust_kafka",
    "kafka_dispatcher",
    "grpc_recsys",
]


def _retrieval_aggregated_kafka(
    mparams, hash_table, use_post_sid, sid_num_levels, global_ids_file_path, config_name
):
    extra_kwargs: dict = phoenix_kafka_kwargs()
    if global_ids_file_path is not None:
        extra_kwargs["global_ids_file_path"] = global_ids_file_path
    return PhoenixKafkaDataset(
        topic_name=settings.RETRIEVAL_KAFKA_TOPIC,
        group_id=_group_id(config_name) + "-direct-numpy",
        max_queue_size=100,
        hash_table=hash_table,
        path=None,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        hash_vocab_size=mparams["hash_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams["num_negatives_per_example"],
        num_global_negatives_per_example=mparams["num_global_negatives_per_example"],
        pad_token=PAD_TOKEN,
        include_candidate_post_ids=True,
        global_ids_reload_interval_seconds=180,
        enable_drop_on_high_qps=False,
        num_kafka_partitions=2048,
        output_vocab_size=mparams["output_vocab_size"],
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        exclude_required_columns=mparams.get("exclude_required_columns", ""),
        **extra_kwargs,
    )


def _retrieval_rust_kafka(
    mparams, hash_table, use_post_sid, sid_num_levels, global_ids_file_path, config_name
):
    ensure_sasl_password_env(_KAFKA_CLUSTER)
    extra_kwargs: dict = {
        "bootstrap_servers": _KAFKA_CLUSTER,
        "sasl_plain_username": cluster_sasl_username(_KAFKA_CLUSTER),
        "otel_endpoint": _OTEL_ENDPOINT,
    }
    if global_ids_file_path is not None:
        extra_kwargs["global_ids_file_path"] = global_ids_file_path
    return RustKafkaDataset(
        topic_name=settings.RETRIEVAL_RUST_KAFKA_TOPIC,
        group_id=_group_id(config_name) + "-rust-kafka",
        hash_table=hash_table,
        path=None,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        hash_vocab_size=mparams["hash_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams["num_negatives_per_example"],
        num_global_negatives_per_example=mparams["num_global_negatives_per_example"],
        pad_token=PAD_TOKEN,
        include_candidate_post_ids=True,
        global_ids_reload_interval_seconds=180,
        enable_drop_on_high_qps=False,
        output_vocab_size=mparams["output_vocab_size"],
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        exclude_required_columns=mparams.get("exclude_required_columns", ""),
        **extra_kwargs,
    )


def _retrieval_offline_kafka_dump(
    mparams, hash_table, use_post_sid, sid_num_levels, global_ids_file_path, config_name
):
    del config_name
    extra_kwargs: dict = {}
    if global_ids_file_path is not None:
        extra_kwargs["global_ids_file_path"] = global_ids_file_path
    return PhoenixDataset(
        hash_table=hash_table,
        path=_RETRIEVAL_DUMP_PATH,
        history_seq_len=mparams["history_seq_len"],
        candidate_seq_len=mparams["candidate_seq_len"],
        input_vocab_size=mparams["input_vocab_size"],
        hash_vocab_size=mparams["hash_vocab_size"],
        num_continuous_actions=mparams["num_continuous_actions"],
        num_negatives_per_example=mparams["num_negatives_per_example"],
        num_global_negatives_per_example=mparams["num_global_negatives_per_example"],
        num_kafka_partitions=1024,
        include_candidate_post_ids=True,
        date_range=mparams.get("date_range", None),
        use_post_sid=use_post_sid,
        sid_num_levels=sid_num_levels,
        **extra_kwargs,
    )


RETRIEVAL_DATASET_FACTORIES = {
    "aggregated_kafka": _retrieval_aggregated_kafka,
    "rust_kafka": _retrieval_rust_kafka,
    "offline_kafka_dump": _retrieval_offline_kafka_dump,
}

RETRIEVAL_EXTRA_DATASET_TYPES = [
    "rust_kafka",
    "offline_kafka_dump",
]


config_registry.RANKING_DATASET_FACTORIES.update(RANKING_DATASET_FACTORIES)
config_registry.RANKING_EXTRA_DATASET_TYPES.extend(RANKING_EXTRA_DATASET_TYPES)
config_registry.RETRIEVAL_DATASET_FACTORIES.update(RETRIEVAL_DATASET_FACTORIES)
config_registry.RETRIEVAL_EXTRA_DATASET_TYPES.extend(RETRIEVAL_EXTRA_DATASET_TYPES)
config_registry.GLOBAL_IDS_RESOLVERS.append(resolve_global_ids_file_path)
