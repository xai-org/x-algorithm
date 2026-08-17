# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from datetime import timedelta
from typing import List, Literal, Tuple, overload

import numpy as np
import numpy.typing as npt

class PyMmEmbeddingsClient:
    def __init__(
        self,
        *,
        shared_path: str | None = None,
        worker_id: int = 0,
        emb_dim: int = 1024,
    ) -> None: ...
    def fetch_mm_embeddings_array(
        self, post_ids: npt.NDArray[np.int64], emb_dim: int
    ) -> npt.NDArray[np.float16]: ...
    def wait_until_ready(self) -> None: ...

class PySemanticIdClient:
    def __init__(self, endpoint: str, sid_num_levels: int) -> None: ...

class RankingBatchPrep:
    def __init__(
        self,
        user_hashes: npt.NDArray[np.int32],
        user_ip_hashes: npt.NDArray[np.int32],
        history_post_hashes: npt.NDArray[np.int32],
        history_auth_hashes: npt.NDArray[np.int32],
        history_product_surfaces: npt.NDArray[np.int32],
        history_actions: npt.NDArray[np.bool_],
        history_continuous_actions: npt.NDArray[np.float32],
        candidate_post_hashes: npt.NDArray[np.int32],
        candidate_auth_hashes: npt.NDArray[np.int32],
        candidate_product_surfaces: npt.NDArray[np.int32],
        candidate_embeddings: Tuple[npt.NDArray[np.float16], ...] | None,
        candidate_search_query_embeddings: npt.NDArray[np.float32] | None = None,
        user_categorical_features: npt.NDArray[np.int16] | None = None,
        user_bool_features: npt.NDArray[np.bool_] | None = None,
        user_float_features: npt.NDArray[np.float32] | None = None,
        user_int64_features: npt.NDArray[np.int64] | None = None,
        user_installed_apps: npt.NDArray[np.bool_] | None = None,
        history_categorical_features: npt.NDArray[np.int16] | None = None,
        history_bool_features: npt.NDArray[np.bool_] | None = None,
        history_float_features: npt.NDArray[np.float32] | None = None,
        history_int64_features: npt.NDArray[np.int64] | None = None,
        candidate_categorical_features: npt.NDArray[np.int16] | None = None,
        candidate_bool_features: npt.NDArray[np.bool_] | None = None,
        candidate_float_features: npt.NDArray[np.float32] | None = None,
        candidate_int64_features: npt.NDArray[np.int64] | None = None,
        history_impr_ts: npt.NDArray[np.int32] | None = None,
        history_post_creation_ts_sec: npt.NDArray[np.int32] | None = None,
        candidate_impr_ts: npt.NDArray[np.int32] | None = None,
        candidate_post_creation_ts_sec: npt.NDArray[np.int32] | None = None,
        candidate_line_item_ids: npt.NDArray[np.int64] | None = None,
        candidate_campaign_ids: npt.NDArray[np.int64] | None = None,
        candidate_funding_instrument_ids: npt.NDArray[np.int64] | None = None,
        user_client_id_idx: npt.NDArray[np.int32] | None = None,
        user_country_code_idx: npt.NDArray[np.int32] | None = None,
        user_gender_idx: npt.NDArray[np.int32] | None = None,
        user_age_bucket_idx: npt.NDArray[np.int32] | None = None,
        user_age_in_years_idx: npt.NDArray[np.int32] | None = None,
        user_inferred_gender_idx: npt.NDArray[np.int32] | None = None,
        user_inferred_gender_score: npt.NDArray[np.float32] | None = None,
        candidate_conversion_dense_features: npt.NDArray[np.float32] | None = None,
        user_conversion_history_hashes: npt.NDArray[np.int32] | None = None,
        candidate_account_hashes: npt.NDArray[np.int32] | None = None,
        history_post_sids: npt.NDArray[np.uint16] | None = None,
        candidate_post_sids: npt.NDArray[np.uint16] | None = None,
    ) -> None: ...
    user_hashes: npt.NDArray[np.int32]
    user_ip_hashes: npt.NDArray[np.int32]
    history_post_hashes: npt.NDArray[np.int32]
    history_auth_hashes: npt.NDArray[np.int32]
    history_product_surfaces: npt.NDArray[np.int32]
    history_actions: npt.NDArray[np.bool_]
    history_continuous_actions: npt.NDArray[np.float32]
    candidate_post_hashes: npt.NDArray[np.int32]
    candidate_auth_hashes: npt.NDArray[np.int32]
    candidate_product_surfaces: npt.NDArray[np.int32]
    candidate_embeddings: Tuple[npt.NDArray[np.float16], ...] | None
    candidate_search_query_embeddings: npt.NDArray[np.float32] | None
    history_categorical_features: npt.NDArray[np.int16] | None
    candidate_categorical_features: npt.NDArray[np.int16] | None
    history_post_sids: npt.NDArray[np.uint16] | None
    candidate_post_sids: npt.NDArray[np.uint16] | None

class RetrievalBatchPrep:
    def __init__(
        self,
        user_hashes: npt.NDArray[np.int32],
        user_ip_hashes: npt.NDArray[np.int32],
        history_post_hashes: npt.NDArray[np.int32],
        history_auth_hashes: npt.NDArray[np.int32],
        history_actions: npt.NDArray[np.bool_],
        history_continuous_actions: npt.NDArray[np.float32],
        history_product_surfaces: npt.NDArray[np.int32],
        history_post_ids: npt.NDArray[np.int64],
        user_categorical_features: npt.NDArray[np.int16] | None = None,
        user_bool_features: npt.NDArray[np.bool_] | None = None,
        user_float_features: npt.NDArray[np.float32] | None = None,
        user_int64_features: npt.NDArray[np.int64] | None = None,
        user_installed_apps: npt.NDArray[np.bool_] | None = None,
        history_categorical_features: npt.NDArray[np.int16] | None = None,
        history_bool_features: npt.NDArray[np.bool_] | None = None,
        history_float_features: npt.NDArray[np.float32] | None = None,
        history_int64_features: npt.NDArray[np.int64] | None = None,
        history_post_sids: npt.NDArray[np.uint16] | None = None,
    ) -> None: ...
    user_hashes: npt.NDArray[np.int32]
    user_ip_hashes: npt.NDArray[np.int32]
    history_post_hashes: npt.NDArray[np.int32]
    history_auth_hashes: npt.NDArray[np.int32]
    history_actions: npt.NDArray[np.bool_]
    history_continuous_actions: npt.NDArray[np.float32]
    history_product_surfaces: npt.NDArray[np.int32]
    history_categorical_features: npt.NDArray[np.int16] | None
    history_post_ids: npt.NDArray[np.int64]
    history_post_sids: npt.NDArray[np.uint16] | None

class ReqUserSequenceAggrgated:
    user_ids: npt.NDArray[np.int64]
    post_lengths: npt.NDArray[np.int64]
    author_ids: npt.NDArray[np.int64]
    tweet_ids: npt.NDArray[np.int64]
    actions: npt.NDArray[np.bool_]
    impressed_time_ms: npt.NDArray[np.int64]

class ReqCandidateSequence:
    author_ids: npt.NDArray[np.int64]
    tweet_ids: npt.NDArray[np.int64]
    can_lengths: npt.NDArray[np.int64]

class RetrieveRequestBatch:
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def reply(
        self,
        dataset_types: List[int],
        all_top_k_indices: List[npt.NDArray[np.int32]],
        all_top_k_scores: List[npt.NDArray[np.float32]],
        all_post_ids: npt.NDArray[np.int64],
        all_author_ids: npt.NDArray[np.int64],
        large_k: int,
        orig_batch_size: int,
    ) -> None: ...
    def get_top_ks(self) -> List[int]: ...
    def get_topic_entity_ids(self) -> List[List[int]]: ...
    def get_topic_filter_mode(self) -> int: ...
    def get_pre_fetched_seed_embeddings(
        self,
        emb_dim: int,
    ) -> List[tuple[List[int], npt.NDArray[np.float32]]]: ...

class PredictRequestBatch:
    def get_candidate_sets(self, max_num_cands: int) -> ReqCandidateSequence: ...
    def get_return_logprob(self) -> list[bool]: ...
    def get_top_logprobs_num(self) -> list[int]: ...
    def reply(
        self,
        log_probs: npt.NDArray[np.float32],
        candidate_continuous_predictions: npt.NDArray[np.float32],
        batch_size: int,
    ): ...
    def len(self) -> int: ...

class RecsysPredictorServer:
    def __init__(
        self,
        grpc_port: int,
        metrics_port: int,
        batch_size: int,
        max_inflight_requests: int,
        *,
        channel_size: int = 2048,
        enqueue_timeout_ms: int = 1000,
        queue_max_staleness_ms: int = 0,
        mm_client: PyMmEmbeddingsClient | None = None,
        sid_client: PySemanticIdClient | None = None,
        user_id_table_size: int = 100_000,
        user_hash_scales: list[int] = ...,
        user_biases: list[int] = ...,
        user_modulus: int = 2_859_568_897,
        item_id_table_size: int = 100_000,
        item_hash_vocab_size: int = 0,
        item_hash_scales: list[int] = ...,
        item_biases: list[int] = ...,
        item_modulus: int = 2_361_375_383,
        author_id_table_size: int = 10_000,
        author_hash_scales: list[int] = ...,
        author_biases: list[int] = ...,
        author_modulus: int = 631_860_353,
        output_vocab_size: int = 64,
        num_continuous_actions: int = 8,
        history_seq_len: int = 1000,
        candidate_seq_len: int = 1400,
        multimodal_embedding_dim: int = 0,
        search_query_embedding_dim: int = 0,
        ip_id_table_size: int = 0,
        ip_hash_scales: list[int] = ...,
        ip_biases: list[int] = ...,
        ip_modulus: int = 1_073_741_789,
        num_user_categorical_features: int = 0,
        num_user_bool_features: int = 0,
        num_user_float_features: int = 0,
        num_user_int64_features: int = 0,
        num_user_installed_apps: int = 0,
        num_post_categorical_features: int = 0,
        num_post_bool_features: int = 0,
        num_post_float_features: int = 0,
        num_post_int64_features: int = 0,
        enable_stale_post: bool = False,
        enable_async_response_compression: bool = False,
        sid_num_levels: int = 0,
        enable_deadline_admission: bool = False,
        deadline_margin_ms: float = 30.0,
        deadline_post_ms: float = 2.0,
        deadline_fallback_budget_ms: float = 1200.0,
        service_time_ewma_alpha: float = 0.2,
        pipeline_depth: int = 1,
        prefetch_mm_query_for_retrieval: bool = False,
        mm_query_sample_size: int = 0,
        mm_query_max_age_hours: int = 48,
        mm_query_positive_action_types: list[int] = ...,
        peer_serve_enabled: bool = False,
        peer_serve_rate_limit_gib_per_s: float = 0.0,
        peer_serve_max_concurrent_sends: int = 4,
        peer_download_enabled: bool = False,
        tls_cert_path: str | None = None,
        tls_key_path: str | None = None,
        tls_client_ca_path: str | None = None,
    ) -> None: ...
    def set_ready(self, is_ready: bool) -> None: ...
    def check_reload_request(self) -> bool: ...
    def reset_reload_request(self) -> None: ...
    def set_checkpoint_loading(self) -> None: ...
    def clear_checkpoint_loading(self) -> None: ...
    def reset_dequeue_timer(self) -> None: ...
    def record_service_time_ms(self, ms: float) -> None: ...
    def set_deadline_admission_enabled(self, enabled: bool) -> None: ...
    def is_finished(self) -> bool: ...
    def cancel(self) -> None: ...
    def wait(self, timeout: timedelta | None) -> None: ...
    def dequeue(self) -> PredictRequestBatch | None: ...
    def set_serving_prefix(self, prefix: str) -> None: ...
    def take_reload_directive(self) -> Tuple[List[str], str] | None: ...
    def publish_peer_manifest(
        self,
        prefix: str,
        entries: List[Tuple[str, str, int, int]],
        blobs: List[Tuple[str, bytes]],
    ) -> bool: ...
    def revoke_peer_manifest(self, drain_timeout_secs: float = 10.0) -> bool: ...
    def hash_user_id(self, user_id: int, index: int) -> int: ...
    def hash_item_id(self, item_id: int, index: int) -> int: ...
    def hash_author_id(self, author_id: int, index: int) -> int: ...
    def hash_user_ids(self, user_ids: list[int]) -> list[int]: ...
    def hash_item_ids(self, item_ids: list[int]) -> list[int]: ...
    def hash_author_ids(self, author_ids: list[int]) -> list[int]: ...
    def compute_batch_inputs_rust_quick(
        self,
        request: PredictRequestBatch,
        request_in_flight_id: int,
        input: RankingBatchPrep,
        batch_size: int | None = None,
    ) -> bool: ...

class RecsysRetrievalPredictorServer:
    def __init__(
        self,
        grpc_port: int,
        metrics_port: int,
        batch_size: int,
        max_inflight_requests: int,
        *,
        channel_size: int = 2048,
        enqueue_timeout_ms: int = 1000,
        queue_max_staleness_ms: int = 0,
        mm_client: PyMmEmbeddingsClient | None = None,
        sid_client: PySemanticIdClient | None = None,
        user_id_table_size: int = 100_000,
        user_hash_scales: list[int] = ...,
        user_biases: list[int] = ...,
        user_modulus: int = 2_859_568_897,
        item_id_table_size: int = 100_000,
        item_hash_vocab_size: int = 0,
        item_hash_scales: list[int] = ...,
        item_biases: list[int] = ...,
        item_modulus: int = 2_361_375_383,
        author_id_table_size: int = 10_000,
        author_hash_scales: list[int] = ...,
        author_biases: list[int] = ...,
        author_modulus: int = 631_860_353,
        output_vocab_size: int = 64,
        num_continuous_actions: int = 8,
        history_seq_len: int = 1000,
        candidate_seq_len: int = 1400,
        multimodal_embedding_dim: int = 0,
        search_query_embedding_dim: int = 0,
        ip_id_table_size: int = 0,
        ip_hash_scales: list[int] = ...,
        ip_biases: list[int] = ...,
        ip_modulus: int = 1_073_741_789,
        num_user_categorical_features: int = 0,
        num_user_bool_features: int = 0,
        num_user_float_features: int = 0,
        num_user_int64_features: int = 0,
        num_user_installed_apps: int = 0,
        num_post_categorical_features: int = 0,
        num_post_bool_features: int = 0,
        num_post_float_features: int = 0,
        num_post_int64_features: int = 0,
        enable_stale_post: bool = False,
        enable_async_response_compression: bool = False,
        sid_num_levels: int = 0,
        enable_deadline_admission: bool = False,
        deadline_margin_ms: float = 30.0,
        deadline_post_ms: float = 2.0,
        deadline_fallback_budget_ms: float = 1200.0,
        service_time_ewma_alpha: float = 0.2,
        pipeline_depth: int = 1,
        prefetch_mm_query_for_retrieval: bool = False,
        mm_query_sample_size: int = 0,
        mm_query_max_age_hours: int = 48,
        mm_query_positive_action_types: list[int] = ...,
        peer_serve_enabled: bool = False,
        peer_serve_rate_limit_gib_per_s: float = 0.0,
        peer_serve_max_concurrent_sends: int = 4,
        peer_download_enabled: bool = False,
        tls_cert_path: str | None = None,
        tls_key_path: str | None = None,
        tls_client_ca_path: str | None = None,
    ) -> None: ...
    def set_ready(self, is_ready: bool) -> None: ...
    def check_reload_request(self) -> bool: ...
    def reset_reload_request(self) -> None: ...
    def set_checkpoint_loading(self) -> None: ...
    def clear_checkpoint_loading(self) -> None: ...
    def reset_dequeue_timer(self) -> None: ...
    def record_service_time_ms(self, ms: float) -> None: ...
    def set_deadline_admission_enabled(self, enabled: bool) -> None: ...
    def is_finished(self) -> bool: ...
    def cancel(self) -> None: ...
    def wait(self, timeout: timedelta | None) -> None: ...
    def dequeue(self) -> RetrieveRequestBatch | None: ...
    def set_serving_prefix(self, prefix: str) -> None: ...
    def take_reload_directive(self) -> Tuple[List[str], str] | None: ...
    def publish_peer_manifest(
        self,
        prefix: str,
        entries: List[Tuple[str, str, int, int]],
        blobs: List[Tuple[str, bytes]],
    ) -> bool: ...
    def revoke_peer_manifest(self, drain_timeout_secs: float = 10.0) -> bool: ...
    def hash_user_id(self, user_id: int, index: int) -> int: ...
    def hash_item_id(self, item_id: int, index: int) -> int: ...
    def hash_author_id(self, author_id: int, index: int) -> int: ...
    def hash_user_ids(self, user_ids: list[int]) -> list[int]: ...
    def hash_item_ids(self, item_ids: list[int]) -> list[int]: ...
    def hash_author_ids(self, author_ids: list[int]) -> list[int]: ...
    def compute_batch_inputs_rust_quick(
        self,
        request: RetrieveRequestBatch,
        request_in_flight_id: int,
        input: RetrievalBatchPrep,
        batch_size: int | None = None,
    ) -> bool: ...

def init_parallel(arr: npt.NDArray[np.uint8], num_threads: int): ...
def madvise_hugepage(arr: npt.NDArray[np.uint8]): ...
def embedding_gather(
    table: npt.NDArray[np.uint8],
    row_indexes: npt.NDArray[np.uint32],
    rows: Tuple[npt.NDArray[np.uint8], ...],
    row_size: int,
    num_threads: int,
): ...
def load_tensor(
    path: str,
    urls: str,
    shard_sources: List[Tuple[str, str, int, int]],
    tensor: npt.NDArray[np.uint8],
    row_size: int,
    num_row_segments: int,
) -> None: ...
@overload
def load_tensor_no_resharding(
    path: str,
    urls: str,
    name: str,
    tensor: npt.NDArray[np.uint8],
    rate_limit_bytes_per_sec: int | None = None,
    max_concurrent_downloads: int | None = None,
    collect_layout: Literal[False] = False,
) -> Tuple[int]: ...
@overload
def load_tensor_no_resharding(
    path: str,
    urls: str,
    name: str,
    tensor: npt.NDArray[np.uint8],
    rate_limit_bytes_per_sec: int | None = None,
    max_concurrent_downloads: int | None = None,
    *,
    collect_layout: Literal[True],
) -> Tuple[int, int, int]: ...
@overload
def maybe_load_fully_replicated_tensors(
    elapsed_samples: int,
    urls: str,
    tensors: List[Tuple[str, npt.NDArray[np.uint8]]],
    rate_limit_bytes_per_sec: int | None = None,
    max_concurrent_downloads: int | None = None,
    target_prefix: str | None = None,
    collect_entry_names: Literal[False] = False,
) -> Tuple[str, str, float]: ...
@overload
def maybe_load_fully_replicated_tensors(
    elapsed_samples: int,
    urls: str,
    tensors: List[Tuple[str, npt.NDArray[np.uint8]]],
    rate_limit_bytes_per_sec: int | None = None,
    max_concurrent_downloads: int | None = None,
    target_prefix: str | None = None,
    *,
    collect_entry_names: Literal[True],
) -> Tuple[str, str, float, List[Tuple[str, str]]]: ...
def upload_files_to_storage(
    local_base_dir: str, relative_paths: List[str], storage_url: str
) -> None: ...
def download_prefix_to_local(
    storage_url: str,
    local_dir: str,
    max_concurrent: int | None = None,
    shallow: bool = False,
) -> int: ...
def download_files_from_storage(
    storage_url: str,
    relative_paths: List[str],
    local_dir: str,
    max_concurrent: int | None = None,
) -> int: ...
def upload_inline_data_to_storage(
    data_entries: List[Tuple[str, bytes]],
    output_url: str,
) -> None: ...
def copy_chunks_to_storage(
    chunks: List[Tuple[str, int, int, str]],
    output_url: str,
    num_threads: int,
) -> None: ...
def reshard_col_to_row_and_upload(
    col_shard_info: List[Tuple[str, int, int]],
    output_url: str,
    tensor_name: str,
    num_rows: int,
    row_size_bytes: int,
    num_output_shards: int,
    num_threads: int,
) -> None: ...
def load_reshard_row_to_col(
    base_path: str,
    shard_keys: List[Tuple[int, str]],
    output_dir: str,
    tensor_name: str,
    num_rows: int,
    row_size_bytes: int,
    num_col_shards: int,
    num_threads: int,
) -> None: ...
def list_storage_objects(storage_url: str) -> List[str]: ...
def find_latest_storage_checkpoint(
    base_url: str,
    name_size_pairs: List[Tuple[str, int]],
    fixed_prefix: str | None = None,
) -> Tuple[str, int] | None: ...
def load_arrays_from_storage(
    storage_url: str,
    name_array_pairs: List[Tuple[str, npt.NDArray[np.uint8]]],
    max_concurrent_downloads: int | None = None,
) -> None: ...
def remove_old_from_storage(
    storage_url: str, n: int, name_size_pairs: List[Tuple[str, int]]
) -> None: ...
def adler32_parallel(data: npt.NDArray[np.uint8]) -> int: ...
def adler32_combine_py(checksum1: int, checksum2: int, len2: int) -> int: ...
def adler32_shard_partial(
    shard: npt.NDArray[np.uint8],
    shard_width: int,
    shard_index: int,
    total_cols: int,
    num_chunks: int,
) -> npt.NDArray[np.uint32]: ...
def adler32_combine_partials(
    partials: npt.NDArray[np.uint32],
    total_len: int,
) -> int: ...
