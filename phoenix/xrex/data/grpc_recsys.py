# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from threading import Thread

import numpy as np
from xai_configlib import configclass

from xrex.data import rust_ext
from xrex.data.parquet_recsys import DataPosition, PhoenixDataset
from xrex.data.recsys.recsys_batch import (
    NUM_USER_INSTALLED_APPS,
    PostSeq,
    RecsysFeaturesBatch,
    apply_negative_sampling,
    empty_feature_arrays,
    empty_user_feature_arrays,
)

rank_logger = logging.getLogger("rank")


@configclass
class PhoenixGrpcDataset(PhoenixDataset):
    server_address: str = ""
    num_data_loaders: int = 8
    num_splits: int = 8

    def shutdown(self):
        pass

    def make(
        self,
        *,
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        server_hosts: list[str],
        server_port: int = 8898,
        skip_rows: int = 0,
        keep_and_pad_partial_batch: bool | None = None,
        prefetch_factor: int = 2,
        resume_position: DataPosition | None = None,
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        del (
            run_server,
            server_hosts,
            server_port,
            skip_rows,
            keep_and_pad_partial_batch,
            prefetch_factor,
            resume_position,
        )

        if not self.server_address:
            raise ValueError(
                "PhoenixGrpcDataset requires server_address: a gRPC address pattern "
                "for the data loader service, with a '{}' placeholder for the loader "
                "index (e.g. 'http://loader-{}.loaders.svc:9090')."
            )

        grpc_client = rust_ext.load("xai_recsys_dataloader")

        rank_logger.info(f"Starting gRPC sequential processing (shard {shard_index}/{num_shards})")

        hash_keys = self.hash_table.hash_keys
        real_addresses = [self.server_address.format(i) for i in range(self.num_data_loaders)]
        real_address = real_addresses[shard_index % len(real_addresses)]
        dataloader = grpc_client.RecsysDataLoaderPyClient(
            real_address,
            self.num_splits,
            user_id_table_size=hash_keys.user_id_table_size,
            user_hash_scales=hash_keys.user_hash_scales,
            user_biases=hash_keys.user_biases,
            user_modulus=hash_keys.user_modulus,
            item_id_table_size=hash_keys.item_id_table_size,
            item_hash_vocab_size=hash_keys.item_hash_vocab_size,
            item_hash_scales=hash_keys.item_hash_scales,
            item_biases=hash_keys.item_biases,
            item_modulus=hash_keys.item_modulus,
            author_id_table_size=hash_keys.author_id_table_size,
            author_hash_scales=hash_keys.author_hash_scales,
            author_biases=hash_keys.author_biases,
            author_modulus=hash_keys.author_modulus,
            output_vocab_size=self.hash_table.output_vocab_size,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
        )

        prefetch_queue = Queue(maxsize=2)

        SENTINEL = object()

        def producer() -> None:
            try:
                rank_logger.info(
                    f"Starting gRPC prefetch producer thread (shard {shard_index}/{num_shards}, prefetch=2)"
                )

                while True:
                    import time

                    batch_create_start = time.perf_counter()
                    batch = self._create_recsys_features_batch(batch_size)
                    assert batch["history_seq"]["impr_ts"] is not None
                    assert batch["history_seq"]["actions"] is not None
                    assert batch["candidate_seq"]["impr_ts"] is not None
                    assert batch["candidate_seq"]["actions"] is not None
                    batch_create_duration = time.perf_counter() - batch_create_start

                    grpc_start = time.perf_counter()
                    actual_batch_size = dataloader.get_columnar_training_batch(
                        batch_size,
                        shard_index,
                        num_shards,
                        batch["user_hashes"],
                        batch["history_seq"]["impr_ts"],
                        batch["history_seq"]["actions"],
                        batch["history_seq"]["post_hashes"],
                        batch["history_seq"]["auth_hashes"],
                        batch["candidate_seq"]["impr_ts"],
                        batch["candidate_seq"]["actions"],
                        batch["candidate_seq"]["post_hashes"],
                        batch["candidate_seq"]["auth_hashes"],
                    )
                    grpc_duration = time.perf_counter() - grpc_start

                    if actual_batch_size == 0:
                        continue

                    neg_sampling_start = time.perf_counter()
                    user_ids = np.arange(actual_batch_size, dtype=np.int64)

                    candidate_seq_with_negatives = apply_negative_sampling(
                        user_ids,
                        batch["candidate_seq"],
                        self.num_negatives_per_example,
                        actual_batch_size,
                        self.candidate_seq_len,
                    )
                    neg_sampling_duration = time.perf_counter() - neg_sampling_start

                    final_batch_start = time.perf_counter()
                    final_batch = RecsysFeaturesBatch(
                        user_hashes=batch["user_hashes"],
                        user_ip_hashes=batch["user_ip_hashes"],
                        history_seq=batch["history_seq"],
                        candidate_seq=candidate_seq_with_negatives,
                        user_categorical_features=batch["user_categorical_features"],
                        user_bool_features=batch["user_bool_features"],
                        user_float_features=batch["user_float_features"],
                        user_int64_features=batch["user_int64_features"],
                        user_installed_apps_multihot=batch["user_installed_apps_multihot"],
                        num_positive_candidates=None,
                    )
                    final_batch_duration = time.perf_counter() - final_batch_start

                    rank_logger.info(
                        f"Python pipeline timing - batch_create: {batch_create_duration:.3f}s, "
                        f"grpc: {grpc_duration:.3f}s, neg_sampling: {neg_sampling_duration:.3f}s, "
                        f"final_batch: {final_batch_duration:.3f}s, "
                        f"total: {batch_create_duration + grpc_duration + neg_sampling_duration + final_batch_duration:.3f}s"
                        f"actual_batch_size: {actual_batch_size}"
                        f"shard {shard_index}: {real_address}"
                    )

                    prefetch_queue.put((final_batch, None))

            except Exception as e:
                rank_logger.error(f"Exception in gRPC prefetch producer: {e}")
                prefetch_queue.put(e)
            finally:
                prefetch_queue.put(SENTINEL)

        thread = Thread(target=producer)
        thread.start()

        def generator():
            while True:
                item = prefetch_queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

            thread.join()

        return generator()

    def _create_recsys_features_batch(self, batch_size: int) -> RecsysFeaturesBatch:
        return RecsysFeaturesBatch(
            user_hashes=np.zeros((batch_size, self.hash_table.num_user_hashes), dtype=np.int32),
            user_ip_hashes=np.zeros((batch_size, self.hash_table.num_ip_hashes), dtype=np.int32),
            history_seq=PostSeq(
                impr_ts=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                actions=np.zeros(
                    (batch_size, self.history_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                continuous_actions=np.zeros(
                    (batch_size, self.history_seq_len, self.num_continuous_actions),
                    dtype=np.float32,
                ),
                post_hashes=np.zeros(
                    (batch_size, self.history_seq_len, self.hash_table.num_item_hashes),
                    dtype=np.int32,
                ),
                auth_hashes=np.zeros(
                    (batch_size, self.history_seq_len, self.hash_table.num_author_hashes),
                    dtype=np.int32,
                ),
                ip_hashes=np.zeros(
                    (batch_size, self.history_seq_len, self.hash_table.num_ip_hashes),
                    dtype=np.int32,
                ),
                product_surface=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                client_app_id=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                post_creation_ts_sec=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                post_ids=None,
                promoted_ids=None,
                line_item_objective=None,
                safety_label_mask=np.zeros((batch_size, self.history_seq_len), dtype=np.int64),
                embedding=None,
                search_query_embeddings=None,
                **empty_feature_arrays(batch_size, self.history_seq_len),
            ),
            candidate_seq=PostSeq(
                impr_ts=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int32),
                actions=np.zeros(
                    (batch_size, self.candidate_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                continuous_actions=np.zeros(
                    (batch_size, self.candidate_seq_len, self.num_continuous_actions),
                    dtype=np.float32,
                ),
                post_hashes=np.zeros(
                    (batch_size, self.candidate_seq_len, self.hash_table.num_item_hashes),
                    dtype=np.int32,
                ),
                auth_hashes=np.zeros(
                    (batch_size, self.candidate_seq_len, self.hash_table.num_author_hashes),
                    dtype=np.int32,
                ),
                ip_hashes=np.zeros(
                    (batch_size, self.candidate_seq_len, self.hash_table.num_ip_hashes),
                    dtype=np.int32,
                ),
                product_surface=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int32),
                client_app_id=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int32),
                post_creation_ts_sec=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int32),
                post_ids=None,
                promoted_ids=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int64),
                line_item_objective=None,
                safety_label_mask=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int64),
                embedding=None,
                search_query_embeddings=None,
                **empty_feature_arrays(batch_size, self.candidate_seq_len),
            ),
            **empty_user_feature_arrays(batch_size),
            user_installed_apps_multihot=np.zeros(
                (batch_size, NUM_USER_INSTALLED_APPS), dtype=np.bool_
            ),
        )

    @property
    def fill_token(self) -> int:
        return self.input_vocab_size

    def example_data_shape(self, batch_size: int):
        import jax

        example_data: RecsysFeaturesBatch = self.example_data(batch_size)
        user_hashes = example_data["user_hashes"]
        history_seq = example_data["history_seq"]
        candidate_seq = example_data["candidate_seq"]

        history_seq_shape = {}
        for k, v in history_seq.items():
            if v is not None and isinstance(v, np.ndarray):
                history_seq_shape[k] = jax.ShapeDtypeStruct(v.shape, v.dtype)

        candidate_seq_shape = {}
        for k, v in candidate_seq.items():
            if v is not None and isinstance(v, np.ndarray):
                candidate_seq_shape[k] = jax.ShapeDtypeStruct(v.shape, v.dtype)

        user_hashes_shape = jax.ShapeDtypeStruct(user_hashes.shape, user_hashes.dtype)

        batch_shape = {
            "user_hashes": user_hashes_shape,
            "history_seq": history_seq_shape,
            "candidate_seq": candidate_seq_shape,
            "user_categorical_features": jax.ShapeDtypeStruct(
                example_data["user_categorical_features"].shape,
                example_data["user_categorical_features"].dtype,
            ),
            "user_bool_features": jax.ShapeDtypeStruct(
                example_data["user_bool_features"].shape,
                example_data["user_bool_features"].dtype,
            ),
            "user_float_features": jax.ShapeDtypeStruct(
                example_data["user_float_features"].shape,
                example_data["user_float_features"].dtype,
            ),
            "user_int64_features": jax.ShapeDtypeStruct(
                example_data["user_int64_features"].shape,
                example_data["user_int64_features"].dtype,
            ),
            "user_installed_apps_multihot": jax.ShapeDtypeStruct(
                example_data["user_installed_apps_multihot"].shape,
                example_data["user_installed_apps_multihot"].dtype,
            ),
        }

        return batch_shape

    def example_data(
        self, batch_size: int, num_negatives_per_example: int | None = None
    ) -> RecsysFeaturesBatch:
        history_seq_len = self.history_seq_len
        num_negatives_per_example = num_negatives_per_example or self.num_negatives_per_example
        num_neg_blocks = 2 if self.search_query_embedding_dim > 0 else 1
        candidate_seq_len = self.candidate_seq_len * (
            1 + num_neg_blocks * num_negatives_per_example
        )

        history = PostSeq(
            impr_ts=np.zeros((batch_size, history_seq_len), dtype=np.int32),
            actions=np.zeros((batch_size, history_seq_len, self.output_vocab_size), dtype=np.bool_),
            continuous_actions=np.zeros(
                (batch_size, history_seq_len, self.num_continuous_actions), dtype=np.float32
            ),
            post_hashes=np.zeros(
                (batch_size, history_seq_len, self.hash_table.num_item_hashes), dtype=np.int32
            ),
            auth_hashes=np.zeros(
                (batch_size, history_seq_len, self.hash_table.num_author_hashes), dtype=np.int32
            ),
            post_ids=None,
            product_surface=np.zeros((batch_size, history_seq_len), dtype=np.int32),
            client_app_id=np.zeros((batch_size, history_seq_len), dtype=np.int32),
            post_creation_ts_sec=np.zeros((batch_size, history_seq_len), dtype=np.int32),
            promoted_ids=None,
            line_item_objective=None,
            safety_label_mask=np.zeros((batch_size, history_seq_len), dtype=np.int64),
            embedding=None,
            search_query_embeddings=None,
            **empty_feature_arrays(batch_size, history_seq_len),
        )

        candidates = PostSeq(
            impr_ts=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
            actions=np.zeros(
                (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
            ),
            post_hashes=np.zeros(
                (batch_size, candidate_seq_len, self.hash_table.num_item_hashes), dtype=np.int32
            ),
            auth_hashes=np.zeros(
                (batch_size, candidate_seq_len, self.hash_table.num_author_hashes),
                dtype=np.int32,
            ),
            post_ids=None,
            product_surface=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
            client_app_id=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
            post_creation_ts_sec=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
            continuous_actions=np.zeros((batch_size, candidate_seq_len, 2), dtype=np.float32),
            promoted_ids=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
            line_item_objective=None,
            safety_label_mask=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
            embedding=None,
            search_query_embeddings=None,
            **empty_feature_arrays(batch_size, candidate_seq_len),
        )
        return RecsysFeaturesBatch(
            user_hashes=np.zeros((batch_size, self.hash_table.num_user_hashes), dtype=np.int32),
            user_ip_hashes=np.zeros((batch_size, self.hash_table.num_ip_hashes), dtype=np.int32),
            history_seq=history,
            candidate_seq=candidates,
            **empty_user_feature_arrays(batch_size),
            user_installed_apps_multihot=np.zeros(
                (batch_size, NUM_USER_INSTALLED_APPS), dtype=np.bool_
            ),
            num_positive_candidates=None,
        )
