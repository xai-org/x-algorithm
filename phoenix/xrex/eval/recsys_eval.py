# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import copy
import functools
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from xrex.data.retrieval_dataset import PHOENIX_INDEX_BASE, RetrievalDataset
from xrex.eval.eval_utils import EvaluationTaskNew
from xrex.eval.metrics_recsys import (
    RecsysAccumulatedLossMetrics,
    RecsysRecallMetrics,
    RecsysSumMetrics,
    build_post_id_sorter,
    lookup_post_indices,
)

from xai_configlib import configclass

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


_membership_validation_done = False

_metadata_cache: tuple[tuple[Any, ...], pd.DataFrame] | None = None


def _read_metadata_table(path: Path, columns: List[str]) -> pd.DataFrame:
    global _metadata_cache
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    key = (resolved, stat.st_mtime_ns, stat.st_size, tuple(columns))
    if _metadata_cache is None or _metadata_cache[0] != key:
        _metadata_cache = None
        _metadata_cache = (key, pq.read_table(resolved, columns=columns).to_pandas())
    return _metadata_cache[1]


def target_slice_membership(
    table_dataset_types: np.ndarray,
    post_indices: np.ndarray,
    target_dataset_type: RetrievalDataset,
) -> np.ndarray:
    stored_types = np.asarray(table_dataset_types).reshape(-1)
    in_table = post_indices != -1
    stored = stored_types[np.where(in_table, post_indices, 0)]
    slice_values = np.asarray(RetrievalDataset.mask_values(target_dataset_type.value))
    return in_table & np.isin(stored, slice_values)


def _weighted_slot_scores(
    users: np.ndarray, cands: np.ndarray, slot_weights: Optional[Tuple[float, ...]]
) -> np.ndarray:
    if slot_weights is None:
        return users @ cands.T
    width = users.shape[-1]
    total = np.zeros((users.shape[0], cands.shape[0]), dtype=np.float32)
    for s, w in enumerate(slot_weights):
        if w != 0.0:
            total += w * (users @ cands[:, s * width : (s + 1) * width].T)
    return total


@functools.lru_cache(maxsize=2)
def _make_gather_in_batch_jit(mesh: jax.sharding.Mesh):
    gather_out_sharding = NamedSharding(mesh, P(("stage", "expert", "replica", "data"), None, None))

    @jax.jit
    def _gather_fn(table, indices):
        safe_indices = jnp.maximum(indices, 0)
        gathered = jnp.take(table, safe_indices, axis=0)
        return jax.lax.with_sharding_constraint(gathered, gather_out_sharding)

    return _gather_fn


@configclass
class RecsysTwoTowerEval(EvaluationTaskNew):
    metrics: Optional[
        List[Tuple[str, RecsysAccumulatedLossMetrics | RecsysRecallMetrics | RecsysSumMetrics]]
    ] = None

    train_metrics: Optional[Dict[str, float]] = None
    metric_groups: List[str] = field(
        default_factory=lambda: [
            "all",
        ]
    )
    candidate_seq_len: int = 64
    target_dataset_type: RetrievalDataset = RetrievalDataset.HOME
    query_head_weights: Optional[Tuple[float, ...]] = None
    recall_ema_half_life_reports: float = 50.0

    positive_actions: List[int] = field(default_factory=lambda: list())
    implicit_negative_actions: List[int] = field(default_factory=lambda: list())
    explicit_negative_actions: List[int] = field(default_factory=lambda: list())
    metadata_path: Path = PHOENIX_INDEX_BASE / "post_author_pairs_metadata/metadata_1day.parquet"
    metadata_stats_unique_groups: List[str] = field(
        default_factory=lambda: [
            "author_id",
        ]
    )
    metadata_stats_median_groups: List[str] = field(
        default_factory=lambda: [
            "fav_count",
            "reply_count",
            "author_followers_count",
        ]
    )
    metadata_stats_mean_groups: List[str] = field(
        default_factory=lambda: [
            "has_video",
        ]
    )

    @property
    def metadata_stats(self) -> List[str]:
        return (
            self.metadata_stats_unique_groups
            + self.metadata_stats_median_groups
            + self.metadata_stats_mean_groups
        )

    def initialize(self):
        self.metrics = self._create_retrieval_metrics()

    def clear_metrics(self):
        if self.metrics is None:
            return
        for _, metric in self.metrics:
            metric.clear()

    def run(
        self,
        train_dataset: Iterator,
        forward_fn: Callable,
        loss_fn: Callable,
        mesh: jax.sharding.Mesh,
        name=None,
        rng=None,
        **kwargs,
    ):
        _eval_bs = kwargs.get("eval_bs_per_device", 0)
        try:
            eval_bs_per_device = int(_eval_bs)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("eval_bs_per_device must be an integer") from None
        if eval_bs_per_device != _eval_bs:
            raise ValueError("eval_bs_per_device must be an integer")
        _eval_bs = eval_bs_per_device
        self.metric_groups = kwargs.get("metric_groups", self.metric_groups)
        candidate_tower_forward_fn = kwargs.get("candidate_tower_forward_fn", None)
        assert candidate_tower_forward_fn is not None
        all_post_embeddings = kwargs.get("all_post_embeddings", None)
        dataset_types = kwargs.get("dataset_types", None)
        all_post_ids = kwargs.get("all_post_ids", None)
        if dataset_types is not None and self.target_dataset_type is RetrievalDataset.HOME:
            is_home_slice = (dataset_types == RetrievalDataset.HOME_COLD.value) | (
                dataset_types == RetrievalDataset.HOME_HOT.value
            )
            dataset_types = jnp.where(is_home_slice, RetrievalDataset.HOME.value, dataset_types)
        assert all_post_embeddings is not None and all_post_ids is not None
        table_dataset_types_np = (
            None if dataset_types is None else np.asarray(jax.device_get(dataset_types))
        )
        sorted_post_ids, post_id_sort_idx = build_post_id_sorter(all_post_ids)
        _force_validate = os.environ.get("XAI_EVAL_VALIDATE_MEMBERSHIP") == "1"
        if self.metrics is None:
            self.initialize()
        rank_logger.info("start eval...")
        with mesh:
            assert rng is not None

            total_start_time = time.time()
            i = 0
            data_sharding = NamedSharding(
                mesh, P(("stage", "expert", "replica", "data"), ("seq", "model"))
            )

            _batch_pspec = P(("stage", "expert", "replica", "data"))
            _gather_in_batch_embeddings = _make_gather_in_batch_jit(mesh)

            target_dataset_types = (self.target_dataset_type.value,)
            while True:
                if self.max_steps > 0 and i >= self.max_steps:
                    break
                i += 1
                step_start_time = time.time()
                try:
                    batch = next(train_dataset)
                    is_packed = batch.get("packing_layout") is not None
                    if is_packed:
                        D, B = batch["user_hashes"].shape[:2]
                        total_C = batch["candidate_seq"]["post_hashes"].shape[1] // B
                        eval_C = self.candidate_seq_len

                        def _trim(v, D=D, B=B, total_C=total_C, eval_C=eval_C):
                            if v is None or v.ndim < 2 or v.shape[1] != B * total_C:
                                return v
                            per_user = v.reshape(D, B, total_C, *v.shape[2:])
                            trimmed = per_user[:, :, :eval_C]
                            return trimmed.reshape(D, B * eval_C, *v.shape[2:])

                        batch["candidate_seq"] = {
                            k: _trim(v) for k, v in batch["candidate_seq"].items()
                        }
                    else:
                        batch["candidate_seq"] = {
                            k: v[:, : self.candidate_seq_len] if v is not None else None
                            for k, v in batch["candidate_seq"].items()
                        }

                    def is_local_data(data):
                        def check_leaf(x):
                            return isinstance(x, np.ndarray)

                        return all(jax.tree_util.tree_leaves(jax.tree.map(check_leaf, data)))

                    if is_local_data(batch):
                        cpu_batch = batch
                        jax_batch = multihost_utils.host_local_array_to_global_array(
                            copy.deepcopy(batch),
                            mesh,
                            P(("stage", "expert", "replica", "data"), ("seq", "model")),
                        )
                    else:
                        local_train_data = multihost_utils.global_array_to_host_local_array(
                            copy.deepcopy(batch),
                            mesh,
                            P(("stage", "expert", "replica", "data"), ("seq", "model")),
                        )
                        cpu_batch = jax.tree.map(jax.device_get, local_train_data)
                        jax_batch = batch

                except Exception as e:
                    rank_logger.info(f"Eval {name} encountered error at step {i}:")
                    rank_logger.info(e)
                    break

                batch_loss, train_metrics = loss_fn(jax_batch)
                sample_scores = None
                self.train_metrics = {key: value.item() for key, value in train_metrics.items()}
                if is_packed:
                    target_post_ids = cpu_batch["candidate_seq"]["post_ids"].reshape(
                        D * B,
                        self.candidate_seq_len,
                    )
                    actions = cpu_batch["candidate_seq"]["actions"].reshape(
                        D * B,
                        self.candidate_seq_len,
                        -1,
                    )
                    local_batch_size = D * B
                else:
                    target_post_ids = cpu_batch["candidate_seq"]["post_ids"]
                    actions = cpu_batch["candidate_seq"]["actions"]
                    local_batch_size = next(iter(cpu_batch.values())).shape[0]
                in_batch_post_indices = lookup_post_indices(
                    target_post_ids, sorted_post_ids, post_id_sort_idx
                )
                global _membership_validation_done
                if not _membership_validation_done or _force_validate:
                    _ref_index = {pid: i for i, pid in enumerate(all_post_ids)}
                    _ref = np.array(
                        [[_ref_index.get(p, -1) for p in row] for row in target_post_ids]
                    )
                    if not (in_batch_post_indices == _ref).all():
                        raise AssertionError("lookup_post_indices != legacy dict lookup")
                    if not (
                        (in_batch_post_indices != -1) == np.isin(target_post_ids, all_post_ids)
                    ).all():
                        raise AssertionError("membership mask != legacy np.isin")
                    _membership_validation_done = True
                    rank_logger.info(
                        "membership lookup validated against legacy dict/np.isin paths"
                    )
                missing_mask = in_batch_post_indices == -1
                is_in_global_posts = ~missing_mask
                if table_dataset_types_np is not None:
                    is_in_target_slice = target_slice_membership(
                        table_dataset_types_np, in_batch_post_indices, self.target_dataset_type
                    )
                else:
                    is_in_target_slice = is_in_global_posts
                in_batch_post_indices_global = multihost_utils.host_local_array_to_global_array(
                    in_batch_post_indices.astype(np.int32),
                    mesh,
                    _batch_pspec,
                )
                in_batch_embeddings_global = _gather_in_batch_embeddings(
                    all_post_embeddings, in_batch_post_indices_global
                )
                _user_emb_fn = kwargs.get("user_emb_fn", None)
                assert _user_emb_fn is not None, "user_emb_fn required for in-batch recall"
                _user_emb = _user_emb_fn(jax_batch)
                if _user_emb.ndim == 3:
                    _user_emb = _user_emb[:, 0]
                _local_start = jax.process_index() * local_batch_size
                _user_local = np.asarray(
                    jax.device_get(_user_emb[_local_start : _local_start + local_batch_size])
                ).astype(np.float32)
                _cand = np.asarray(
                    jax.device_get(
                        in_batch_embeddings_global[_local_start : _local_start + local_batch_size]
                    )
                ).astype(np.float32)
                _cand[missing_mask] = 0.0
                _cand_flat = _cand.reshape(local_batch_size * self.candidate_seq_len, -1)
                in_batch_top_k_scores = _weighted_slot_scores(
                    _user_local, _cand_flat, self.query_head_weights
                )

                global_top_large_k_indices, _ = forward_fn(
                    jax_batch,
                    all_post_embeddings,
                    dataset_types,
                    1_000,
                    target_dataset_types,
                    _eval_bs,
                )
                global_top_large_k_indices = jax.device_get(global_top_large_k_indices)
                if is_local_data(batch):
                    local_rank = jax.process_index()
                    global_local_batch_size = (
                        global_top_large_k_indices.shape[0] // jax.process_count()
                    )
                    global_start = local_rank * global_local_batch_size
                    global_top_large_k_indices = global_top_large_k_indices[
                        global_start : global_start + global_local_batch_size
                    ]

                global_top_large_k_ids = all_post_ids[global_top_large_k_indices]
                _gr_n = global_top_large_k_ids.shape[0]
                if _eval_bs > 0 and _gr_n < target_post_ids.shape[0]:
                    _local_experts = _gr_n // _eval_bs
                    _users_per_expert = target_post_ids.shape[0] // _local_experts
                    global_recall_target_post_ids = target_post_ids.reshape(
                        _local_experts, _users_per_expert, self.candidate_seq_len
                    )[:, :_eval_bs].reshape(_gr_n, self.candidate_seq_len)
                    global_recall_actions = actions.reshape(
                        _local_experts, _users_per_expert, self.candidate_seq_len, -1
                    )[:, :_eval_bs].reshape(_gr_n, self.candidate_seq_len, actions.shape[-1])
                    global_recall_is_in = is_in_global_posts.reshape(
                        _local_experts, _users_per_expert, self.candidate_seq_len
                    )[:, :_eval_bs].reshape(_gr_n, self.candidate_seq_len)
                    global_recall_in_slice = is_in_target_slice.reshape(
                        _local_experts, _users_per_expert, self.candidate_seq_len
                    )[:, :_eval_bs].reshape(_gr_n, self.candidate_seq_len)
                else:
                    global_recall_target_post_ids = target_post_ids[:_gr_n]
                    global_recall_actions = actions[:_gr_n]
                    global_recall_is_in = is_in_global_posts[:_gr_n]
                    global_recall_in_slice = is_in_target_slice[:_gr_n]

                columns = ["post_id"] + self.metadata_stats
                for attempt in range(3):
                    try:
                        metadata_table = _read_metadata_table(self.metadata_path, columns)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise e
                        logger.warning(f"Error reading metadata table: {e}, retrying...")
                        time.sleep(5)
                global_top_large_k_metadata_table = self.get_metadata_for_post_ids(
                    global_top_large_k_ids, metadata_table
                )

                assert self.metrics is not None
                real_candidate_mask = target_post_ids != 0
                in_index_mask = (~missing_mask) & real_candidate_mask
                has_positive_actions = (
                    np.sum(actions[:, :, self.positive_actions], axis=-1) > 0
                    if self.positive_actions
                    else np.zeros_like(real_candidate_mask, dtype=bool)
                )
                positive_mask = has_positive_actions & real_candidate_mask
                positive_in_index_mask = positive_mask & in_index_mask
                positive_in_slice_mask = positive_mask & is_in_target_slice

                num_real_candidates = float(real_candidate_mask.sum())
                num_candidates_in_index = float(in_index_mask.sum())
                num_positives = float(positive_mask.sum())
                num_positives_in_index = float(positive_in_index_mask.sum())
                num_positives_in_slice = float(positive_in_slice_mask.sum())

                for key, metric in self.metrics:
                    if key == "loss":
                        assert isinstance(metric, RecsysAccumulatedLossMetrics)
                        metric.add(batch_loss)
                    elif key == "candidate_in_index_frac":
                        assert isinstance(metric, RecsysRecallMetrics)
                        metric.add_ratio_counts(num_candidates_in_index, num_real_candidates)
                    elif key == "positive_in_index_frac":
                        assert isinstance(metric, RecsysRecallMetrics)
                        metric.add_ratio_counts(num_positives_in_index, num_positives)
                    elif key == "num_positives":
                        assert isinstance(metric, RecsysSumMetrics)
                        metric.add(num_positives)
                    elif key == "num_positives_in_index":
                        assert isinstance(metric, RecsysSumMetrics)
                        metric.add(num_positives_in_index)
                    elif key == "positive_in_slice_frac":
                        assert isinstance(metric, RecsysRecallMetrics)
                        metric.add_ratio_counts(num_positives_in_slice, num_positives)
                    elif key == "num_positives_in_slice":
                        assert isinstance(metric, RecsysSumMetrics)
                        metric.add(num_positives_in_slice)
                    elif key == "num_candidates":
                        assert isinstance(metric, RecsysSumMetrics)
                        metric.add(num_real_candidates)
                    elif key == "num_candidates_in_index":
                        assert isinstance(metric, RecsysSumMetrics)
                        metric.add(num_candidates_in_index)
                    elif key.startswith("in_batch_recall@"):
                        assert isinstance(metric, RecsysRecallMetrics)
                        metric.run_in_batch_recall_by_forward_fn(
                            in_batch_top_k_scores, ~missing_mask, actions, self.positive_actions
                        )

                    elif key.startswith("global_recall@"):
                        assert isinstance(
                            metric,
                            RecsysRecallMetrics,
                        )
                        global_top_k_ids = global_top_large_k_ids[:, : int(key.split("@")[1])]

                        metric.run_batch_global_recall(
                            global_top_k_ids,
                            global_recall_target_post_ids,
                            global_recall_actions,
                            self.positive_actions,
                            self.implicit_negative_actions + self.explicit_negative_actions,
                            global_post_ids=all_post_ids,
                            is_in_global_posts=global_recall_is_in,
                        )

                    elif key.startswith("slice_recall@"):
                        assert isinstance(metric, RecsysRecallMetrics)
                        slice_top_k_ids = global_top_large_k_ids[:, : int(key.split("@")[1])]
                        metric.run_batch_global_recall(
                            slice_top_k_ids,
                            global_recall_target_post_ids,
                            global_recall_actions,
                            self.positive_actions,
                            self.implicit_negative_actions + self.explicit_negative_actions,
                            global_post_ids=all_post_ids,
                            is_in_global_posts=global_recall_in_slice,
                        )

                    elif key.startswith("implicit_negative_global_recall@"):
                        assert isinstance(
                            metric,
                            RecsysRecallMetrics,
                        )
                        implicit_negative_global_top_k_ids = global_top_large_k_ids[
                            :, : int(key.split("@")[1])
                        ]

                        metric.run_batch_global_recall(
                            implicit_negative_global_top_k_ids,
                            global_recall_target_post_ids,
                            global_recall_actions,
                            positive_actions=self.implicit_negative_actions,
                            negative_actions=self.positive_actions,
                            global_post_ids=all_post_ids,
                            is_in_global_posts=global_recall_is_in,
                        )

                    elif key.startswith("explicit_negative_global_recall@"):
                        assert isinstance(
                            metric,
                            RecsysRecallMetrics,
                        )
                        explicit_negative_global_top_k_ids = global_top_large_k_ids[
                            :, : int(key.split("@")[1])
                        ]

                        metric.run_batch_global_recall(
                            explicit_negative_global_top_k_ids,
                            global_recall_target_post_ids,
                            global_recall_actions,
                            positive_actions=self.explicit_negative_actions,
                            negative_actions=self.positive_actions,
                            global_post_ids=all_post_ids,
                            is_in_global_posts=global_recall_is_in,
                        )

                    elif key.startswith("global_recall_stat@"):
                        assert isinstance(metric, RecsysAccumulatedLossMetrics)
                        _stat = key.split("@")[1]
                        _k = int(key.split("@")[2])
                        global_top_k_ids_flattened = global_top_large_k_ids[:, :_k].flatten()
                        _ids_and_counts = np.unique_counts(global_top_k_ids_flattened)
                        _ids_and_counts_mapping = dict(
                            zip(_ids_and_counts.values, _ids_and_counts.counts)
                        )
                        global_top_k_metadata = global_top_large_k_metadata_table[
                            global_top_large_k_metadata_table["post_id"].isin(
                                _ids_and_counts.values
                            )
                        ].copy()
                        if len(global_top_k_metadata) == 0:
                            logger.warning(
                                f"No metadata for stat {_stat} found for global top {_k} ids"
                            )
                            continue
                        global_top_k_metadata["unique_count"] = global_top_k_metadata[
                            "post_id"
                        ].map(_ids_and_counts_mapping)
                        if _stat in self.metadata_stats_unique_groups:
                            val = np.float64(
                                np.unique(np.array(global_top_k_metadata[_stat])).shape[0]
                                / len(global_top_k_ids_flattened)
                            )
                        elif _stat in self.metadata_stats_mean_groups:
                            val = np.mean(
                                np.repeat(
                                    np.array(global_top_k_metadata[_stat]),
                                    global_top_k_metadata["unique_count"],
                                )
                            )
                        elif _stat in self.metadata_stats_median_groups:
                            val = np.median(
                                np.repeat(
                                    np.array(global_top_k_metadata[_stat]),
                                    global_top_k_metadata["unique_count"],
                                )
                            )
                        else:
                            raise KeyError(f"Invalid metadata stat: {_stat}")
                        metric.add(val)

                    elif key.startswith("global_random_scores"):
                        assert isinstance(metric, RecsysAccumulatedLossMetrics)
                        if sample_scores is None:
                            rng, init_rng = jax.random.split(rng)
                            num_samples = 2**14 - 2**14 % data_sharding.num_devices
                            sample_indices = jax.random.choice(
                                init_rng,
                                all_post_embeddings.shape[0],
                                shape=(num_samples,),
                                replace=False,
                            )
                            sample_embeddings = all_post_embeddings[sample_indices]
                            sample_embeddings = jax.device_put(sample_embeddings, data_sharding)
                            _, sample_scores = forward_fn(
                                jax_batch,
                                sample_embeddings,
                                jnp.full(
                                    (sample_embeddings.shape[0], 1), self.target_dataset_type.value
                                ),
                                sample_embeddings.shape[0],
                                target_dataset_types,
                            )
                        sample_scores_flat = sample_scores.flatten()
                        if "mean" in key:
                            metric.add(jnp.mean(sample_scores_flat))
                        elif "max" in key:
                            metric.add(jnp.max(sample_scores_flat))
                        else:
                            raise ValueError(f"Invalid metric key: {key}")
                    else:
                        raise ValueError(f"Invalid metric key: {key}")
                rank_logger.info(f"eval step {i} for eval {name}")
                rank_logger.info(f"step {i} took {time.time() - step_start_time:.2f} seconds")

        total_time = time.time() - total_start_time
        rank_logger.info(f"Total evaluation time: {total_time:.2f} seconds")
        return self

    def _create_retrieval_metrics(self):
        if "all" in self.metric_groups:
            self.metric_groups = [
                "loss",
                "index_overlap",
                "in_batch_recall",
                "global_recall",
                "negative_global_recall",
                "global_random_scores",
            ]
        metrics = []
        global_recalled_stats = self.metadata_stats

        if "loss" in self.metric_groups:
            metrics.append(("loss", RecsysAccumulatedLossMetrics()))

        ema_hl = self.recall_ema_half_life_reports
        if "index_overlap" in self.metric_groups:
            for name, metric in (
                ("candidate_in_index_frac", RecsysRecallMetrics(k=1, ema_half_life_reports=ema_hl)),
                ("positive_in_index_frac", RecsysRecallMetrics(k=1, ema_half_life_reports=ema_hl)),
                ("num_candidates", RecsysSumMetrics()),
                ("num_candidates_in_index", RecsysSumMetrics()),
                ("num_positives", RecsysSumMetrics()),
                ("num_positives_in_index", RecsysSumMetrics()),
                ("positive_in_slice_frac", RecsysRecallMetrics(k=1, ema_half_life_reports=ema_hl)),
                ("num_positives_in_slice", RecsysSumMetrics()),
            ):
                if isinstance(metric, RecsysRecallMetrics):
                    metric.initialize()
                metrics.append((name, metric))

        if "in_batch_recall" in self.metric_groups:
            for k in [1, 10, 100, 1000]:
                recall_metric = RecsysRecallMetrics(k, ema_half_life_reports=ema_hl)
                recall_metric.initialize()
                metrics.append((f"in_batch_recall@{k}", recall_metric))

        if "global_recall" in self.metric_groups:
            for k in [10, 100, 600, 1_000]:
                recall_metric = RecsysRecallMetrics(k, ema_half_life_reports=ema_hl)
                recall_metric.initialize()
                metrics.append((f"global_recall@{k}", recall_metric))

                slice_recall_metric = RecsysRecallMetrics(k, ema_half_life_reports=ema_hl)
                slice_recall_metric.initialize()
                metrics.append((f"slice_recall@{k}", slice_recall_metric))

                implicit_negative_recall_metric = RecsysRecallMetrics(
                    k, ema_half_life_reports=ema_hl
                )
                implicit_negative_recall_metric.initialize()
                metrics.append(
                    (f"implicit_negative_global_recall@{k}", implicit_negative_recall_metric)
                )

                explicit_negative_recall_metric = RecsysRecallMetrics(
                    k, ema_half_life_reports=ema_hl
                )
                explicit_negative_recall_metric.initialize()
                metrics.append(
                    (f"explicit_negative_global_recall@{k}", explicit_negative_recall_metric)
                )

                for stat in global_recalled_stats:
                    if stat == "author_id":
                        metrics.append(
                            (f"global_recall_stat@{stat}@{k}", RecsysAccumulatedLossMetrics())
                        )
                    else:
                        metrics.append(
                            (f"global_recall_stat@{stat}@{k}", RecsysAccumulatedLossMetrics())
                        )

        if "global_random_scores" in self.metric_groups:
            metrics.append(("global_random_scores_mean", RecsysAccumulatedLossMetrics()))
            metrics.append(("global_random_scores_max", RecsysAccumulatedLossMetrics()))

        return metrics

    def get_metadata_for_post_ids(self, post_ids: np.ndarray, metadata_table) -> pd.DataFrame:
        if len(post_ids) == 0:
            raise ValueError("post_ids is empty")

        unique_ids = np.unique(post_ids.flatten())
        return metadata_table[metadata_table["post_id"].isin(unique_ids)]


def run_recsys_evals(
    evals: List[Tuple[str, RecsysTwoTowerEval]],
    train_dataset: Iterable,
    forward_fn: Callable,
    loss_fn: Callable,
    mesh: jax.sharding.Mesh,
    **kwargs,
) -> Dict[str, EvaluationTaskNew]:
    return {
        name: conf.run(train_dataset, forward_fn, loss_fn, mesh, name=name, **kwargs)
        for name, conf in evals
    }
