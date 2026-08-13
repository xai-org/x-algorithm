# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import copy
import functools
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from xai_configlib import configclass

from xrex.data.retrieval_dataset import PHOENIX_INDEX_BASE, RetrievalDataset
from xrex.eval.eval_utils import EvaluationTaskNew
from xrex.eval.metrics_recsys import (
    RecsysAccumulatedLossMetrics,
    RecsysRecallMetrics,
    RecsysSumMetrics,
    build_post_id_sorter,
    lookup_post_indices,
)

if TYPE_CHECKING:
    from xrex.inference.sid_post_index import SidBeamResolver

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


_membership_validation_done = False


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
    metrics: list[tuple[str, RecsysAccumulatedLossMetrics | RecsysRecallMetrics | RecsysSumMetrics]] | None = None

    train_metrics: dict[str, float] | None = None
    metric_groups: list[str] = field(
        default_factory=lambda: [
            "all",
        ]
    )
    candidate_seq_len: int = 64
    target_dataset_type: RetrievalDataset = RetrievalDataset.HOME
    recall_ema_half_life_reports: float = 50.0

    positive_actions: list[int] = field(default_factory=lambda: list())
    implicit_negative_actions: list[int] = field(default_factory=lambda: list())
    explicit_negative_actions: list[int] = field(default_factory=lambda: list())
    metadata_path: Path = PHOENIX_INDEX_BASE / "post_author_pairs_metadata/metadata_1day.parquet"
    metadata_stats_unique_groups: list[str] = field(
        default_factory=lambda: [
            "author_id",
        ]
    )
    metadata_stats_median_groups: list[str] = field(
        default_factory=lambda: [
            "fav_count",
            "reply_count",
            "author_followers_count",
        ]
    )
    metadata_stats_mean_groups: list[str] = field(
        default_factory=lambda: [
            "has_video",
        ]
    )

    @property
    def metadata_stats(self) -> list[str]:
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
        self.metric_groups = kwargs.get("metric_groups", self.metric_groups)
        candidate_tower_forward_fn = kwargs.get("candidate_tower_forward_fn")
        assert candidate_tower_forward_fn is not None
        all_post_embeddings = kwargs.get("all_post_embeddings")
        dataset_types = kwargs.get("dataset_types")
        all_post_ids = kwargs.get("all_post_ids")
        assert all_post_embeddings is not None and all_post_ids is not None
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
                in_batch_post_indices_global = multihost_utils.host_local_array_to_global_array(
                    in_batch_post_indices.astype(np.int32),
                    mesh,
                    _batch_pspec,
                )
                in_batch_embeddings_global = _gather_in_batch_embeddings(
                    all_post_embeddings, in_batch_post_indices_global
                )
                _user_emb_fn = kwargs.get("user_emb_fn")
                assert _user_emb_fn is not None, "user_emb_fn required for in-batch recall"
                _user_emb = _user_emb_fn(jax_batch)
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
                in_batch_top_k_scores = _user_local @ _cand_flat.T

                _eval_bs = kwargs.get("eval_bs_per_device", 0)
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
                _eval_bs = kwargs.get("eval_bs_per_device", 0)
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
                else:
                    global_recall_target_post_ids = target_post_ids[:_gr_n]
                    global_recall_actions = actions[:_gr_n]
                    global_recall_is_in = is_in_global_posts[:_gr_n]

                columns = ["post_id"] + self.metadata_stats
                for attempt in range(3):
                    try:
                        metadata_table = pq.read_table(
                            self.metadata_path, columns=columns
                        ).to_pandas()
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

                num_real_candidates = float(real_candidate_mask.sum())
                num_candidates_in_index = float(in_index_mask.sum())
                num_positives = float(positive_mask.sum())
                num_positives_in_index = float(positive_in_index_mask.sum())

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
                            val = jnp.array(
                                np.unique(np.array(global_top_k_metadata[_stat])).shape[0]
                                / len(global_top_k_ids_flattened)
                            )
                        elif _stat in self.metadata_stats_mean_groups:
                            val = jnp.mean(
                                np.repeat(
                                    np.array(global_top_k_metadata[_stat]),
                                    global_top_k_metadata["unique_count"],
                                )
                            )
                        elif _stat in self.metadata_stats_median_groups:
                            val = jnp.median(
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
                            sample_indices = jax.random.choice(
                                init_rng,
                                all_post_embeddings.shape[0],
                                shape=(2**14,),
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
            for k in [10, 100, 1_000]:
                recall_metric = RecsysRecallMetrics(k, ema_half_life_reports=ema_hl)
                recall_metric.initialize()
                metrics.append((f"global_recall@{k}", recall_metric))

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


@configclass
class RecsysSIDRetrievalEval(EvaluationTaskNew):
    num_levels: int = 6
    beam_width: int = 128
    positive_actions: list[int] = field(default_factory=lambda: list())
    hard_negative_actions: list[int] = field(default_factory=lambda: list())

    train_metrics: dict[str, float] | None = None

    _sid_resolver: Optional["SidBeamResolver"] = None
    _beam_depth_hits: Any = None
    _beam_num_valid: int = 0
    _post_recall_hits: int = 0
    _post_in_corpus: int = 0
    _post_recall_in_corpus_hits: int = 0
    _post_recall_given_sid_hit: int = 0
    _sid_full_hits: int = 0
    _resolve_n_beams: int = 0
    _resolve_n_empty: int = 0
    _resolve_n_collision: int = 0
    _resolve_n_filled: int = 0
    _resolve_wall_s: float = 0.0
    _actual_batches: int = 0

    def initialize(self):
        self._sid_resolver = None
        self._reset_accumulators()

    def set_resolver(self, resolver: "SidBeamResolver") -> None:
        self._sid_resolver = resolver

    def _reset_accumulators(self):
        self._beam_depth_hits = np.zeros(self.num_levels, dtype=np.float64)
        self._beam_num_valid = 0
        self._post_recall_hits = 0
        self._post_in_corpus = 0
        self._post_recall_in_corpus_hits = 0
        self._post_recall_given_sid_hit = 0
        self._sid_full_hits = 0
        self._resolve_n_beams = 0
        self._resolve_n_empty = 0
        self._resolve_n_collision = 0
        self._resolve_n_filled = 0
        self._resolve_wall_s = 0.0
        self._actual_batches = 0

    def clear_metrics(self):
        self._reset_accumulators()

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
        del forward_fn, loss_fn, rng
        beam_forward_fn = kwargs.get("beam_forward_fn")
        if beam_forward_fn is None:
            raise ValueError("RecsysSIDRetrievalEval requires beam_forward_fn")
        if kwargs.get("sid_resolver") is not None:
            self._sid_resolver = kwargs["sid_resolver"]
        if "host_local_batch_size" not in kwargs:
            raise ValueError(
                "RecsysSIDRetrievalEval requires host_local_batch_size "
                "(from model_config.eval_bs_per_device)"
            )
        host_local_b = int(kwargs["host_local_batch_size"])
        if host_local_b <= 0:
            raise ValueError(f"host_local_batch_size must be > 0, got {host_local_b}")

        self._reset_accumulators()
        has_resolve = self._sid_resolver is not None
        rank_logger.info(
            "start sid retrieval eval %s (beam_width=%d, host_local_B=%d, post_recall=%s)...",
            name,
            self.beam_width,
            host_local_b,
            has_resolve,
        )
        data_pspec = P(("stage", "expert", "replica", "data"), ("seq", "model"))

        with mesh:
            total_start = time.time()
            i = 0
            while True:
                if self.max_steps > 0 and i >= self.max_steps:
                    break
                i += 1
                step_start = time.time()
                try:
                    batch = next(train_dataset)

                    def is_local_data(data):
                        return all(
                            jax.tree_util.tree_leaves(
                                jax.tree.map(lambda x: isinstance(x, np.ndarray), data)
                            )
                        )

                    if is_local_data(batch):
                        cpu_batch = batch
                    else:
                        local_train_data = multihost_utils.global_array_to_host_local_array(
                            copy.deepcopy(batch), mesh, data_pspec
                        )
                        cpu_batch = jax.tree.map(jax.device_get, local_train_data)

                    cpu_batch = self._fit_host_local_batch(cpu_batch, host_local_b)
                    jax_batch = multihost_utils.host_local_array_to_global_array(
                        copy.deepcopy(cpu_batch), mesh, data_pspec
                    )
                except StopIteration:
                    rank_logger.info(f"  Dataset exhausted after {i - 1} eval batches")
                    break
                except Exception as e:
                    rank_logger.info(f"Eval {name} encountered error at step {i}: {e}")
                    break

                self._actual_batches += 1

                out = beam_forward_fn(jax_batch)
                beam_codes_jax = self._normalize_beam_codes(out)
                beam_codes_np = np.array(
                    jax.device_get(
                        multihost_utils.global_array_to_host_local_array(
                            beam_codes_jax, mesh, data_pspec
                        )
                    ),
                    dtype=np.int32,
                )
                if beam_codes_np.shape[0] != host_local_b:
                    raise ValueError(
                        f"beam codes host-local B={beam_codes_np.shape[0]} != "
                        f"host_local_batch_size={host_local_b}"
                    )
                self._accumulate_beam(cpu_batch, beam_codes_np)

                rank_logger.info(f"sid eval step {i} took {time.time() - step_start:.2f}s")

            total_time = time.time() - total_start
            rank_logger.info(f"sid retrieval eval: {total_time:.2f}s")

        return self

    @staticmethod
    def _normalize_beam_codes(out) -> jax.Array:
        codes = out[0] if isinstance(out, (tuple, list)) else out
        if codes.ndim == 2:
            return codes[:, None, :]
        return codes

    def _extract_targets(self, cpu_batch):
        _raw_sids = (
            jax.device_get(cpu_batch["candidate_seq"]["post_sids"])
            if isinstance(cpu_batch["candidate_seq"]["post_sids"], jax.Array)
            else cpu_batch["candidate_seq"]["post_sids"]
        )
        cand_sids = np.array(_raw_sids, dtype=np.int32) - 1
        cand_actions = np.array(
            jax.device_get(cpu_batch["candidate_seq"]["actions"])
            if isinstance(cpu_batch["candidate_seq"]["actions"], jax.Array)
            else cpu_batch["candidate_seq"]["actions"]
        )
        _raw_pids = cpu_batch["candidate_seq"].get("post_ids")
        if _raw_pids is None:
            raise ValueError(
                "candidate_seq.post_ids is None — set include_candidate_post_ids=True "
                "for post_recall@K (SIDs alone are not enough)"
            )
        if isinstance(_raw_pids, jax.Array):
            _raw_pids = jax.device_get(_raw_pids)
        cand_post_ids = np.asarray(_raw_pids, dtype=np.int64)
        has_positive = np.sum(cand_actions[:, :, self.positive_actions], axis=-1) > 0
        has_hard_neg = np.sum(cand_actions[:, :, self.hard_negative_actions], axis=-1) > 0
        has_sid = np.all(cand_sids >= 0, axis=-1)
        valid_positive = has_positive & ~has_hard_neg & has_sid
        return cand_sids, cand_post_ids, valid_positive

    def _accumulate_beam(self, cpu_batch, beam_codes_np):
        cand_sids, cand_post_ids, valid_positive = self._extract_targets(cpu_batch)
        B = beam_codes_np.shape[0]
        K = beam_codes_np.shape[1]

        resolved_post_ids = None
        if self._sid_resolver is not None:
            t_res = time.time()
            resolved_post_ids, rstats = self._sid_resolver.resolve_to_post_ids(
                beam_codes_np, max_posts=self.beam_width
            )
            self._resolve_wall_s += time.time() - t_res
            self._resolve_n_beams += rstats.n_beams
            self._resolve_n_empty += rstats.n_empty
            self._resolve_n_collision += rstats.n_collision
            self._resolve_n_filled += rstats.n_filled

        for user_i in range(B):
            valid_mask = valid_positive[user_i]
            if not np.any(valid_mask):
                continue
            first_pos_idx = int(np.argmax(valid_mask))
            target = cand_sids[user_i, first_pos_idx, :]
            target_post = int(cand_post_ids[user_i, first_pos_idx])
            self._beam_num_valid += 1

            sid_full_hit = False
            for d in range(1, self.num_levels + 1):
                target_prefix = target[:d]
                for k in range(K):
                    if np.array_equal(beam_codes_np[user_i, k, :d], target_prefix):
                        self._beam_depth_hits[d - 1] += 1
                        if d == self.num_levels:
                            sid_full_hit = True
                        break

            if sid_full_hit:
                self._sid_full_hits += 1

            if resolved_post_ids is not None:
                slate = resolved_post_ids[user_i]
                post_hit = bool(np.any(slate == target_post))
                if post_hit:
                    self._post_recall_hits += 1
                resolver = self._sid_resolver
                assert resolver is not None
                in_corpus = target_post in resolver.post_id_set
                if in_corpus:
                    self._post_in_corpus += 1
                    if post_hit:
                        self._post_recall_in_corpus_hits += 1
                if sid_full_hit and post_hit:
                    self._post_recall_given_sid_hit += 1

    @staticmethod
    def _fit_host_local_batch(cpu_batch, target: int):
        def _fit(x):
            if not hasattr(x, "shape") or len(getattr(x, "shape", ())) == 0:
                return x
            arr = np.asarray(x)
            b = int(arr.shape[0])
            if b == target:
                return arr
            if b > target:
                return arr[:target]
            pad_width = [(0, target - b)] + [(0, 0)] * (arr.ndim - 1)
            return np.pad(arr, pad_width)

        return jax.tree.map(_fit, cpu_batch)

    @staticmethod
    def _multihost_sum_scalar(value: float | int) -> float:
        local = np.array([float(value)], dtype=np.float64)
        if jax.process_count() > 1:
            return float(np.asarray(multihost_utils.process_allgather(local, tiled=True)).sum())
        return float(value)

    def collect_metrics(self) -> dict[str, float]:
        num_batches = float(self._actual_batches)
        beam_num_valid = self._multihost_sum_scalar(self._beam_num_valid)
        depth_hits = np.array(
            [self._multihost_sum_scalar(self._beam_depth_hits[d]) for d in range(self.num_levels)],
            dtype=np.float64,
        )
        post_recall_hits = self._multihost_sum_scalar(self._post_recall_hits)
        post_in_corpus = self._multihost_sum_scalar(self._post_in_corpus)
        post_recall_in_corpus_hits = self._multihost_sum_scalar(self._post_recall_in_corpus_hits)
        post_recall_given_sid_hit = self._multihost_sum_scalar(self._post_recall_given_sid_hit)
        sid_full_hits = self._multihost_sum_scalar(self._sid_full_hits)
        resolve_n_beams = self._multihost_sum_scalar(self._resolve_n_beams)
        resolve_n_empty = self._multihost_sum_scalar(self._resolve_n_empty)
        resolve_n_collision = self._multihost_sum_scalar(self._resolve_n_collision)
        resolve_n_filled = self._multihost_sum_scalar(self._resolve_n_filled)

        metrics: dict[str, float] = {
            "eval/sid/num_batches": num_batches,
            "eval/sid/beam_num_valid_users": beam_num_valid,
        }
        bn = max(beam_num_valid, 1.0)
        for d in range(1, self.num_levels + 1):
            metrics[f"eval/sid/hit@{self.beam_width}_depth{d}"] = float(depth_hits[d - 1] / bn)
        if self._sid_resolver is not None:
            k = self.beam_width
            metrics[f"eval/sid/post_recall@{k}"] = float(post_recall_hits / bn)
            metrics["eval/sid/post_in_corpus_pct"] = float(post_in_corpus / bn)
            ic = max(post_in_corpus, 1.0)
            metrics[f"eval/sid/post_recall_in_corpus@{k}"] = float(post_recall_in_corpus_hits / ic)
            sh = max(sid_full_hits, 1.0)
            metrics[f"eval/sid/post_recall_given_hit@{k}"] = float(post_recall_given_sid_hit / sh)
            rb = max(resolve_n_beams, 1.0)
            metrics["eval/sid/resolve_empty_frac"] = float(resolve_n_empty / rb)
            metrics["eval/sid/resolve_collision_frac"] = float(resolve_n_collision / rb)
            metrics["eval/sid/resolve_filled_per_user"] = float(resolve_n_filled / bn)
            metrics["eval/sid/resolve_wall_s"] = float(self._resolve_wall_s)
        return metrics


def sid_eval_batch_geometry(eval_bs_per_device: int, num_devices: int) -> tuple[int, int, int]:
    if eval_bs_per_device <= 0:
        raise ValueError(f"eval_bs_per_device must be > 0, got {eval_bs_per_device}")
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")
    n_proc = max(int(jax.process_count()), 1)
    eval_global_b = int(eval_bs_per_device) * int(num_devices)
    if eval_global_b % n_proc != 0:
        raise ValueError(f"eval global B={eval_global_b} not divisible by process_count={n_proc}")
    return int(eval_bs_per_device), eval_global_b, eval_global_b // n_proc


def make_sid_retrieval_eval_task(
    *,
    max_steps: int,
    num_levels: int,
    beam_width: int,
    sid_resolver: Optional["SidBeamResolver"] = None,
) -> RecsysSIDRetrievalEval:
    from xrex.models.recsys_sid_retrieval_model import (
        HARD_NEGATIVE_ACTIONS,
        POSITIVE_ACTIONS,
    )

    task = RecsysSIDRetrievalEval(
        max_steps=max_steps,
        num_levels=num_levels,
        beam_width=beam_width,
        positive_actions=list(POSITIVE_ACTIONS),
        hard_negative_actions=list(HARD_NEGATIVE_ACTIONS),
    )
    task.initialize()
    if sid_resolver is not None:
        task.set_resolver(sid_resolver)
    return task


def finalize_sid_eval_metrics(
    eval_task: RecsysSIDRetrievalEval,
    *,
    corpus_load_s: float,
    beam_wall_s: float,
    total_wall_s: float,
    sid_resolver: "SidBeamResolver",
    eval_bs_per_device: int,
) -> dict[str, float]:
    metrics = eval_task.collect_metrics()
    metrics["eval/sid/corpus_load_s"] = float(corpus_load_s)
    metrics["eval/sid/corpus_num_posts"] = float(sid_resolver.num_posts)
    metrics["eval/sid/corpus_num_unique_sids"] = float(len(sid_resolver.sid_to_post_idx))
    metrics["eval/sid/beam_eval_wall_s"] = float(beam_wall_s)
    metrics["eval/sid/total_eval_wall_s"] = float(total_wall_s)
    metrics["eval/sid/corpus_load_frac"] = float(corpus_load_s / max(total_wall_s, 1e-6))
    metrics["eval/sid/eval_bs_per_device"] = float(eval_bs_per_device)
    return metrics


def log_sid_eval_summary(
    metrics: dict[str, float],
    *,
    label: str,
    soft_step: int | None = None,
    posts: int | None = None,
) -> None:
    posts_n = posts if posts is not None else int(metrics.get("eval/sid/corpus_num_posts", 0))
    total_wall_s = float(metrics.get("eval/sid/total_eval_wall_s", 0.0))
    corpus_load_s = float(metrics.get("eval/sid/corpus_load_s", 0.0))
    beam_wall_s = float(metrics.get("eval/sid/beam_eval_wall_s", 0.0))
    if soft_step is not None:
        rank_logger.info(
            "%s (step %s, %d users, total=%.1fs corpus_load=%.2fs (%.0f%%) beam=%.1fs, posts=%d):",
            label,
            soft_step,
            int(metrics.get("eval/sid/beam_num_valid_users", 0)),
            total_wall_s,
            corpus_load_s,
            100.0 * corpus_load_s / max(total_wall_s, 1e-6),
            beam_wall_s,
            posts_n,
        )
    else:
        rank_logger.info(
            "%s (%d users, total=%.1fs corpus_load=%.2fs (%.0f%%) beam=%.1fs, posts=%d):",
            label,
            int(metrics.get("eval/sid/beam_num_valid_users", 0)),
            total_wall_s,
            corpus_load_s,
            100.0 * corpus_load_s / max(total_wall_s, 1e-6),
            beam_wall_s,
            posts_n,
        )
    for k, v in sorted(metrics.items()):
        rank_logger.info(f"  {k}: {v:.4f}")


def run_recsys_evals(
    evals: list[tuple[str, RecsysTwoTowerEval]],
    train_dataset: Iterable,
    forward_fn: Callable,
    loss_fn: Callable,
    mesh: jax.sharding.Mesh,
    **kwargs,
) -> dict[str, EvaluationTaskNew]:
    return {
        name: conf.run(train_dataset, forward_fn, loss_fn, mesh, name=name, **kwargs)
        for name, conf in evals
    }
