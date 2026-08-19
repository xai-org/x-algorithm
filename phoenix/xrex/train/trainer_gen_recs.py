# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import copy
import logging
import os
import time
import typing
from dataclasses import field

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from xai_configlib import configclass

from xrex.data.recsys.gen_recs_semantics import (
    build_recall_metrics,
    l2_normalize,
)
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.model_utils import get_parameter
from xrex.models.recsys_embedding import (
    RecsysEmbeddings,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_gen_recs_model import RecsysGenRecsModelConfig
from xrex.models.recsys_model import block_candidate_reduce
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.train.misc import PostEmbeddings, RecsysTrainingState
from xrex.train.trainer_recsys import RecsysTrainer
from xrex.utils.aot import JittedOrCompiled

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@configclass
class GenRecsTrainer(RecsysTrainer):
    gen_recs_eval_forward_fn: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    gen_recs_project_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _gen_recs_last_resolved_parquet: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_source_post_ids: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_source_author_ids: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_source_mm_embeddings: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_source_pid_to_idx: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_eval_table_embs: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_eval_table_post_ids: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_eval_table_pid_set: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    _gen_recs_eval_table_size: int = field(init=False, repr=False, compare=False, default=0)
    _gen_recs_post_emb_built: bool = field(init=False, repr=False, compare=False, default=False)
    _gen_recs_eval_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)

    @property
    def using_seqpack(self) -> bool:
        return False

    def init(self, batch: RecsysFeaturesBatch, rng: jax.Array) -> RecsysTrainingState:
        state = super().init(batch, rng)
        return state._replace(
            post_embeddings=self._family_init_post_embeddings(state.post_embeddings)
        )

    def create_executable(self) -> None:
        super().create_executable()
        self._family_create_executable_hook()
        self._family_post_sharding_hook()

    def _family_init_post_embeddings(self, post_embeddings):
        if isinstance(self.model_config, RecsysGenRecsModelConfig):
            post_embeddings = self.model_config.make_post_embeddings()
        return post_embeddings

    def _family_create_executable_hook(self) -> None:
        if isinstance(self.model_config, RecsysGenRecsModelConfig):

            @hk.transform
            def gen_recs_eval_forward_fn(
                batch: RecsysFeaturesBatch,
                recsys_embeddings: RecsysEmbeddings,
            ):
                assert isinstance(self.model_config, RecsysGenRecsModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                input_embeddings, padding_mask, cand_offset, target_embeddings, mm_valid_mask = (
                    model.build_gen_recs_inputs(batch, recsys_embeddings)
                )
                predicted_embeddings = model.forward_gen_recs(
                    input_embeddings, padding_mask, is_training=False
                )
                C = target_embeddings.shape[1]
                pred_start = cand_offset - 1
                preds = predicted_embeddings[:, pred_start : pred_start + C, :]
                return preds, target_embeddings, mm_valid_mask

            self.gen_recs_eval_forward_fn = gen_recs_eval_forward_fn

            _gen_recs_cfg = self.model_config

            @hk.transform
            def gen_recs_project_fn(
                mm_embeddings: jax.Array,
                post_hash_embeddings: jax.Array | None = None,
                author_hash_embeddings: jax.Array | None = None,
            ) -> jax.Array:
                model = _gen_recs_cfg.make(
                    sharding_context=make_legacy_sharding_context(self.mesh),
                )
                if _gen_recs_cfg.candidate_embedding_mode == "post_combined":
                    assert post_hash_embeddings is not None and author_hash_embeddings is not None

                    cfg = _gen_recs_cfg
                    B, L, num_item_hashes, D = post_hash_embeddings.shape
                    _, _, num_author_hashes, _ = author_hash_embeddings.shape
                    post_embs_flat = post_hash_embeddings.reshape(B, L * num_item_hashes, D)
                    auth_embs_flat = author_hash_embeddings.reshape(B, L * num_author_hashes, D)

                    ps_ids = jnp.full((B, L), 10, dtype=jnp.int32)
                    embed_init = hk.initializers.VarianceScaling(
                        cfg.embed_init_scale, mode="fan_out"
                    )
                    ps_embedding_table = get_parameter(
                        "product_surface_embedding_table",
                        shape=[cfg.product_surface_vocab_size, cfg.emb_table_width],
                        init=embed_init,
                        dtype=jnp.float32,
                        pspec=P(),
                    )
                    ps_one_hot = jax.nn.one_hot(ps_ids, cfg.product_surface_vocab_size)
                    ps_embs = jnp.dot(ps_one_hot, ps_embedding_table).astype(jnp.bfloat16)

                    dummy_post_hashes = jnp.zeros((B, L, num_item_hashes), dtype=jnp.int32)
                    dummy_auth_hashes = jnp.zeros((B, L, num_author_hashes), dtype=jnp.int32)
                    embs, _ = block_candidate_reduce(
                        dummy_post_hashes,
                        dummy_auth_hashes,
                        post_embs_flat,
                        auth_embs_flat,
                        ps_embs,
                        cfg.hash_table.hash_keys,
                        cfg.model_config.scale_config.emb_lr_multiplier,
                        cfg.embed_init_scale,
                        cfg.use_product_surface,
                        multimodal_embeddings=mm_embeddings,
                    )
                    return embs
                return model._project_mm_to_candidate_space(mm_embeddings)

            self.gen_recs_project_fn = gen_recs_project_fn

    def _family_post_sharding_hook(self) -> None:
        if isinstance(self.model_config, RecsysGenRecsModelConfig):
            if self.model_config.candidate_embedding_mode == "post_combined":
                self.gen_recs_project_jit = JittedOrCompiled(
                    jax.jit(
                        self.gen_recs_project_fn.apply,
                        in_shardings=(
                            self.state_sharding.params,
                            None,
                            self.data_sharding,
                            self.data_sharding,
                            self.data_sharding,
                        ),
                        out_shardings=self.data_sharding,
                    )
                )
            else:
                self.gen_recs_project_jit = JittedOrCompiled(
                    jax.jit(
                        self.gen_recs_project_fn.apply,
                        in_shardings=(self.state_sharding.params, None, self.data_sharding),
                        out_shardings=self.data_sharding,
                    )
                )

    def _maybe_inject_global_neg_embeddings(self, batch):
        if not isinstance(self.model_config, RecsysGenRecsModelConfig):
            return batch
        cfg = self.model_config
        if cfg.num_global_negatives <= 0:
            return batch

        cand_emb = batch["candidate_seq"].get("embedding")
        cand_pids = batch["candidate_seq"].get("post_ids")
        if cand_emb is None or cand_pids is None:
            return batch

        raw_mm_embs = getattr(self, "_gen_recs_source_mm_embeddings", None)
        pid_to_idx = getattr(self, "_gen_recs_source_pid_to_idx", None)
        if raw_mm_embs is None or pid_to_idx is None:
            return batch

        C = cfg.candidate_seq_len
        B = cand_emb.shape[0]
        for b in range(B):
            for j in range(C, cand_emb.shape[1]):
                pid = int(cand_pids[b, j])
                idx = pid_to_idx.get(pid)
                if idx is not None:
                    cand_emb[b, j, :] = raw_mm_embs[idx]

        return batch

    def eval(self, soft_step: int):
        if isinstance(self.model_config, RecsysGenRecsModelConfig):
            return self.eval_gen_recs(soft_step)
        return super().eval(soft_step)

    def maybe_build_gen_recs_post_embeddings(self):
        if not isinstance(self.model_config, RecsysGenRecsModelConfig):
            return

        from dataclasses import replace

        import numpy as np

        from xrex.data.recsys.gen_recs_semantics import load_mm_embedding_table
        from xrex.data.retrieval_dataset import RetrievalDataset

        cfg = self.model_config
        parquet_path = cfg.eval_post_emb_path
        if parquet_path is None or not os.path.exists(parquet_path):
            rank_logger.warning(f"Post embedding parquet not found: {parquet_path}, skipping")
            return

        max_posts = cfg.max_posts

        resolved_path = os.path.realpath(parquet_path)
        last_resolved = getattr(self, "_gen_recs_last_resolved_parquet", None)
        if last_resolved == resolved_path and getattr(self, "_gen_recs_post_emb_built", False):
            return
        self._gen_recs_last_resolved_parquet = resolved_path

        t0 = time.time()
        if jax.process_index() == 0:
            post_ids, author_ids, embeddings = load_mm_embedding_table(
                parquet_path,
                embedding_type=getattr(self.dataset, "multimodal_embedding_type", "v3"),
            )
            M = min(len(post_ids), max_posts)
            post_ids = post_ids[:M]
            author_ids = author_ids[:M]
            embeddings = embeddings[:M].astype(np.float32)
            rank_logger.info(
                f"Rank-0 loaded {M} video posts in {time.time() - t0:.1f}s, broadcasting..."
            )
        else:
            M = 0
            post_ids = np.empty(0, dtype=np.int64)
            author_ids = np.empty(0, dtype=np.int64)
            embeddings = np.empty((0, cfg.mm_target_dim), dtype=np.float32)

        M = int(
            np.array(
                multihost_utils.broadcast_one_to_all(
                    np.array(M, dtype=np.int32),
                    is_source=jax.process_index() == 0,
                )
            )
        )

        if jax.process_index() != 0:
            embeddings = np.zeros((M, cfg.mm_target_dim), dtype=np.float32)
        embeddings = np.array(
            multihost_utils.broadcast_one_to_all(
                embeddings,
                is_source=jax.process_index() == 0,
            ),
            dtype=np.float32,
        )

        if jax.process_index() != 0:
            post_ids = np.zeros(M, dtype=np.int64)
            author_ids = np.zeros(M, dtype=np.int64)
        post_ids_i32 = post_ids.view(np.int32)
        author_ids_i32 = author_ids.view(np.int32)
        post_ids_i32, author_ids_i32 = multihost_utils.broadcast_one_to_all(
            (post_ids_i32, author_ids_i32),
            is_source=jax.process_index() == 0,
        )
        post_ids = np.asarray(post_ids_i32, dtype=np.int32).view(np.int64)
        author_ids = np.asarray(author_ids_i32, dtype=np.int32).view(np.int64)

        rank_logger.info(f"Post embeddings ready: {M} posts in {time.time() - t0:.1f}s")

        self._gen_recs_source_post_ids = post_ids.copy()
        self._gen_recs_source_author_ids = author_ids.copy()
        self._gen_recs_source_mm_embeddings = embeddings.copy()
        self._gen_recs_source_pid_to_idx = {int(pid): idx for idx, pid in enumerate(post_ids)}

        t1 = time.time()

        padded_mm = np.zeros((max_posts, cfg.mm_target_dim), dtype=np.float32)
        padded_mm[:M] = embeddings
        pid_padded = np.zeros(max_posts, dtype=np.int64)
        pid_padded[:M] = post_ids
        aid_padded = np.zeros(max_posts, dtype=np.int64)
        aid_padded[:M] = author_ids

        post_hashes = cfg.hash_table.get_item_hash(pid_padded)
        author_hashes = cfg.hash_table.get_author_hash(aid_padded)
        combined_hashes_np = np.concatenate([post_hashes, author_hashes], axis=1)

        emb_sharding = jax.sharding.NamedSharding(self.mesh, self.data_sharding.spec)
        mm_sharded = jax.make_array_from_callback(
            (max_posts, cfg.mm_target_dim),
            emb_sharding,
            lambda idx: jnp.array(padded_mm[idx], dtype=jnp.bfloat16),
        )
        hashes_sharded = jax.make_array_from_callback(
            (max_posts, combined_hashes_np.shape[1]),
            emb_sharding,
            lambda idx: jnp.array(combined_hashes_np[idx], dtype=jnp.int32),
        )

        num_item_hashes = cfg.hash_table.hash_keys.num_item_hashes
        num_author_hashes = cfg.hash_table.hash_keys.num_author_hashes
        d = cfg.emb_table_width

        if cfg.candidate_embedding_mode == "post_combined":
            hash_embs = self._lookup(self.state.emb_table, hashes_sharded)
            post_hash_embs = hash_embs.x[:, :num_item_hashes, :].reshape(
                max_posts, 1, num_item_hashes, d
            )
            auth_hash_embs = hash_embs.x[:, num_item_hashes:, :].reshape(
                max_posts, 1, num_author_hashes, d
            )
            mm_input = mm_sharded.reshape(max_posts, 1, cfg.mm_target_dim)

            rng = jax.random.PRNGKey(0)
            projected = self.gen_recs_project_jit(
                self.state.params, rng, mm_input, post_hash_embs, auth_hash_embs
            )
            projected = projected.reshape(max_posts, -1)
        else:
            rng = jax.random.PRNGKey(0)
            projected = self.gen_recs_project_jit(self.state.params, rng, mm_sharded)

        projected = l2_normalize(projected)
        emb_dim = projected.shape[-1]

        rank_logger.info(
            "Updated gen-recs candidate table: (%s, %s) (%s posts from %s) in %.1fs",
            M,
            emb_dim,
            M,
            resolved_path,
            time.time() - t1,
        )

        eval_table_embs = projected[:M]
        eval_table_replicated = jax.device_put(
            eval_table_embs,
            jax.sharding.NamedSharding(self.mesh, P(None, None)),
        )
        self._gen_recs_eval_table_embs = eval_table_replicated
        self._gen_recs_eval_table_post_ids = post_ids.copy()
        self._gen_recs_eval_table_pid_set = set(post_ids.tolist())
        self._gen_recs_eval_table_size = M

        global_emb = projected.astype(jnp.bfloat16)

        global_post_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(pid_padded), self.mesh, P(None)
        )
        global_author_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(aid_padded), self.mesh, P(None)
        )

        original = self.state.post_embeddings
        self.state = self.state._replace(
            post_embeddings=PostEmbeddings(
                post_ids=global_post_ids,
                author_ids=global_author_ids,
                embeddings=replace(
                    original.embeddings, x=global_emb, pspec=self.data_sharding.spec
                ),
                dataset_types=multihost_utils.host_local_array_to_global_array(
                    jnp.full((max_posts, 1), RetrievalDataset.PAD.value, dtype=jnp.int32),
                    self.mesh,
                    P(None),
                ),
            )
        )
        self._gen_recs_post_emb_built = True
        rank_logger.info(f"Gen-recs post embeddings built: [{M}/{max_posts}, {emb_dim}]")

    def eval_gen_recs(self, soft_step: int) -> dict[str, float]:
        assert isinstance(self.model_config, RecsysGenRecsModelConfig)
        assert isinstance(self.state, RecsysTrainingState)

        if not getattr(self, "_gen_recs_post_emb_built", False):
            self.maybe_build_gen_recs_post_embeddings()
        if not getattr(self, "_gen_recs_post_emb_built", False):
            rank_logger.warning("Gen-recs post embeddings not available, skipping eval")
            return {}

        eval_table_embs = getattr(self, "_gen_recs_eval_table_embs", None)
        eval_table_post_ids = getattr(self, "_gen_recs_eval_table_post_ids", None)
        eval_table_pid_set = getattr(self, "_gen_recs_eval_table_pid_set", None)
        if eval_table_embs is None or eval_table_post_ids is None:
            rank_logger.warning("No gen-recs eval table available, skipping eval")
            return {}

        rank_logger.info("Running gen-recs global recall eval at step %s", soft_step)
        t0 = time.time()

        M = getattr(self, "_gen_recs_eval_table_size", len(eval_table_post_ids))
        table_jax = eval_table_embs.astype(jnp.bfloat16)
        table_post_ids_np = np.asarray(eval_table_post_ids, dtype=np.int64)
        Ks = [1, 10, 100, 1000]
        k_top = 1000
        num_track_pos = 10

        rank_logger.info("  Eval table: %s valid video posts (GPU)", M)

        if not hasattr(self, "_gen_recs_eval_jit"):
            self._gen_recs_eval_jit = jax.jit(self.gen_recs_eval_forward_fn.apply)

        _rng, eval_rng = jax.random.split(self.state.rng)

        agg: dict[str, float | int] = {}
        num_eval_batches = int(os.environ.get("GEN_RECS_EVAL_BATCHES", "20"))
        eval_iter = self.dataset_thread_iterator(self._eval_dataset, prepare_data=False)
        data_pspec = P(("stage", "expert", "replica", "data"), ("seq", "model"))

        for batch_idx in range(num_eval_batches):
            try:
                cpu_batch = next(eval_iter)
            except StopIteration:
                rank_logger.info(f"  Dataset exhausted after {batch_idx} eval batches")
                break

            cand_emb = cpu_batch.get("candidate_seq", {}).get("embedding")
            if cand_emb is None:
                continue

            jax_batch = multihost_utils.host_local_array_to_global_array(
                copy.deepcopy(cpu_batch), self.mesh, data_pspec
            )
            recsys_emb_param = self.get_recsys_embeddings(jax_batch, self.state.emb_table)
            recsys_emb = get_recsys_embed_param_to_jax_array(recsys_emb_param)
            preds_jax, _targets_jax, mm_valid_jax = self._gen_recs_eval_jit(
                self.state.params, eval_rng, jax_batch, recsys_emb
            )

            preds = np.array(
                jax.device_get(
                    multihost_utils.global_array_to_host_local_array(
                        preds_jax, self.mesh, data_pspec
                    )
                ),
                dtype=np.float32,
            )
            mm_valid = np.array(
                jax.device_get(
                    multihost_utils.global_array_to_host_local_array(
                        mm_valid_jax, self.mesh, data_pspec
                    )
                ),
                dtype=np.float32,
            )

            _B, C, _D = preds.shape
            cand_post_ids = cpu_batch.get("candidate_seq", {}).get("post_ids")
            assert cand_post_ids is not None, (
                "Global recall eval requires include_candidate_post_ids=True in dataset config"
            )
            cand_post_ids = cand_post_ids[:, :C]

            valid_mask = mm_valid > 0
            assert eval_table_pid_set is not None
            in_table = np.vectorize(lambda pid: int(pid) in eval_table_pid_set)(cand_post_ids)
            valid_mask = valid_mask & in_table

            npc_raw = cpu_batch.get("num_positive_candidates")
            if npc_raw is not None:
                npc = np.asarray(npc_raw)[:, 0]
                col_indices = np.arange(C)[None, :]
                valid_mask = valid_mask & (col_indices < npc[:, None])

            gt_total = int(np.sum(mm_valid > 0))
            gt_found = int(np.sum(valid_mask))
            agg["gt_total"] = agg.get("gt_total", 0) + gt_total
            agg["gt_found"] = agg.get("gt_found", 0) + gt_found

            v_preds = preds[valid_mask]
            N = len(v_preds)
            agg["num_eval"] = agg.get("num_eval", 0) + N
            if N == 0:
                continue

            pn = np.linalg.norm(v_preds, axis=-1, keepdims=True)
            v_preds_n = np.where(pn > 1e-8, v_preds / pn, 0.0)

            v_preds_jax = jnp.array(v_preds_n, dtype=jnp.bfloat16)
            scores_jax = jnp.dot(v_preds_jax, table_jax.T).astype(jnp.float32)
            _, top_k_idx = jax.lax.top_k(scores_jax, k_top)
            top_k_idx = np.array(jax.device_get(top_k_idx))

            v_target_pids = cand_post_ids[valid_mask]
            top_k_pids = table_post_ids_np[top_k_idx]

            for k in Ks:
                hits = np.sum(np.any(top_k_pids[:, :k] == v_target_pids[:, None], axis=1))
                agg[f"hits@{k}"] = agg.get(f"hits@{k}", 0) + int(hits)

            v_positions = np.where(valid_mask)[1]
            for c in range(num_track_pos):
                pos_mask = v_positions == c
                agg[f"pos_{c}_total"] = agg.get(f"pos_{c}_total", 0) + int(np.sum(pos_mask))
                for k in Ks:
                    hits = np.any(top_k_pids[:, :k] == v_target_pids[:, None], axis=1)
                    agg[f"pos_{c}_hits@{k}"] = agg.get(f"pos_{c}_hits@{k}", 0) + int(
                        np.sum(hits & pos_mask)
                    )

            rank_logger.info(
                f"  Batch {batch_idx}: {N} evaluated, "
                f"running recall@10={agg.get('hits@10', 0) / max(agg.get('num_eval', 1), 1):.4f}"
            )

        for key in list(agg.keys()):
            dtype = np.int64 if isinstance(agg[key], int) else np.float64
            agg[key] = float(
                multihost_utils.process_allgather(
                    np.array(agg[key], dtype=dtype), tiled=False
                ).sum()
            )

        metrics = build_recall_metrics(agg, ks=Ks, num_track_pos=num_track_pos)
        metrics["gen-recs-eval-table-valid-posts"] = float(M)
        metrics["gen-recs-eval-time-sec"] = time.time() - t0

        rank_logger.info(f"Gen-recs eval done in {time.time() - t0:.1f}s")
        for k, v in sorted(metrics.items()):
            rank_logger.info(f"  {k}: {v:.6f}")
        return metrics
