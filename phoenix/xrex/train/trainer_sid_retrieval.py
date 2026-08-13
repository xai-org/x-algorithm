# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import copy
import logging
import time
import typing
from dataclasses import field

import haiku as hk
import jax
from xai_configlib import configclass

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.recsys_embedding import RecsysEmbeddingsParameter
from xrex.models.recsys_sid_retrieval_model import RecsysSIDRetrievalConfig
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.train.misc import RecsysTrainingState
from xrex.train.trainer_recsys import RecsysTrainer

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@configclass
class SidRetrievalTrainer(RecsysTrainer):
    _make_sid_generate_fn: typing.Any = field(default=None, init=False, repr=False)
    _sid_beam_generate_fns: dict[int, typing.Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _sid_beam_generate_jits: dict[int, typing.Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _sid_beam_resolver: object | None = field(default=None, init=False, repr=False)

    def create_executable(self) -> None:
        super().create_executable()
        self._family_create_executable_hook()

    def _family_create_executable_hook(self) -> None:
        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        _sid_eval_config = copy.deepcopy(self.model_config)
        object.__setattr__(
            _sid_eval_config.model_config.attn_config,
            "attn_impl",
            "jax_attn",
        )

        def _make_sid_generate_fn(beam_width: int):
            @hk.transform
            def fn(
                batch: RecsysFeaturesBatch,
                recsys_embeddings: RecsysEmbeddingsParameter,
            ):
                model = _sid_eval_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh),
                )
                return model.generate(batch, recsys_embeddings, beam_width=beam_width)

            return fn

        self._make_sid_generate_fn = _make_sid_generate_fn
        self._sid_beam_generate_fns = {}
        self._sid_beam_generate_jits = {}

    def eval(self, soft_step: int):
        if isinstance(self.model_config, RecsysSIDRetrievalConfig):
            return self.eval_sid_retrieval(soft_step)
        return super().eval(soft_step)

    def _reload_sid_beam_resolver(self) -> tuple[object, float]:
        from xrex.data.retrieval_dataset import RetrievalDataset
        from xrex.inference.sid_post_index import SidBeamResolver

        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        t0 = time.time()
        rank_logger.info(
            "Reloading SID beam resolver corpus (HOME / 1fav_1day) for post_recall@K..."
        )
        resolver = SidBeamResolver.from_datasets(
            [RetrievalDataset.HOME],
            sid_num_levels=self.model_config.sid_num_levels,
            codebook_size=self.model_config.sid_codebook_size,
        )
        load_s = time.time() - t0
        self._sid_beam_resolver = resolver
        rank_logger.info(
            "SID beam resolver reloaded: %d posts, %d unique SIDs (%.2fs)",
            resolver.num_posts,
            len(resolver.sid_to_post_idx),
            load_s,
        )
        return resolver, load_s

    def eval_sid_retrieval(self, soft_step: int) -> dict[str, float]:
        from xrex.eval.recsys_eval import (
            finalize_sid_eval_metrics,
            log_sid_eval_summary,
            make_sid_retrieval_eval_task,
            sid_eval_batch_geometry,
        )

        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        assert isinstance(self.state, RecsysTrainingState)

        rank_logger.info("Running SID retrieval eval at step %s", soft_step)
        t0 = time.time()
        rng = jax.random.PRNGKey(soft_step)
        beam_width = max(1, int(self.model_config.eval_beam_width))
        eval_bs, eval_global_b, host_local_b = sid_eval_batch_geometry(
            int(self.model_config.eval_bs_per_device), self.num_devices
        )
        if eval_bs > self.bs_per_device:
            raise ValueError(
                f"eval_bs_per_device={eval_bs} > train bs_per_device={self.bs_per_device}; "
                "eval B must fit in the train batch (slice prefix) or lower train B"
            )

        if beam_width not in self._sid_beam_generate_jits:
            fn = self._make_sid_generate_fn(beam_width)
            self._sid_beam_generate_fns[beam_width] = fn
            self._sid_beam_generate_jits[beam_width] = jax.jit(fn.apply)
        beam_jit = self._sid_beam_generate_jits[beam_width]

        saved_batch_size = int(self.batch_size)

        def beam_forward_fn(jax_batch):
            emb = self.get_recsys_embeddings(jax_batch, self.state.emb_table)
            return beam_jit(self.state.params, rng, jax_batch, emb)

        try:
            object.__setattr__(self, "batch_size", eval_global_b)
            sid_resolver, corpus_load_s = self._reload_sid_beam_resolver()
            eval_task = make_sid_retrieval_eval_task(
                max_steps=int(self.model_config.eval_num_batches),
                num_levels=self.model_config.sid_num_levels,
                beam_width=beam_width,
                sid_resolver=sid_resolver,
            )
            t_beam = time.time()
            eval_iter = self.dataset_thread_iterator(
                self._eval_dataset or self.train_dataset, prepare_data=False
            )
            rank_logger.info(
                "SID eval: eval_bs_per_device=%d global_B=%d host_local_B=%d "
                "steps=%d train_bs_per_device=%d",
                eval_bs,
                eval_global_b,
                host_local_b,
                int(self.model_config.eval_num_batches),
                self.bs_per_device,
            )
            eval_task.run(
                train_dataset=eval_iter,
                forward_fn=lambda _b: None,
                loss_fn=lambda _b: (0.0, {}),
                mesh=self.mesh,
                name="sid_retrieval",
                beam_forward_fn=beam_forward_fn,
                sid_resolver=sid_resolver,
                host_local_batch_size=host_local_b,
            )
            beam_wall_s = time.time() - t_beam
            metrics = finalize_sid_eval_metrics(
                eval_task,
                corpus_load_s=corpus_load_s,
                beam_wall_s=beam_wall_s,
                total_wall_s=time.time() - t0,
                sid_resolver=sid_resolver,
                eval_bs_per_device=eval_bs,
            )
            log_sid_eval_summary(metrics, label="SID retrieval eval", soft_step=soft_step)
            return metrics
        finally:
            object.__setattr__(self, "batch_size", saved_batch_size)
