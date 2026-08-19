# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import optax
from xai_configlib import Config, configclass

from xrex.optimizers.recsys.protocol import RecsysEmbeddingOptimizer
from xrex.optimizers.recsys.rowwise_adagrad import (
    RecsysRowwiseAdagradConfig,
    RecsysRowwiseAdagradOptimizer,
)


@configclass
class RecsysEmbeddingOptimConfig(Config):
    rowwise_adagrad: RecsysRowwiseAdagradConfig = RecsysRowwiseAdagradConfig()

    def _active(self) -> str:
        return "rowwise_adagrad"

    def make(self) -> optax.GradientTransformation | None:
        return None

    def make_optimizer(
        self, shared_optim: optax.GradientTransformation
    ) -> RecsysEmbeddingOptimizer:
        del shared_optim
        cfg = self.rowwise_adagrad
        return RecsysRowwiseAdagradOptimizer(
            learning_rate=cfg.learning_rate,
            eps=cfg.eps,
            initial_accumulator_value=cfg.initial_accumulator_value,
            half_life_steps=cfg.half_life_steps,
            lazy_decay=cfg.lazy_decay,
            weight_decay=cfg.weight_decay,
        )
