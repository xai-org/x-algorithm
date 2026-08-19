# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from xai_configlib import Config, configclass


@configclass
class ForwardMetrics(Config):
    @abstractmethod
    def run_batch(self, logits, inputs, targets, mask=None):
        raise NotImplementedError

    @abstractmethod
    def get(self) -> float | dict[str | None, float] | None:
        raise NotImplementedError

    def after_run(
        self,
        eval_name: str,
        elapsed_samples: int,
        metric_folder: str | None = None,
        is_data_shard_output_rank: bool = True,
    ) -> None:
        pass


@dataclass
class WeightedAverageAccumulator:
    sum: float = 0.0
    weight: float = 0.0

    def add(self, metrics, weight):
        self.sum += metrics
        self.weight += weight

    def get(self):
        if self.weight == 0.0:
            return jnp.array(0.0)
        else:
            return self.sum / self.weight


@dataclass
class SimpleSumAccumulator:
    sum: float = 0.0

    def add(self, value):
        self.sum += value

    def get(self):
        return jnp.array(self.sum)
