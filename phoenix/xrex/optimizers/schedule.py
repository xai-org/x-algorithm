# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Callable
from typing import Any

from xai_configlib import Config, configclass


@configclass
class BaseSchedule(Config):
    def make(self):
        raise NotImplementedError


@configclass
class BaseSampleSchedule(Config):
    def make(self) -> Callable[..., float]:
        raise NotImplementedError


@configclass
class ConstantSampleSchedule(BaseSampleSchedule):
    learning_rate: float = 1.0

    def make(self) -> Callable[..., float]:
        def schedule(*_args: Any, **_kwargs: Any) -> float:
            return self.learning_rate

        return schedule
