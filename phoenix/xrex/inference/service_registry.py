# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from collections.abc import Callable
from typing import Any

EXTRA_MODEL_CFGS: dict[str, Any] = {}

EXTRA_CONFIG_TYPES: list[type] = []

EXTRA_RUNNERS: dict[str, type] = {}

RETRIEVAL_DATASET_SERVICE_TYPES: list[str] = ["retrieval"]

SERVICE_ENV_HOOKS: list[Callable[[str], None]] = []

RUNNER_CLASS_HOOKS: list[Callable[[type, Any], type]] = []

CHECKPOINT_RESOLVERS: list[Callable[[Any, str | None], bool]] = []

STORAGE_OVERRIDE_HOOKS: list[Callable[[str, list[str]], None]] = []

CHECKPOINT_STORE_DETECTORS: list[Callable[[Any], Any]] = []

CHECKPOINT_STORAGE_LOADERS: list[Callable[[Any, Any, Any], Any | None]] = []
