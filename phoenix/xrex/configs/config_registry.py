# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Callable
from pathlib import Path
from typing import Any

RANKING_DATASET_FACTORIES: dict[str, Callable[..., Any]] = {}
RANKING_EXTRA_DATASET_TYPES: list[str] = []

RETRIEVAL_DATASET_FACTORIES: dict[str, Callable[..., Any]] = {}
RETRIEVAL_EXTRA_DATASET_TYPES: list[str] = []
GLOBAL_IDS_RESOLVERS: list[Callable[[Path | None], Path | None]] = []

RANKING_MODEL_CFG_BUILDERS: list[Callable[[Callable[..., Any]], dict[str, Any]]] = []
RETRIEVAL_MODEL_CFG_BUILDERS: list[Callable[[Callable[..., Any]], dict[str, Any]]] = []

RANKING_CONFIG_HOOKS: list[Callable[[dict[str, Any]], None]] = []
RETRIEVAL_CONFIG_HOOKS: list[Callable[[dict[str, Any]], None]] = []

RETRIEVAL_EVAL_BUILDERS: list[Callable[..., list[Any]]] = []
EVAL_WANDB_HOOK_FACTORIES: list[Callable[..., Any]] = []
