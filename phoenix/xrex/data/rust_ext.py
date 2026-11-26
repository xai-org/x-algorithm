# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import importlib
from types import ModuleType

_BUILD_HINT = {
    "xai_recsys_kafka_reader": "reading the training topics straight off Kafka",
    "xai_recsys_parquet_reader": "reading sharded parquet batch files",
}

_PURE_PYTHON_ALTERNATIVE = {
    "xai_recsys_kafka_reader": "xrex.data.streaming.kafkaloader.PhoenixKafkaDataset",
    "xai_recsys_parquet_reader": "xrex.data.parquet_recsys.PhoenixDataset",
}


def load(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        alt = _PURE_PYTHON_ALTERNATIVE[name]
        raise ImportError(
            f"{name} is not installed. It is the native extension behind "
            f"{_BUILD_HINT[name]}, and it is not part of this package: build it "
            f"from the corresponding Rust crate and put it on the import path, "
            f"or use the pure-Python feed {alt}, which carries the same "
            f"semantics without a build step."
        ) from exc
