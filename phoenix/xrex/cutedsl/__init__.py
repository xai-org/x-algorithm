# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import importlib.metadata


def cutlass_dsl_version() -> str:
    return importlib.metadata.version("nvidia-cutlass-dsl")


def cutlass_dsl_is_current() -> bool:
    return tuple(int(x) for x in cutlass_dsl_version().split(".")[:2]) >= (4, 7)
