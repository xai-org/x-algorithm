# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence

import pyarrow as pa

rank_logger = logging.getLogger("rank")

_PARTITION_SEG_RE = re.compile(r"partition=\d+/")


def sidecar_path_for(
    batch_file_path: str,
    sidecar_root: str,
    suffix: str,
    *,
    path_pattern: re.Pattern[str] = _PARTITION_SEG_RE,
) -> str:
    match = path_pattern.search(batch_file_path)
    if match is None:
        raise ValueError(f"not a partition batch file path: {batch_file_path}")
    rel = batch_file_path[match.start() :]
    if not rel.endswith(".parquet"):
        raise ValueError(f"not a parquet path: {batch_file_path}")
    return os.path.join(sidecar_root, rel[: -len(".parquet")] + suffix)


def set_column(batch: pa.RecordBatch, name: str, col: pa.Array) -> pa.RecordBatch:
    if len(col) != batch.num_rows:
        raise ValueError(f"{name}: {len(col)} rows != batch rows {batch.num_rows}")
    if name in batch.schema.names:
        batch = batch.drop_columns([name])
    return batch.append_column(pa.field(name, col.type, nullable=True), col)


class SourceCounter:
    def __init__(self, columns: str | Sequence[str], log_label: str, every: int = 1000):
        self._single_column = isinstance(columns, str)
        names = (columns,) if isinstance(columns, str) else columns
        sources = (
            ("sidecar", "native", "none") if self._single_column else ("native", "sidecar", "none")
        )
        self._log_label = log_label
        self._every = every
        self._files = 0
        self._counts = {name: dict.fromkeys(sources, 0) for name in names}
        self._seen: set[tuple[str, str]] = set()

    def note(self, fname: str, sources: Mapping[str, str]) -> None:
        self._files += 1
        for column, source in sources.items():
            self._counts[column][source] += 1
            if (column, source) not in self._seen:
                self._seen.add((column, source))
                detail = f"source={source}" if self._single_column else f"{column} from {source}"
                rank_logger.info(f"{self._log_label}: first file with {detail}: {fname}")
        if self._files % self._every == 0:
            counts = next(iter(self._counts.values())) if self._single_column else self._counts
            rank_logger.info(f"{self._log_label}: sources after {self._files} files: {counts}")

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        return {column: dict(counts) for column, counts in self._counts.items()}
