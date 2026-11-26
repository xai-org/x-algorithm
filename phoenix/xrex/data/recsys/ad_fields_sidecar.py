# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import os
import re
from collections.abc import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from xrex.data.recsys import sidecar

COLUMNS = ("campaignIdSeq", "lineItemIdSeq", "unifiedCardTypeSeq", "mediaContentTypeSeq")
SEQ_REFERENCE_COLUMN = "impressedTimeMsSeq"

_PARTITION_SEG_RE = re.compile(r"partition=\d+/(?:\d+/)?batch_\d+\.parquet$")


def sidecar_path_for(batch_file_path: str, sidecar_root: str) -> str:
    return sidecar.sidecar_path_for(
        batch_file_path, sidecar_root, ".ad_fields.parquet", path_pattern=_PARTITION_SEG_RE
    )


def load_sidecar(path: str, expected_rows: int) -> dict[str, pa.Array]:
    names = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in COLUMNS if c in names])
    if table.num_rows != expected_rows:
        raise ValueError(
            f"ad-field sidecar {path} has {table.num_rows} rows, batch file has {expected_rows}"
        )
    return {name: table.column(name).combine_chunks() for name in table.column_names}


def seq_len_of(col: pa.Array) -> int:
    if pa.types.is_fixed_size_list(col.type):
        return col.type.list_size
    lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    if len(lengths) == 0:
        return 0
    if col.null_count or (lengths != lengths[0]).any():
        raise ValueError("ad-field columns need one list length on every row")
    return int(lengths[0])


def canonical_column(col: pa.Array, seq_len: int) -> pa.FixedSizeListArray:
    if (n := seq_len_of(col)) != seq_len:
        raise ValueError(f"ad-field column has lists of {n}, want {seq_len}")
    if pa.types.is_fixed_size_list(col.type):
        values = col.values.slice(col.offset * seq_len, len(col) * seq_len)
    else:
        values = col.flatten()
    values = values.fill_null(0).cast(pa.int64())
    return pa.FixedSizeListArray.from_arrays(values, seq_len)


def zeros_column(num_rows: int, seq_len: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(num_rows * seq_len, dtype=np.int64)), seq_len
    )


class FileJoin:
    def __init__(self, fname: str, sidecar_root: str, file_columns: list[str], num_rows: int):
        native = set(file_columns)
        need_sidecar = [c for c in COLUMNS if c not in native]
        sidecar: Mapping[str, pa.Array] = {}
        if need_sidecar:
            path = sidecar_path_for(fname, sidecar_root)
            if os.path.exists(path):
                sidecar = load_sidecar(path, num_rows)
        self.sidecar: dict[str, pa.Array] = {c: sidecar[c] for c in need_sidecar if c in sidecar}
        self.sources: dict[str, str] = {
            c: "native" if c in native else "sidecar" if c in self.sidecar else "none"
            for c in COLUMNS
        }

    def apply(self, batch: pa.RecordBatch, row_pos: int) -> pa.RecordBatch:
        seq_len = seq_len_of(batch.column(SEQ_REFERENCE_COLUMN))
        for name in COLUMNS:
            source = self.sources[name]
            if source == "native":
                col = canonical_column(batch.column(name), seq_len)
            elif source == "sidecar":
                window = self.sidecar[name].slice(row_pos, batch.num_rows)
                col = canonical_column(window, seq_len)
            else:
                col = zeros_column(batch.num_rows, seq_len)
            batch = sidecar.set_column(batch, name, col)
        return batch


class SourceCounter(sidecar.SourceCounter):
    def __init__(self, every: int = 1000):
        super().__init__(COLUMNS, "ad fields", every)
