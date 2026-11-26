# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

rank_logger = logging.getLogger("rank")

_STREAM_COLUMN_PREFIX = "kafka_"
_logged: set[tuple[str, str]] = set()


def dump_layout(record_batch: pa.RecordBatch) -> pa.RecordBatch:
    names = record_batch.schema.names
    record_columns = [
        column
        for name, column in zip(names, record_batch.columns)
        if not name.startswith(_STREAM_COLUMN_PREFIX)
    ]
    if record_columns and all(column.null_count for column in record_columns):
        undecoded = pc.is_null(record_columns[0])
        for column in record_columns[1:]:
            undecoded = pc.and_(undecoded, pc.is_null(column))
        if pc.any(undecoded).as_py():
            record_batch = record_batch.filter(pc.invert(undecoded))

    kept: dict[str, pa.Array] = {}
    for name, column in zip(names, record_batch.columns):
        if column.null_count == len(column) and not name.startswith(_STREAM_COLUMN_PREFIX):
            _log_once(name, "has no value in a record batch: read as a column the records lack")
            continue
        fixed = _fixed_size(column)
        if fixed is None:
            _log_once(name, "has lists of several lengths in a record batch: left as a list")
        kept[name] = column if fixed is None else fixed
    return pa.RecordBatch.from_arrays(list(kept.values()), names=list(kept))


def ragged_lengths(
    record_batch: pa.RecordBatch, columns: list[str]
) -> dict[str, list[tuple[int, int]]]:
    ragged: dict[str, list[tuple[int, int]]] = {}
    for name in columns:
        if name not in record_batch.schema.names:
            continue
        array = record_batch.column(name)
        if _fixed_size_at_every_depth(array.type):
            continue
        lengths: list[tuple[int, int]] = []
        while True:
            if pa.types.is_fixed_size_list(array.type):
                lengths.append((array.type.list_size, array.type.list_size))
            elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
                found = pc.min_max(pc.list_value_length(array)).as_py()
                lengths.append((found["min"] or 0, found["max"] or 0))
            else:
                break
            array = array.flatten()
        if any(low != high for low, high in lengths):
            ragged[name] = lengths
    return ragged


def _fixed_size_at_every_depth(data_type: pa.DataType) -> bool:
    while pa.types.is_fixed_size_list(data_type):
        data_type = data_type.value_type
    return not (pa.types.is_list(data_type) or pa.types.is_large_list(data_type))


def _log_once(column: str, what: str) -> None:
    if (column, what) not in _logged:
        _logged.add((column, what))
        rank_logger.warning("Iceberg column %s %s", column, what)


def _fixed_size(array: pa.Array) -> pa.Array | None:
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return array
    leaf = array.type
    while pa.types.is_list(leaf) or pa.types.is_large_list(leaf):
        leaf = leaf.value_type
    if pa.types.is_nested(leaf):
        return array
    lengths = pc.min_max(pc.list_value_length(array)).as_py()
    size = lengths["min"]
    if size is None or size != lengths["max"]:
        return None
    if size == 0:
        return array
    offsets = array.offsets
    if array.null_count == 0:
        values, null = array.values.slice(offsets[0].as_py(), len(array) * size), None
    else:
        null = pc.is_null(array).to_numpy(zero_copy_only=False)
        starts = offsets.slice(0, len(array)).to_numpy().astype(np.int64)
        slots = (starts[:, None] + np.arange(size)).ravel()
        values = array.values.take(pa.array(slots, mask=np.repeat(null, size)))
    values = _fixed_size(values)
    if values is None:
        return None
    return pa.FixedSizeListArray.from_arrays(
        values, size, mask=None if null is None else pa.array(null)
    )
