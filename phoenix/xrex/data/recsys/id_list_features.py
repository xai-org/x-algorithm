# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xrex.data.recsys import sidecar

COLUMN = "userIdListFeatures"
HASHES_KEY = "user_id_list_hashes"
MASK_KEY = "user_id_list_mask"

CANONICAL_TYPE: pa.DataType = pa.list_(
    pa.struct([pa.field("name", pa.string()), pa.field("ids", pa.list_(pa.int64()))])
)


class IdListFamily(enum.Enum):
    TOP_POST = "top_post"
    TOP_AUTHOR = "top_author"
    TOP_CONVERSION_POST = "top_conversion_post"
    TOP_CONVERSION_AUTHOR = "top_conversion_author"
    SEQ_POST = "seq_post"
    SEQ_AUTHOR = "seq_author"
    TOP_ORGANIC_POST = "top_organic_post"
    SEQ_ORGANIC_POST = "seq_organic_post"

    @property
    def is_author(self) -> bool:
        return self in (
            IdListFamily.TOP_AUTHOR,
            IdListFamily.TOP_CONVERSION_AUTHOR,
            IdListFamily.SEQ_AUTHOR,
        )

    @property
    def is_organic_post(self) -> bool:
        return self in (IdListFamily.TOP_ORGANIC_POST, IdListFamily.SEQ_ORGANIC_POST)


def family_of(name: str) -> IdListFamily:
    if is_history_list(name):
        return parse_history_list(name).family
    if name.startswith("top_cp_"):
        return IdListFamily.TOP_CONVERSION_POST
    if name.startswith("top_cau_"):
        return IdListFamily.TOP_CONVERSION_AUTHOR
    if name.startswith("top_p_"):
        return IdListFamily.TOP_POST
    if name.startswith("top_au_"):
        return IdListFamily.TOP_AUTHOR
    if name.startswith("latest_au_"):
        return IdListFamily.SEQ_AUTHOR
    if name.startswith(("latest_p_", "longest_dwell_p_")):
        return IdListFamily.SEQ_POST
    raise ValueError(f"unknown id-list feature name: {name!r}")


_K_IN_NAME_RE = re.compile(
    r"^(?:hist_)?(?:top_(?:p|au|pau|cp|cau|ap|op)|latest_(?:p|au|pau|cau|ap|op|cp)|longest_dwell_p)_(\d+)_"
)


def default_k(name: str) -> int:
    m = _K_IN_NAME_RE.match(name)
    if m is None:
        raise ValueError(f"cannot infer k from id-list feature name: {name!r}")
    return int(m.group(1))


def feature_name_for(name: str, k: int) -> str:
    m = _K_IN_NAME_RE.match(name)
    if m is None:
        return name
    return name[: m.start(1)] + str(k) + name[m.end(1) :]


def identity_for(name: str) -> str:
    m = _K_IN_NAME_RE.match(name)
    if m is None:
        return name
    return name[: m.start(1)] + name[m.end(1) + 1 :]


def source_cap(name: str) -> int | None:
    if is_history_list(name):
        return None
    m = _K_IN_NAME_RE.match(name)
    return int(m.group(1)) if m else None


HIST_PREFIX = "hist_"

HISTORY_EVENTS: frozenset[str] = frozenset(
    {
        "FAV",
        "REPLY",
        "RETWEET",
        "QUOTE",
        "FOLLOW_AUTHOR",
        "BOOKMARK",
        "TAKE_SCREENSHOT",
        "CLICK",
        "CLICK_PROFILE",
        "OPEN_LINK",
        "PHOTO_EXPAND",
        "SHOW_MORE_EXPAND",
        "CLICK_HASHTAG",
        "CLICK_MENTION",
        "QUOTED_TWEET_CLICK",
        "TRANSLATE_CLICK",
        "CLICK_GROK_ANALYZE",
        "EXTERNAL_LINK_SESSION",
        "EXTERNAL_LINK_LONG_DWELLED",
        "VIDEO_QUALITY_VIEW",
        "RECAP_DWELLED",
        "SHARE",
        "RESHARE",
        "ENGAGE",
        "BRAND_ENGAGE",
        "CAROUSEL_SWIPE",
        "ADS_SESSION_CONVERSION",
        "ADS_SITE_VISIT_CONVERSION",
        "ADS_LONG_SITE_DWELL_CONVERSION",
        "ADS_LANDING_PAGE_VIEW_CONVERSION",
        "ADS_PURCHASE_CONVERSION",
        "ADS_CONTENT_VIEW_CONVERSION",
        "ADS_SIGN_UP_CONVERSION",
        "ADS_CUSTOM_CONVERSION",
        "ADS_ADD_TO_CART_CONVERSION",
        "CUSTOM_SIGNUP_ADDTOCART",
        "OTHER_CONVERSION",
        "ANY_CONVERSION",
    }
)

ENTITIES = ("au", "pau", "cau", "ap", "op", "cp")
_AUTHOR_ENTITIES = ("au", "pau", "cau")
_PROMOTED_ENTITIES = ("pau", "cau", "ap", "cp")
_ORGANIC_ENTITIES = ("op",)

_HIST_NAME_RE = re.compile(
    r"^hist_(?P<kind>top|latest)_(?P<entity>au|pau|cau|ap|op|cp)_(?P<k>\d+)_"
    r"(?:(?P<window>\d+)(?P<unit>d|hr|h)_)?(?P<event>[A-Z][A-Z0-9_]*)$"
)

_HIST_FAMILY = {
    ("top", "au"): IdListFamily.TOP_AUTHOR,
    ("top", "pau"): IdListFamily.TOP_AUTHOR,
    ("latest", "au"): IdListFamily.SEQ_AUTHOR,
    ("latest", "pau"): IdListFamily.SEQ_AUTHOR,
    ("top", "cau"): IdListFamily.TOP_CONVERSION_AUTHOR,
    ("latest", "cau"): IdListFamily.TOP_CONVERSION_AUTHOR,
    ("top", "ap"): IdListFamily.TOP_POST,
    ("latest", "ap"): IdListFamily.SEQ_POST,
    ("top", "cp"): IdListFamily.TOP_CONVERSION_POST,
    ("latest", "cp"): IdListFamily.TOP_CONVERSION_POST,
    ("top", "op"): IdListFamily.TOP_ORGANIC_POST,
    ("latest", "op"): IdListFamily.SEQ_ORGANIC_POST,
}


def is_history_list(name: str) -> bool:
    return name.startswith(HIST_PREFIX)


@dataclasses.dataclass(frozen=True)
class HistoryListSpec:
    name: str
    kind: str
    entity: str
    k: int
    window_days: float | None
    event: str

    @property
    def family(self) -> IdListFamily:
        return _HIST_FAMILY[(self.kind, self.entity)]

    @property
    def is_author(self) -> bool:
        return self.entity in _AUTHOR_ENTITIES

    @property
    def promoted_only(self) -> bool:
        return self.entity in _PROMOTED_ENTITIES

    @property
    def organic_only(self) -> bool:
        return self.entity in _ORGANIC_ENTITIES


def parse_history_list(name: str) -> HistoryListSpec:
    m = _HIST_NAME_RE.match(name)
    if m is None:
        raise ValueError(
            f"not a history id-list name: {name!r} (expected "
            "hist_(top|latest)_(au|pau|cau|ap|op|cp)_<k>_[<window>_]<EVENT>)"
        )
    event = m.group("event")
    if event not in HISTORY_EVENTS:
        raise ValueError(f"{name}: unknown event {event!r}; known: {sorted(HISTORY_EVENTS)}")
    window = None
    if m.group("window"):
        n = float(m.group("window"))
        window = n if m.group("unit") == "d" else n / 24.0
    return HistoryListSpec(
        name=name,
        kind=m.group("kind"),
        entity=m.group("entity"),
        k=int(m.group("k")),
        window_days=window,
        event=event,
    )


@dataclasses.dataclass(frozen=True)
class IdListFeatureSpec:
    name: str
    k: int

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"{self.name}: k must be > 0, got {self.k}")
        if self.name != self.name.strip() or not self.name:
            raise ValueError(f"id-list feature name has surrounding whitespace: {self.name!r}")
        family_of(self.name)
        cap = source_cap(self.name)
        if cap is not None and self.k > cap:
            raise ValueError(
                f"{self.name}: k={self.k} exceeds the {cap} ids the hydrator writes per list; "
                "the extra slots could never fill"
            )

    @property
    def family(self) -> IdListFamily:
        return family_of(self.name)

    @property
    def is_history(self) -> bool:
        return is_history_list(self.name)

    @property
    def feature_name(self) -> str:
        return feature_name_for(self.name, self.k)

    @property
    def identity(self) -> str:
        return identity_for(self.name)


def parse_spec(spec: str) -> IdListFeatureSpec:
    name, sep, k = spec.strip().partition(":")
    name = name.strip()
    return IdListFeatureSpec(name, int(k.strip()) if sep else default_k(name))


def parse_specs(specs: Sequence[str] | str) -> tuple[IdListFeatureSpec, ...]:
    if isinstance(specs, str):
        specs = [s for s in specs.split(",") if s.strip()]
    out = tuple(parse_spec(s) for s in specs)
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate id-list feature names: {names}")
    features = [s.feature_name for s in out]
    if len(set(features)) != len(features):
        raise ValueError(f"duplicate id-list feature names after k substitution: {features}")
    return out


def slot_offsets(specs: Sequence[IdListFeatureSpec]) -> list[int]:
    offsets = [0]
    for s in specs:
        offsets.append(offsets[-1] + s.k)
    return offsets


def total_k(specs: Sequence[IdListFeatureSpec]) -> int:
    return slot_offsets(specs)[-1]


def sidecar_path_for(batch_file_path: str, sidecar_root: str) -> str:
    return sidecar.sidecar_path_for(batch_file_path, sidecar_root, ".list_features.parquet")


def load_sidecar_column(sidecar_path: str, expected_rows: int | None = None) -> pa.Array:
    col = _as_array(pq.read_table(sidecar_path, columns=[COLUMN]).column(COLUMN))
    if expected_rows is not None and len(col) != expected_rows:
        raise ValueError(
            f"sidecar {sidecar_path} has {len(col)} rows, batch file has {expected_rows}"
        )
    return col


def canonical_column(col: pa.Array | pa.ChunkedArray) -> pa.Array:
    arr = _as_array(col)
    if arr.type.equals(CANONICAL_TYPE):
        return arr
    if pa.types.is_null(arr.type):
        return pa.nulls(len(arr), type=CANONICAL_TYPE)
    return arr.cast(CANONICAL_TYPE)


def null_column(num_rows: int) -> pa.Array:
    return pa.nulls(num_rows, type=CANONICAL_TYPE)


def set_column(batch: pa.RecordBatch, col: pa.Array) -> pa.RecordBatch:
    return sidecar.set_column(batch, COLUMN, col)


def empty_arrays(
    batch_size: int, specs: Sequence[IdListFeatureSpec]
) -> tuple[np.ndarray, np.ndarray]:
    width = total_k(specs)
    return np.zeros((batch_size, width), dtype=np.int64), np.zeros(
        (batch_size, width), dtype=np.bool_
    )


def _as_array(col: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 0:
            return pa.array([], type=col.type)
        combined = col.combine_chunks()
        if isinstance(combined, pa.ChunkedArray):
            return combined.chunk(0)
        return combined
    return col


def parse_id_lists(
    col: pa.Array | pa.ChunkedArray | None,
    specs: Sequence[IdListFeatureSpec],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids_out, mask_out = empty_arrays(batch_size, specs)
    if not specs or col is None:
        return ids_out, mask_out
    arr: pa.Array = _as_array(col)
    if len(arr) != batch_size:
        raise ValueError(f"{COLUMN}: {len(arr)} rows != batch_size {batch_size}")
    if pa.types.is_null(arr.type) or arr.null_count == len(arr):
        return ids_out, mask_out
    if not pa.types.is_list(arr.type) and not pa.types.is_large_list(arr.type):
        raise ValueError(f"{COLUMN}: expected List<Struct>, got {arr.type}")

    outer_offsets = np.asarray(arr.offsets, dtype=np.int64)
    base = outer_offsets[0]
    outer_offsets = outer_offsets - base
    row_valid = np.ones(batch_size, dtype=np.bool_)
    if arr.null_count:
        row_valid = arr.is_valid().to_numpy(zero_copy_only=False)
    structs = arr.values.slice(base, outer_offsets[-1])
    n_structs = len(structs)
    if n_structs == 0:
        return ids_out, mask_out
    struct_row = np.repeat(np.arange(batch_size, dtype=np.int64), np.diff(outer_offsets))
    struct_valid = row_valid[struct_row]
    if structs.null_count:
        struct_valid &= structs.is_valid().to_numpy(zero_copy_only=False)

    names = structs.field("name")
    ids_lists = structs.field("ids")
    dict_names = names.dictionary_encode()
    vocab = dict_names.dictionary.to_pylist()
    if not vocab:
        return ids_out, mask_out
    slot_of_vocab = np.full(len(vocab), -1, dtype=np.int64)
    k_of_slot = np.asarray([s.k for s in specs], dtype=np.int64)
    off_of_slot = np.asarray(slot_offsets(specs)[:-1], dtype=np.int64)
    spec_index = {s.name: i for i, s in enumerate(specs)}
    for vi, vname in enumerate(vocab):
        if vname in spec_index:
            slot_of_vocab[vi] = spec_index[vname]
    codes = dict_names.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    slot = np.where(codes >= 0, slot_of_vocab[np.clip(codes, 0, None)], -1)
    struct_valid &= slot >= 0

    inner_offsets = np.asarray(ids_lists.offsets, dtype=np.int64)
    inner_base = inner_offsets[0]
    inner_offsets = inner_offsets - inner_base
    flat_ids = ids_lists.values.slice(inner_base, inner_offsets[-1])
    if ids_lists.null_count:
        struct_valid &= ids_lists.is_valid().to_numpy(zero_copy_only=False)

    values = flat_ids.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    keep = values != 0
    if keep.all():
        valid_count = np.diff(inner_offsets)
        valid_idx = None
    else:
        cum = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
        valid_count = cum[inner_offsets[1:]] - cum[inner_offsets[:-1]]
        valid_idx = np.nonzero(keep)[0]
        inner_offsets = cum[inner_offsets]
    struct_valid &= valid_count > 0

    key = struct_row * (len(specs) + 1) + np.where(struct_valid, slot, len(specs))
    if n_structs > 1:
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        dup = np.zeros(n_structs, dtype=np.bool_)
        dup[order[1:]] = sorted_key[1:] == sorted_key[:-1]
        struct_valid &= ~dup

    take = np.where(struct_valid, np.minimum(valid_count, k_of_slot[np.clip(slot, 0, None)]), 0)
    total = int(take.sum())
    if total == 0:
        return ids_out, mask_out

    sel = np.nonzero(take)[0]
    rep = take[sel]
    pos = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(rep) - rep, rep)
    src = np.repeat(inner_offsets[sel], rep) + pos
    if valid_idx is not None:
        src = valid_idx[src]
    dst_row = np.repeat(struct_row[sel], rep)
    dst_col = np.repeat(off_of_slot[slot[sel]], rep) + pos
    ids_out[dst_row, dst_col] = values[src]
    mask_out[dst_row, dst_col] = True
    return ids_out, mask_out


def author_columns(specs: Sequence[IdListFeatureSpec]) -> np.ndarray:
    out = np.zeros(total_k(specs), dtype=np.bool_)
    offsets = slot_offsets(specs)
    for i, s in enumerate(specs):
        if s.family.is_author:
            out[offsets[i] : offsets[i + 1]] = True
    return out


def hash_id_lists(
    ids: np.ndarray,
    specs: Sequence[IdListFeatureSpec],
    item_hash: Callable[[np.ndarray], np.ndarray],
    author_hash: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    is_author = author_columns(specs)
    parts: list[np.ndarray] = []
    for sel, fn in ((~is_author, item_hash), (is_author, author_hash)):
        if sel.any():
            parts.append(np.asarray(fn(ids[:, sel]), dtype=np.int32))
    if not parts:
        return np.zeros((ids.shape[0], 0, 0), dtype=np.int32)
    num_hashes = parts[0].shape[-1]
    if any(p.shape[-1] != num_hashes for p in parts):
        raise ValueError(
            "id-list features need the same number of item and author hashes, got "
            f"{[p.shape[-1] for p in parts]}"
        )
    out = np.zeros((ids.shape[0], ids.shape[1], num_hashes), dtype=np.int32)
    i = 0
    for sel in (~is_author, is_author):
        if sel.any():
            out[:, sel] = parts[i]
            i += 1
    return out


def fill_rates(mask: np.ndarray, specs: Sequence[IdListFeatureSpec]) -> dict[str, float]:
    offsets = slot_offsets(specs)
    rows = max(mask.shape[0], 1)
    return {
        s.feature_name: float(mask[:, offsets[i] : offsets[i + 1]].any(axis=1).sum()) / rows
        for i, s in enumerate(specs)
    }


def fill_stats(
    mask: np.ndarray, specs: Sequence[IdListFeatureSpec]
) -> dict[str, tuple[float, float, float]]:
    offsets = slot_offsets(specs)
    rows = max(mask.shape[0], 1)
    out = {}
    for i, s in enumerate(specs):
        m = mask[:, offsets[i] : offsets[i + 1]]
        counts = m.sum(axis=1)
        out[s.feature_name] = (
            float((counts > 0).sum()) / rows,
            float((counts >= s.k).sum()) / rows,
            float(counts.sum()) / (rows * s.k),
        )
    return out


def fill_warnings(stats: dict[str, tuple[float, float, float]]) -> list[str]:
    out = []
    for name, (nonempty, full, _) in stats.items():
        if full > 0.95:
            out.append(
                f"{name}: full for {full:.0%} of users; k caps the list, consider raising it"
            )
        if nonempty < 0.05:
            out.append(f"{name}: non-empty for only {nonempty:.1%} of users; consider dropping it")
    return out
