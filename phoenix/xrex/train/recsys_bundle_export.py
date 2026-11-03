# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
import typing
import zlib
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import numpy as np

from xai_checkpointing.tree_util import tree_to_dict
from xrex.data.recsys.ads_late_window import compose_late_engagement_log_probs
from xrex.models.model_utils import unwrap_tree
from xrex.models.page_rl import PageDecodeInputs
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.sharding_context import make_legacy_sharding_context

if typing.TYPE_CHECKING:
    from xrex.train.trainer_recsys import RecsysTrainer

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_DIR = "export"
MANIFEST_NAME = f"{BUNDLE_DIR}/MANIFEST.json"
KIND_RANKING = "recsys_ranking_forward"
KIND_RETRIEVAL = "recsys_retrieval_forward"

POST_TABLE_KEY = "post_embeddings.embeddings"
POST_DATASET_TYPES_KEY = "post_embeddings.dataset_types"
POST_IDS_KEY = "post_embeddings.post_ids"
AUTHOR_IDS_KEY = "post_embeddings.author_ids"
POST_SCALES_KEY = "post_embeddings.scales"
DATASET_RANGES_KEY = "post_embeddings.dataset_ranges"
TOPIC_BITMAPS_KEY = "post_embeddings.topic_bitmaps"
MOL_SIDE_TABLE_KEY = "post_embeddings.mol_side_table"
TOPIC_QUERY_KEY = "topic_query"
TOPIC_FILTER_MODES = 6
DECODE_OUTPUT_NAMES = ("picked_idx", "picked_prob", "picked_score")

_RETRIEVAL_RUNNER_ATTRS = (
    "large_k",
    "retrieval_dataset_types",
    "enable_async_topk",
    "enable_radix_select_topk",
    "enable_int8_post_table",
    "enable_dataset_slice_topk",
    "enable_bloom_filter",
    "enable_topic_filter",
)

REQUEST_BATCH_KEYS: frozenset[str] = frozenset(
    {
        "user_hashes",
        "user_ip_hashes",
        "user_categorical_features",
        "user_bool_features",
        "user_float_features",
        "user_int64_features",
        "user_installed_apps_multihot",
        "user_conversion_history_hashes",
        "history_seq.post_hashes",
        "history_seq.auth_hashes",
        "history_seq.product_surface",
        "history_seq.actions",
        "history_seq.continuous_actions",
        "history_seq.impr_ts",
        "history_seq.post_creation_ts_sec",
        "history_seq.categorical_features",
        "history_seq.bool_features",
        "history_seq.float_features",
        "history_seq.int64_features",
        "history_seq.post_sids",
        "candidate_seq.post_hashes",
        "candidate_seq.auth_hashes",
        "candidate_seq.product_surface",
        "candidate_seq.impr_ts",
        "candidate_seq.post_creation_ts_sec",
        "candidate_seq.categorical_features",
        "candidate_seq.bool_features",
        "candidate_seq.float_features",
        "candidate_seq.int64_features",
        "candidate_seq.embedding",
        "candidate_seq.search_query_embeddings",
        "candidate_seq.line_item_ids",
        "candidate_seq.campaign_ids",
        "candidate_seq.funding_instrument_ids",
        "candidate_seq.conversion_dense_features",
        "candidate_seq.account_hashes",
        "candidate_seq.post_sids",
    }
)


def _template_fill(key: str, value: Any) -> bool | int | float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0
    first = arr.reshape(-1)[0]
    if not np.all(arr == first):
        raise ValueError(
            f"batch key {key!r} is not in REQUEST_BATCH_KEYS but its example_data template "
            "is not constant; register it as a request feature or make the template uniform"
        )
    if arr.dtype == np.bool_:
        return bool(first)
    if np.issubdtype(arr.dtype, np.integer):
        return int(first)
    if np.issubdtype(arr.dtype, np.floating):
        fill = float(first)
        if not np.isfinite(fill):
            raise ValueError(f"batch key {key!r}: non-finite template fill {fill}")
        return fill
    raise ValueError(f"batch key {key!r}: unsupported template dtype {arr.dtype}")


def restamp_manifest(data: bytes) -> bytes:
    manifest = json.loads(data)
    manifest["created_timestamp"] = time.time()
    return json.dumps(manifest, indent=2).encode()


class EmbeddingSlices(NamedTuple):
    hist_post_end: int
    hist_auth_end: int
    cand_post_end: int
    cand_auth_end: int
    user_end: int
    user_ip_end: int


class PackedGeometry(NamedTuple):
    packed_history_len: int
    packed_candidate_len: int
    bs_per_device: int
    merged_batch: int


@dataclasses.dataclass(frozen=True)
class BundleFile:
    name: str
    data: bytes


@dataclasses.dataclass(frozen=True)
class RetrievalExport:
    large_k: int
    target_dataset_types: tuple[tuple[str, int], ...]
    use_async_topk: bool
    use_radix_select_topk: bool
    post_table_shape: tuple[int, int]
    post_table_dtype: str
    dataset_types_shape: tuple[int, ...]
    dataset_types_dtype: str
    int8_post_table: bool = False
    dataset_capacities: tuple[int, ...] | None = None
    rows_from: tuple[tuple[int, ...], ...] | None = None
    max_age_seconds: tuple[float | None, ...] | None = None
    optional_targets: tuple[bool, ...] | None = None
    topic_filter: bool = False
    mol_side_table_shape: tuple[int, int] | None = None

    @property
    def output_names(self) -> list[str]:
        names = []
        for name, _ in self.target_dataset_types:
            names.append(f"indices_{name}")
            names.append(f"scores_{name}")
        return names

    def program_inputs(self, bs: int) -> list[tuple[str, str, Any, str]]:
        from xrex.models.topic_categories import NUM_TOPIC_INT32S

        rows = self.post_table_shape[0]
        table_dtype = np.dtype(np.int8 if self.int8_post_table else self.post_table_dtype)
        inputs = [
            (
                "post_table",
                POST_TABLE_KEY,
                jax.ShapeDtypeStruct(self.post_table_shape, table_dtype),
                "rows",
            ),
            (
                "dataset_types",
                POST_DATASET_TYPES_KEY,
                jax.ShapeDtypeStruct(self.dataset_types_shape, np.dtype(self.dataset_types_dtype)),
                "replicated",
            ),
        ]
        if self.int8_post_table:
            inputs.append(
                ("post_scales", POST_SCALES_KEY, jax.ShapeDtypeStruct((rows,), np.float32), "rows")
            )
        if self.dataset_capacities is not None:
            ranges = jax.ShapeDtypeStruct((len(self.target_dataset_types), 2), np.int32)
            inputs.append(("dataset_ranges", DATASET_RANGES_KEY, ranges, "replicated"))
        if self.topic_filter:
            bitmaps = jax.ShapeDtypeStruct((rows, TOPIC_FILTER_MODES * NUM_TOPIC_INT32S), np.int32)
            query = jax.ShapeDtypeStruct((bs, NUM_TOPIC_INT32S + 1), np.int32)
            inputs.append(("topic_bitmaps", TOPIC_BITMAPS_KEY, bitmaps, "rows"))
            inputs.append(("topic_query", TOPIC_QUERY_KEY, query, "replicated"))
        if self.mol_side_table_shape is not None:
            side = jax.ShapeDtypeStruct(self.mol_side_table_shape, np.float32)
            inputs.append(("weight", MOL_SIDE_TABLE_KEY, side, "replicated"))
        return inputs

    @property
    def derived_kinds(self) -> list[str]:
        return [kind for kind, _, _, _ in self.program_inputs(1)[2:]]

    def manifest_block(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "large_k": self.large_k,
            "target_dataset_types": [
                _target_manifest_entry(self, i, name, value)
                for i, (name, value) in enumerate(self.target_dataset_types)
            ],
            "post_table": {
                "key": POST_TABLE_KEY,
                "rows": self.post_table_shape[0],
                "width": self.post_table_shape[1],
                "dtype": self.post_table_dtype,
            },
            "dataset_types_key": POST_DATASET_TYPES_KEY,
            "post_ids_key": POST_IDS_KEY,
            "author_ids_key": AUTHOR_IDS_KEY,
            "topk_unordered": self.use_radix_select_topk,
        }
        if self.int8_post_table:
            block["post_scales_key"] = POST_SCALES_KEY
        if self.dataset_capacities is not None:
            block["dataset_ranges_key"] = DATASET_RANGES_KEY
            block["dataset_capacities"] = list(self.dataset_capacities)
        if self.topic_filter:
            block["topic_filter"] = _topic_filter_manifest_block()
        return block


def _topic_filter_manifest_block() -> dict[str, Any]:
    from xrex.models import topic_categories as tc

    none: frozenset[int] = frozenset()
    post_bits = {
        eid: sorted(tc.TOPIC_ID_TO_BITS.get(eid, none) | tc._POST_SIDE_GROUP_MEMBERS.get(eid, none))
        for eid in sorted(set(tc.TOPIC_ID_TO_BITS) | set(tc._POST_SIDE_GROUP_MEMBERS))
    }
    return {
        "bitmaps_key": TOPIC_BITMAPS_KEY,
        "query_key": TOPIC_QUERY_KEY,
        "num_modes": TOPIC_FILTER_MODES,
        "num_words": tc.NUM_TOPIC_INT32S,
        "post_bits": [[eid, bits] for eid, bits in post_bits.items()],
        "query_bits": [[eid, sorted(bits)] for eid, bits in sorted(tc.TOPIC_ID_TO_BITS.items())],
    }


def _target_manifest_entry(
    spec: RetrievalExport, index: int, name: str, value: int
) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "value": value}
    rows = spec.rows_from[index] if spec.rows_from is not None else (value,)
    if rows != (value,):
        entry["rows_from"] = list(rows)
    if spec.max_age_seconds is not None and spec.max_age_seconds[index] is not None:
        entry["max_age_seconds"] = int(spec.max_age_seconds[index])
    if spec.optional_targets is not None and spec.optional_targets[index]:
        entry["optional"] = True
    return entry


_parameter_serialization_registered = False


def _pspec_to_json(pspec: Any) -> list[Any]:
    return [None if e is None else (e if isinstance(e, str) else list(e)) for e in tuple(pspec)]


def _pspec_from_json(entries: list[Any]) -> Any:
    from jax.sharding import PartitionSpec

    return PartitionSpec(
        *[None if e is None else (e if isinstance(e, str) else tuple(e)) for e in entries]
    )


def ensure_parameter_serialization_registered() -> None:
    global _parameter_serialization_registered
    if _parameter_serialization_registered:
        return
    from jax import export as jax_export

    from xrex.models.model_utils import Parameter

    def serialize_auxdata(aux: dict[str, Any]) -> bytes:
        aux = dict(aux)
        aux["pspec"] = _pspec_to_json(aux["pspec"])
        return json.dumps(aux, default=list).encode()

    def deserialize_auxdata(data: bytes) -> dict[str, Any]:
        aux = json.loads(data)
        aux["pspec"] = _pspec_from_json(aux["pspec"])
        if isinstance(aux.get("rms_clip_axes"), list):
            aux["rms_clip_axes"] = tuple(aux["rms_clip_axes"])
        return aux

    jax_export.register_pytree_node_serialization(
        Parameter,
        serialized_name="xrex.models.model_utils.Parameter",
        serialize_auxdata=serialize_auxdata,
        deserialize_auxdata=deserialize_auxdata,
    )
    _parameter_serialization_registered = True


_packing_layout_serialization_registered = False


def ensure_packing_layout_serialization_registered() -> None:
    global _packing_layout_serialization_registered
    if _packing_layout_serialization_registered:
        return
    from jax import export as jax_export

    from xrex.data.recsys.sequence_packing import SequencePackedLayout

    jax_export.register_pytree_node_serialization(
        SequencePackedLayout,
        serialized_name="xrex.data.recsys.sequence_packing.SequencePackedLayout",
        serialize_auxdata=lambda aux: json.dumps(list(aux)).encode(),
        deserialize_auxdata=lambda data: tuple(json.loads(data)),
    )
    try:
        from xrex.cutedsl.ranker_attention_varlen_fa4 import (
            BlockSparseLayout,
        )
    except ImportError:
        pass
    else:
        jax_export.register_pytree_node_serialization(
            BlockSparseLayout,
            serialized_name="xrex.cutedsl.ranker_attention_varlen_fa4.BlockSparseLayout",
            serialize_auxdata=lambda aux: json.dumps(list(aux)).encode(),
            deserialize_auxdata=lambda data: tuple(json.loads(data)),
        )
    _packing_layout_serialization_registered = True


_page_decode_serialization_registered = False


def ensure_page_decode_serialization_registered() -> None:
    global _page_decode_serialization_registered
    if _page_decode_serialization_registered:
        return
    from jax import export as jax_export

    jax_export.register_namedtuple_serialization(
        PageDecodeInputs, serialized_name="xrex.models.page_rl.PageDecodeInputs"
    )
    _page_decode_serialization_registered = True


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def _aval_entry(leaf: Any) -> dict[str, Any]:
    return {"shape": [int(d) for d in leaf.shape], "dtype": _dtype_name(leaf.dtype)}


def _to_shape_dtype_struct(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, jax.dtypes.canonicalize_dtype(leaf.dtype)),
        tree,
    )


def _serving_sequence_len(
    *,
    training_seq_len: int,
    training_history_seq_len: int,
    training_candidate_seq_len: int,
    num_negatives_per_example: int,
    num_global_negatives_per_example: int,
    num_user_prefix_tokens: int,
    history_seq_len: int,
    candidate_seq_len: int,
) -> int:
    prefix_tokens = (
        training_seq_len
        - training_history_seq_len
        - training_candidate_seq_len * (1 + num_negatives_per_example)
        - num_global_negatives_per_example
    )
    if prefix_tokens != num_user_prefix_tokens:
        raise ValueError(
            f"cannot derive the serving sequence_len: training sequence_len "
            f"{training_seq_len} - (history {training_history_seq_len} + candidate "
            f"{training_candidate_seq_len} * (1 + {num_negatives_per_example} negatives) "
            f"+ {num_global_negatives_per_example} global negatives) = {prefix_tokens} "
            f"prefix tokens, but the model reserves num_user_prefix_tokens="
            f"{num_user_prefix_tokens}; this config's sequence_len does not follow the "
            "prefix + history + candidates construction, so the export cannot size the "
            "attention kernel for serving"
        )
    return prefix_tokens + history_seq_len + candidate_seq_len


def _user_tower_sequence_len(
    *,
    training_seq_len: int,
    training_history_seq_len: int,
    num_user_prefix_tokens: int,
    history_seq_len: int,
) -> int:
    prefix_tokens = training_seq_len - training_history_seq_len
    if prefix_tokens != num_user_prefix_tokens:
        raise ValueError(
            f"cannot derive the serving user-tower sequence_len: sequence_len "
            f"{training_seq_len} - history_seq_len {training_history_seq_len} = {prefix_tokens} "
            f"prefix tokens, but the model reserves num_user_prefix_tokens="
            f"{num_user_prefix_tokens}. The user tower is built as prefix + history; if "
            "history_seq_len was overridden for serving, also override "
            "model_config.user_tower_config.model_config.sequence_len to "
            f"{num_user_prefix_tokens + history_seq_len} (prefix + serving history), as the "
            "jax slot does — otherwise the attention kernel cannot be sized for serving"
        )
    return prefix_tokens + history_seq_len


def _make_export_config(
    trainer: RecsysTrainer,
    history_seq_len: int,
    candidate_seq_len: int,
    *,
    two_tower: bool = False,
    mesh_devices: int = 1,
):
    from xrex.configs.config_loader import replace_cli_subs

    init_params = trainer.to_dict()
    init_params.pop("__class")
    export_cfg = type(trainer).from_dict(init_params, ensure_class=type(trainer))

    dataset = trainer.dataset
    if two_tower:
        serving_seq_len = _user_tower_sequence_len(
            training_seq_len=int(trainer.model_config.model_config.sequence_len),
            training_history_seq_len=int(dataset.history_seq_len),
            num_user_prefix_tokens=int(trainer.model_config.num_user_prefix_tokens),
            history_seq_len=history_seq_len,
        )
        seq_len_field = "model_config.user_tower_config.model_config"
    else:
        serving_seq_len = _serving_sequence_len(
            training_seq_len=int(trainer.model_config.model_config.sequence_len),
            training_history_seq_len=int(dataset.history_seq_len),
            training_candidate_seq_len=int(dataset.candidate_seq_len),
            num_negatives_per_example=int(getattr(dataset, "num_negatives_per_example", 0)),
            num_global_negatives_per_example=int(
                getattr(dataset, "num_global_negatives_per_example", 0)
            ),
            num_user_prefix_tokens=int(trainer.model_config.num_user_prefix_tokens),
            history_seq_len=history_seq_len,
            candidate_seq_len=candidate_seq_len,
        )
        seq_len_field = "model_config.model_config"

    if mesh_devices < 1:
        raise ValueError(f"mesh_devices must be >= 1, got {mesh_devices}")
    overrides = [
        f"num_devices_per_process={mesh_devices}",
        f"ep={mesh_devices}",
        "dp=1",
        "num_negatives_per_example=0",
        "num_global_negatives_per_example=0",
        f"history_seq_len={history_seq_len}",
        f"candidate_seq_len={candidate_seq_len}",
        f"{seq_len_field}.sequence_len={serving_seq_len}",
        f"{seq_len_field}.attn_config.sequence_len={serving_seq_len}",
    ]
    export_cfg, used = replace_cli_subs(export_cfg, overrides)
    unused = [k for k, v in used.items() if not v]
    if unused:
        raise ValueError(f"StableHLO bundle export overrides did not match config: {unused}")
    return export_cfg


def _token_feature_config(model_config: Any, *, two_tower: bool) -> Any:
    return model_config.user_tower_config if two_tower else model_config


def _batch_template(
    export_cfg: Any, bs: int, *, packed: bool, packed_seq_len: int | None = None
) -> Any:
    model_config = export_cfg.model_config
    batch = export_cfg.dataset.example_data(bs)

    if packed:
        from xrex.data.recsys.sequence_packing import compact_candidate_layout, pack_batch

        batch = pack_batch(
            batch=batch,
            num_devices_per_process=export_cfg.parallel_config.num_devices_per_process,
            num_user_prefix_tokens=model_config.num_user_prefix_tokens,
            dist=None,
            rng=None,
            block_size=export_cfg._seqpack_block_size,
            alloc=export_cfg._alloc,
        )
        layout = batch["packing_layout"]
        if packed_seq_len is not None and packed_seq_len != int(layout.segment_ids.shape[1]):
            batch["packing_layout"] = compact_candidate_layout(
                batch,
                num_user_prefix_tokens=model_config.num_user_prefix_tokens,
                block_size=export_cfg._seqpack_block_size,
                packed_seq_len=packed_seq_len,
            )
        if getattr(export_cfg, "using_fa4", False):
            batch = export_cfg.add_block_sparse_layout(batch)
        layout = batch["packing_layout"]
        if layout.cand_slot_lens is not None:
            batch["packing_layout"] = dataclasses.replace(layout, cand_slot_lens=None)
    elif packed_seq_len is not None:
        raise ValueError("packed_seq_len applies to sequence-packed exports only")
    return batch


def _packed_seq_len_rungs(trainer: Any, bs: int, packed: bool) -> tuple[int | None, ...]:
    if not packed or not getattr(trainer, "compact_candidate_slots", False):
        return (None,)
    full = int(trainer.full_packed_seq_len(bs))
    lens = tuple(int(n) for n in trainer.packed_seq_lens(bs))
    if not lens or lens[-1] != full:
        raise AssertionError(f"bs={bs}: packed_seq_lens {lens} must end at the full row {full}")
    return tuple(None if n == full else n for n in lens)


def _batch_avals(
    export_cfg: Any, bs: int, *, packed: bool, template: Any = None, two_tower: bool = False
) -> Any:
    model_config = export_cfg.model_config
    features = _token_feature_config(model_config, two_tower=two_tower)
    if template is None:
        template = _batch_template(export_cfg, bs, packed=packed)

    batch = _to_shape_dtype_struct(template)

    if model_config.multimodal_embedding_type is not None:
        cand_post = batch["candidate_seq"]["post_hashes"]
        batch["candidate_seq"]["embedding"] = jax.ShapeDtypeStruct(
            (cand_post.shape[0], cand_post.shape[1], features.multimodal_embedding_dim),
            np.float32,
        )

    if features.use_post_sid and features.sid_num_levels > 0:
        for seq_name in ("history_seq", "candidate_seq"):
            post_hashes = batch[seq_name]["post_hashes"]
            batch[seq_name]["post_sids"] = jax.ShapeDtypeStruct(
                (post_hashes.shape[0], post_hashes.shape[1], features.sid_num_levels),
                np.uint16,
            )

    return batch


def _embedding_slices(export_cfg: Any, history_seq_len: int, candidate_seq_len: int):
    ht = export_cfg.dataset.hash_table
    hist_post_seq = ht.num_item_hashes * history_seq_len
    hist_auth_seq = ht.num_author_hashes * history_seq_len
    cand_post_seq = ht.num_item_hashes * candidate_seq_len
    cand_auth_seq = ht.num_author_hashes * candidate_seq_len
    user_seq = ht.num_user_hashes
    ip_seq = ht.num_ip_hashes if export_cfg.model_config.use_ip_address else 0

    user_end = hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq
    return EmbeddingSlices(
        hist_post_end=hist_post_seq,
        hist_auth_end=hist_post_seq + hist_auth_seq,
        cand_post_end=hist_post_seq + hist_auth_seq + cand_post_seq,
        cand_auth_end=hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq,
        user_end=user_end,
        user_ip_end=user_end + ip_seq,
    )


def _packed_embedding_slices(
    export_cfg: Any, batch_avals: Any
) -> tuple[EmbeddingSlices, PackedGeometry]:
    hist = batch_avals["history_seq"]
    cand = batch_avals["candidate_seq"]
    hist_post_seq = int(np.prod(hist["post_hashes"].shape[1:]))
    hist_auth_seq = int(np.prod(hist["auth_hashes"].shape[1:]))
    cand_post_seq = int(np.prod(cand["post_hashes"].shape[1:]))
    cand_auth_seq = int(np.prod(cand["auth_hashes"].shape[1:]))
    user_seq = int(np.prod(batch_avals["user_hashes"].shape[1:]))
    ip_seq = (
        int(np.prod(batch_avals["user_ip_hashes"].shape[1:]))
        if export_cfg.model_config.use_ip_address
        else 0
    )

    user_end = hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq
    slices = EmbeddingSlices(
        hist_post_end=hist_post_seq,
        hist_auth_end=hist_post_seq + hist_auth_seq,
        cand_post_end=hist_post_seq + hist_auth_seq + cand_post_seq,
        cand_auth_end=hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq,
        user_end=user_end,
        user_ip_end=user_end + ip_seq,
    )
    geometry = PackedGeometry(
        packed_history_len=int(hist["post_hashes"].shape[1]),
        packed_candidate_len=int(cand["post_hashes"].shape[1]),
        bs_per_device=int(batch_avals["user_hashes"].shape[1]),
        merged_batch=int(batch_avals["user_hashes"].shape[0]),
    )
    return slices, geometry


def _recsys_embeddings_from_merged(
    merged_embeddings: jax.Array,
    sl: EmbeddingSlices,
    packed_geometry: PackedGeometry | None,
) -> RecsysEmbeddings:
    if packed_geometry is None:
        return RecsysEmbeddings(
            history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :],
            history_author_embeddings=merged_embeddings[:, sl.hist_post_end : sl.hist_auth_end, :],
            candidate_post_embeddings=merged_embeddings[:, sl.hist_auth_end : sl.cand_post_end, :],
            candidate_author_embeddings=merged_embeddings[
                :, sl.cand_post_end : sl.cand_auth_end, :
            ],
            user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :],
            user_ip_embeddings=(
                merged_embeddings[:, sl.user_end : sl.user_ip_end, :]
                if sl.user_ip_end > sl.user_end
                else None
            ),
        )

    g = packed_geometry
    merged = merged_embeddings

    def _section(start: int, end: int, rows: int) -> jax.Array:
        return merged[:, start:end, :].reshape(merged.shape[0], rows, -1, merged.shape[-1])

    return RecsysEmbeddings(
        history_post_embeddings=_section(0, sl.hist_post_end, g.packed_history_len),
        history_author_embeddings=_section(
            sl.hist_post_end, sl.hist_auth_end, g.packed_history_len
        ),
        candidate_post_embeddings=_section(
            sl.hist_auth_end, sl.cand_post_end, g.packed_candidate_len
        ),
        candidate_author_embeddings=_section(
            sl.cand_post_end, sl.cand_auth_end, g.packed_candidate_len
        ),
        user_embeddings=_section(sl.cand_auth_end, sl.user_end, g.bs_per_device),
        user_ip_embeddings=(
            _section(sl.user_end, sl.user_ip_end, g.bs_per_device)
            if sl.user_ip_end > sl.user_end
            else None
        ),
    )


def _make_forward_fn(
    export_cfg: Any,
    embedding_slices: EmbeddingSlices,
    mesh: jax.sharding.Mesh,
    packed_geometry: PackedGeometry | None = None,
):
    import haiku as hk
    import jax.numpy as jnp

    model_config = export_cfg.model_config

    @hk.transform
    def forward_fn(
        batch: Any, merged_embeddings: jax.Array, page_decode: PageDecodeInputs | None = None
    ):
        recsys_embeddings = _recsys_embeddings_from_merged(
            merged_embeddings, embedding_slices, packed_geometry
        )
        model = model_config.make(sharding_context=make_legacy_sharding_context(mesh))
        outputs = model.forward(batch, recsys_embeddings, page_decode)
        log_probs = jax.nn.log_sigmoid(outputs[0]).astype(jnp.bfloat16).astype(jnp.float32)
        if getattr(model_config, "serve_late_window_composed", False):
            log_probs = compose_late_engagement_log_probs(log_probs)
        cont_preds = outputs[1].astype(jnp.bfloat16).astype(jnp.float32)
        has_nan = jnp.any(jnp.isnan(log_probs), axis=tuple(range(1, log_probs.ndim)))
        return log_probs, cont_preds, has_nan, *outputs[2:]

    return forward_fn


def _make_retrieval_forward_fn(
    export_cfg: Any,
    embedding_slices: EmbeddingSlices,
    mesh: jax.sharding.Mesh,
    retrieval: RetrievalExport,
    packed_geometry: PackedGeometry | None = None,
):
    import haiku as hk

    model_config = export_cfg.model_config
    target_values = tuple(value for _, value in retrieval.target_dataset_types)
    derived_kinds = retrieval.derived_kinds

    def topic_inputs(by_kind: dict[str, jax.Array]) -> tuple[jax.Array | None, jax.Array | None]:
        if not retrieval.topic_filter:
            return None, None
        import jax.numpy as jnp

        from xrex.models.topic_categories import NUM_TOPIC_INT32S

        query = by_kind["topic_query"]
        bitmaps = by_kind["topic_bitmaps"]
        by_mode = bitmaps.reshape(bitmaps.shape[0], TOPIC_FILTER_MODES, NUM_TOPIC_INT32S)
        mode = jnp.clip(query[0, NUM_TOPIC_INT32S], 0, TOPIC_FILTER_MODES - 1)
        selected = jax.lax.dynamic_index_in_dim(by_mode, mode, axis=1, keepdims=False)
        return selected, query[:, :NUM_TOPIC_INT32S]

    @hk.transform
    def forward_fn(
        batch: Any,
        merged_embeddings: jax.Array,
        post_table: jax.Array,
        dataset_types: jax.Array,
        *derived: jax.Array,
    ):
        recsys_embeddings = _recsys_embeddings_from_merged(
            merged_embeddings, embedding_slices, packed_geometry
        )
        by_kind = dict(zip(derived_kinds, derived, strict=True))
        topic_bitmaps, topic_user_bitmasks = topic_inputs(by_kind)
        model = model_config.make(sharding_context=make_legacy_sharding_context(mesh))
        mol_side_tables = None
        side_table = by_kind.get("weight")
        if side_table is not None:
            mol_side_tables = model_config.mol_side_tables_from_rows(side_table)
        results = model.forward(
            batch,
            recsys_embeddings,
            post_table,
            dataset_types,
            retrieval.large_k,
            target_values,
            None,
            topic_bitmaps=topic_bitmaps,
            topic_user_bitmasks=topic_user_bitmasks,
            dataset_ranges=by_kind.get("dataset_ranges"),
            use_async_topk=retrieval.use_async_topk,
            use_radix_select_topk=retrieval.use_radix_select_topk,
            post_scales=by_kind.get("post_scales"),
            dataset_capacities=retrieval.dataset_capacities,
            mol_side_tables=mol_side_tables,
        )
        flat: list[jax.Array] = []
        for indices, scores in results:
            flat.append(indices)
            flat.append(scores)
        return tuple(flat)

    return forward_fn


def _scan_custom_call_targets(lowered_text: str) -> list[str]:
    targets = set(re.findall(r"stablehlo\.custom_call\s*@([\w.$-]+)", lowered_text))
    targets |= set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', lowered_text))
    return sorted(targets)


SHARDING_REPLICATED: dict[str, Any] = {"kind": "replicated"}


def _export_mesh(export_cfg: Any, mesh_devices: int) -> jax.sharding.Mesh:
    axis_names = export_cfg.parallel_config.mesh_axis_names()
    mesh_shape = export_cfg.parallel_config.mesh_shape()
    if int(np.prod(mesh_shape)) != mesh_devices:
        raise AssertionError(f"export mesh {mesh_shape} does not have {mesh_devices} devices")
    devices = jax.local_devices()
    if len(devices) < mesh_devices:
        raise ValueError(
            f"the export needs {mesh_devices} local devices for the serving mesh, "
            f"{len(devices)} visible (CUDA_VISIBLE_DEVICES?)"
        )
    return jax.sharding.Mesh(np.array(devices[:mesh_devices]).reshape(mesh_shape), axis_names)


def _retrieval_shardings(
    mesh: Any,
    data_axis: tuple[str, ...],
    bs: int,
    params_avals: Any,
    batch_avals: Any,
    retrieval: RetrievalExport,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from jax.sharding import NamedSharding, PartitionSpec

    replicated = NamedSharding(mesh, PartitionSpec())
    rows = NamedSharding(mesh, PartitionSpec(data_axis))

    def batch_leaf_sharding(path: Any, leaf: Any) -> Any:
        if len(leaf.shape) < 1 or int(leaf.shape[0]) != bs:
            raise ValueError(
                f"batch leaf {jax.tree_util.keystr(path)} has shape {tuple(leaf.shape)}, "
                f"expected a leading batch dim of {bs} for the mesh export"
            )
        return rows

    batch_shardings = jax.tree_util.tree_map_with_path(batch_leaf_sharding, batch_avals)
    params_shardings = jax.tree.map(lambda _: replicated, params_avals)
    in_shardings = (
        params_shardings,
        replicated,
        batch_shardings,
        rows,
        *(
            rows if layout == "rows" else replicated
            for _, _, _, layout in retrieval.program_inputs(bs)
        ),
    )
    return in_shardings, (replicated,) * len(retrieval.output_names)


def _sharding_entry(hlo_sharding: Any, ndim: int, num_devices: int, what: str) -> dict[str, Any]:
    if num_devices == 1 or hlo_sharding is None or hlo_sharding.is_replicated():
        if hlo_sharding is None and num_devices != 1:
            raise AssertionError(f"{what}: the mesh export left the sharding unspecified")
        return dict(SHARDING_REPLICATED)
    if not hlo_sharding.is_tiled():
        raise NotImplementedError(f"{what}: unsupported sharding {hlo_sharding}")
    dims = [int(d) for d in hlo_sharding.tile_assignment_dimensions()]
    if hlo_sharding.replicate_on_last_tile_dim():
        if dims[-1] != 1:
            raise NotImplementedError(f"{what}: partially replicated sharding {hlo_sharding}")
        dims = dims[:-1]
    if len(dims) != ndim:
        raise AssertionError(f"{what}: sharding {hlo_sharding} does not match rank {ndim}")
    tiled = [i for i, d in enumerate(dims) if d != 1]
    if len(tiled) != 1 or dims[tiled[0]] != num_devices:
        raise NotImplementedError(
            f"{what}: sharding {hlo_sharding} is not a single-dim split over {num_devices} devices"
        )
    devices = [int(d) for d in hlo_sharding.tile_assignment_devices()]
    if devices != list(range(num_devices)):
        raise NotImplementedError(
            f"{what}: sharding {hlo_sharding} does not place block i on partition i"
        )
    return {"kind": "tiled", "dim": tiled[0]}


def _mesh_block(mesh: Any, use_shardy: bool) -> dict[str, Any]:
    return {
        "num_devices": int(mesh.size),
        "axis_names": [str(n) for n in mesh.axis_names],
        "axis_sizes": [int(mesh.shape[n]) for n in mesh.axis_names],
        "use_shardy": bool(use_shardy),
    }


def _input_spec(
    params_avals: Any,
    rng_aval: Any,
    batch_avals: Any,
    merged_aval: Any,
    batch_template: Any,
    retrieval_inputs: list[tuple[str, str, Any, str]] | None = None,
    page_decode_avals: PageDecodeInputs | None = None,
) -> list[dict[str, Any]]:
    from xai_checkpointing.tree_util import keystr

    template_leaves = {
        keystr(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(batch_template)[0]
    }
    spec: list[dict[str, Any]] = []

    param_leaves = jax.tree.leaves(params_avals)
    param_keys = list(tree_to_dict(unwrap_tree(params_avals), keep_none=False).keys())
    if len(param_keys) != len(param_leaves):
        raise AssertionError(
            f"params flatten mismatch: {len(param_keys)} keys vs {len(param_leaves)} leaves"
        )
    for key, leaf in zip(param_keys, param_leaves):
        spec.append({"kind": "weight", "key": f"params.{key}", **_aval_entry(leaf)})

    spec.append({"kind": "rng", "key": "rng", **_aval_entry(rng_aval)})

    for path, leaf in jax.tree_util.tree_flatten_with_path(batch_avals)[0]:
        key = keystr(path)
        if key.startswith("packing_layout."):
            spec.append({"kind": "packing_layout", "key": key, **_aval_entry(leaf)})
        elif key in REQUEST_BATCH_KEYS:
            spec.append({"kind": "batch", "key": key, "source": "request", **_aval_entry(leaf)})
        else:
            if key not in template_leaves:
                raise ValueError(
                    f"batch key {key!r} is not in REQUEST_BATCH_KEYS and has no example_data "
                    "template value; register it as a request feature"
                )
            fill = _template_fill(key, template_leaves[key])
            spec.append(
                {
                    "kind": "batch",
                    "key": key,
                    "source": "template",
                    "fill": fill,
                    **_aval_entry(leaf),
                }
            )

    spec.append(
        {"kind": "merged_embeddings", "key": "merged_embeddings", **_aval_entry(merged_aval)}
    )
    for kind, key, aval, _ in retrieval_inputs or ():
        spec.append({"kind": kind, "key": key, **_aval_entry(aval)})
    if page_decode_avals is not None:
        for name, aval in page_decode_avals._asdict().items():
            spec.append({"kind": "page_decode", "key": f"page_decode.{name}", **_aval_entry(aval)})
    return spec


def _page_decode_avals(bs: int, candidate_seq_len: int) -> PageDecodeInputs:
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        PageDecodeInputs.no_decode(bs, candidate_seq_len),
    )


def _retrieval_export(
    runner: Any, dataset_capacities: Mapping[str, int] | None = None
) -> RetrievalExport:
    missing = [name for name in _RETRIEVAL_RUNNER_ATTRS if not hasattr(runner, name)]
    if missing:
        raise NotImplementedError(
            "two-tower StableHLO export needs a RetrievalModelRunner (host-side "
            f"export_native_bundle); trainer lacks {missing}"
        )
    if runner.enable_bloom_filter:
        raise NotImplementedError(
            "two-tower StableHLO export does not support enable_bloom_filter; the native "
            "runtime has no bloom filter"
        )
    topic_filter = bool(runner.enable_topic_filter)
    if topic_filter and dataset_capacities is not None:
        raise ValueError(
            "enable_topic_filter masks the whole table per dataset; it cannot be combined with "
            "--dataset_capacities"
        )
    if runner.enable_dataset_slice_topk and dataset_capacities is None and not topic_filter:
        logger.warning(
            "enable_dataset_slice_topk=True without dataset capacities: the slice path compiles "
            "the checkpoint's per-dataset post-table ranges into the program, so the exported "
            "program masks by dataset type instead (same results, top-k over the whole table); "
            "pass --dataset_capacities for the windowed top-k"
        )
    if os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1":
        raise ValueError(
            "DEBUG_ALLOW_RANDOM_INIT=1 would export a two-tower program without the dataset "
            "type mask; unset it for the export"
        )
    large_k = int(runner.large_k)
    if large_k < 1:
        raise ValueError(f"invalid large_k for the two-tower export: {large_k}")
    datasets = list(runner.retrieval_dataset_types)
    if not datasets:
        raise ValueError("two-tower export needs at least one retrieval dataset type")
    split_home = bool(getattr(runner, "split_home_checkpoint", False)) or bool(
        getattr(getattr(runner, "model_config", None), "split_home_checkpoint", False)
    )
    if split_home and any(ds.name == "HOME" for ds in datasets):
        if topic_filter:
            raise NotImplementedError(
                "enable_topic_filter with split_home_checkpoint: the topic path masks stored "
                "type 1 only, not the cold|hot HOME window"
            )
        if dataset_capacities is None:
            raise ValueError(
                "split_home_checkpoint requires --dataset_capacities so HOME is the "
                "cold|hot window, not a mask for stored type 1"
            )
        if "HOME_COLD" in dataset_capacities and not any(ds.name == "HOME_COLD" for ds in datasets):
            from xrex.data.retrieval_dataset import RetrievalDataset

            datasets.append(RetrievalDataset.HOME_COLD)
    post_embeddings = runner.state_shape.post_embeddings
    table = post_embeddings.embeddings.x
    if len(table.shape) != 2:
        raise ValueError(f"post table must be [rows, width], got {tuple(table.shape)}")
    capacities = None
    if dataset_capacities is not None:
        names = [ds.name for ds in datasets]
        if sorted(dataset_capacities) != sorted(names):
            raise ValueError(
                f"dataset_capacities must name exactly the target datasets {names}, got "
                f"{sorted(dataset_capacities)}"
            )
        capacities = tuple(int(dataset_capacities[name]) for name in names)
        for name, capacity in zip(names, capacities, strict=True):
            if not large_k <= capacity <= int(table.shape[0]):
                raise ValueError(
                    f"dataset capacity {name}={capacity} must lie in [large_k={large_k}, "
                    f"post table rows={int(table.shape[0])}]"
                )
    dataset_types_rows = int(post_embeddings.dataset_types.shape[0])
    if int(table.shape[0]) != dataset_types_rows:
        raise ValueError(
            f"post table has {int(table.shape[0])} rows (padded to training_ep="
            f"{int(getattr(runner, 'training_ep', 0) or 0)}) but dataset_types has "
            f"{dataset_types_rows}; the retrieval forward needs them to agree, so max_posts "
            "must be a multiple of training_ep"
        )
    side_table = getattr(post_embeddings, "mol_side_table", None)
    mol_side_table_shape = None
    if side_table is not None:
        mol_side_table_shape = (int(side_table.x.shape[0]), int(side_table.x.shape[1]))
        if mol_side_table_shape[0] != int(table.shape[0]):
            raise ValueError(
                f"MoL side table has {mol_side_table_shape[0]} rows but the post table has "
                f"{int(table.shape[0])}; the serving forward slices both per shard"
            )
    return RetrievalExport(
        large_k=large_k,
        target_dataset_types=tuple((ds.name, int(ds.value)) for ds in datasets),
        use_async_topk=bool(runner.enable_async_topk),
        use_radix_select_topk=bool(runner.enable_radix_select_topk),
        post_table_shape=(int(table.shape[0]), int(table.shape[1])),
        post_table_dtype=_dtype_name(table.dtype),
        dataset_types_shape=tuple(int(d) for d in post_embeddings.dataset_types.shape),
        dataset_types_dtype=_dtype_name(post_embeddings.dataset_types.dtype),
        int8_post_table=bool(runner.enable_int8_post_table),
        dataset_capacities=capacities,
        rows_from=_split_rows_from(runner, datasets) if split_home else None,
        max_age_seconds=_split_max_age(runner, datasets) if split_home else None,
        optional_targets=_split_optional(runner, datasets) if split_home else None,
        topic_filter=topic_filter,
        mol_side_table_shape=mol_side_table_shape,
    )


def _split_rows_from(runner: Any, datasets: list[Any]) -> tuple[tuple[int, ...], ...] | None:
    if not any(ds.name == "HOME" for ds in datasets):
        return None
    from xrex.data.retrieval_dataset import RetrievalDataset

    home_rows = (
        RetrievalDataset.HOME_COLD.value,
        RetrievalDataset.HOME_HOT.value,
        RetrievalDataset.HOME.value,
    )
    return tuple(home_rows if ds.name == "HOME" else (int(ds.value),) for ds in datasets)


def _split_max_age(runner: Any, datasets: list[Any]) -> tuple[float | None, ...] | None:
    age = float(getattr(runner, "cold_start_max_age_seconds", 0.0) or 0.0)
    if age <= 0:
        return None
    return tuple(age if ds.name == "HOME_COLD" else None for ds in datasets)


def _split_optional(runner: Any, datasets: list[Any]) -> tuple[bool, ...] | None:
    launched = {ds.name for ds in runner.retrieval_dataset_types}
    flags = tuple(ds.name == "HOME_COLD" and ds.name not in launched for ds in datasets)
    return flags if any(flags) else None


def build_bundle(
    trainer: RecsysTrainer,
    *,
    mesh_devices: int = 1,
    dataset_capacities: Mapping[str, int] | None = None,
) -> list[BundleFile]:
    import flatbuffers
    from jax import export as jax_export

    from xrex.models.recsys_model import RecsysAggregatedModelConfig
    from xrex.models.recsys_two_tower_model import RecsysTwoTowerModelConfig

    model_config = trainer.model_config
    two_tower = isinstance(model_config, RecsysTwoTowerModelConfig)
    if not two_tower and not isinstance(model_config, RecsysAggregatedModelConfig):
        raise NotImplementedError(
            "StableHLO bundle export supports ranking (RecsysAggregatedModelConfig) and "
            f"retrieval (RecsysTwoTowerModelConfig) only, got {type(model_config).__name__}"
        )
    retrieval = _retrieval_export(trainer, dataset_capacities) if two_tower else None
    if mesh_devices < 1:
        raise ValueError(f"mesh_devices must be >= 1, got {mesh_devices}")
    if not two_tower and mesh_devices != 1:
        raise NotImplementedError(
            "ranking programs are single-device (one executable per GPU); "
            f"mesh_devices={mesh_devices} is only for retrieval bundles"
        )
    if two_tower and trainer.using_seqpack:
        raise NotImplementedError(
            "the two-tower export supports dense batches only (use_seqpack=False): a packed "
            "batch is laid out per device by the packer, which the SPMD retrieval program "
            "does not model"
        )
    if not trainer.checkpoint_config.copy_port:
        raise ValueError("export_stablehlo_bundle requires checkpoint_config.copy_port")

    buckets = sorted(
        {int(b) for b in str(trainer.export_bundle_bs_per_device).split(",") if b.strip()}
    )
    if not buckets or buckets[0] < 1:
        raise ValueError(
            f"invalid export_bundle_bs_per_device: {trainer.export_bundle_bs_per_device!r}"
        )

    history_seq_len = trainer.export_bundle_history_seq_len or trainer.dataset.history_seq_len
    candidate_seq_len = trainer.export_bundle_candidate_seq_len or trainer.dataset.candidate_seq_len

    if trainer.using_seqpack:
        block = int(trainer._seqpack_block_size)
        prefix = int(model_config.num_user_prefix_tokens)
        if two_tower:
            total = prefix + history_seq_len
            terms = f"{prefix} + {history_seq_len}"
            what = "num_user_prefix_tokens + history_seq_len"
        else:
            total = prefix + candidate_seq_len + history_seq_len
            terms = f"{prefix} + {candidate_seq_len} + {history_seq_len}"
            what = "num_user_prefix_tokens + candidate_seq_len + history_seq_len"
        if total % block:
            raise ValueError(
                f"seqpack export needs ({what}) to be a multiple of the attention block size "
                f"{block}, got {terms} = {total}; "
                "set export_bundle_candidate_seq_len / export_bundle_history_seq_len "
                "to block-aligned serving lengths"
            )

    ensure_parameter_serialization_registered()
    if trainer.using_seqpack:
        ensure_packing_layout_serialization_registered()
    export_cfg = _make_export_config(
        trainer, history_seq_len, candidate_seq_len, two_tower=two_tower, mesh_devices=mesh_devices
    )

    mesh = _export_mesh(export_cfg, mesh_devices)
    device = mesh.devices.flat[0]
    mesh_export = two_tower
    use_shardy = False
    data_axis: tuple[str, ...] = ()
    if mesh_export:
        use_shardy = bool(getattr(jax.config, "jax_use_shardy_partitioner"))
        assert retrieval is not None
        table_rows = retrieval.post_table_shape[0]
        if table_rows % mesh_devices:
            raise ValueError(
                f"post table rows {table_rows} are not divisible by the {mesh_devices}-device "
                "mesh; the runner shards the table evenly over the devices"
            )
        data_axis = ("stage", *model_config.model_config.data_axis)
        logger.info(
            "Exporting retrieval programs for a %d-device mesh %s (shardy=%s)",
            mesh_devices,
            dict(zip(mesh.axis_names, mesh.devices.shape)),
            use_shardy,
        )

    packed = bool(trainer.using_seqpack)
    dense_slices = (
        None if packed else _embedding_slices(export_cfg, history_seq_len, candidate_seq_len)
    )

    params_avals = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), trainer.state_shape.params
    )
    rng_aval = jax.ShapeDtypeStruct((2,), np.uint32)

    output_names = (
        ("log_probs", "cont_preds", "has_nan") if retrieval is None else retrieval.output_names
    )

    files: list[BundleFile] = []
    programs: dict[str, Any] = {}
    decode_programs: dict[str, Any] = {}
    all_custom_call_targets: set[str] = set()

    compact_candidate_slots = packed and bool(getattr(trainer, "compact_candidate_slots", False))
    serve_page_decoder = bool(getattr(trainer, "serve_page_decoder", False))
    if serve_page_decoder and retrieval is not None:
        raise NotImplementedError("page decode programs are exported for ranking bundles")
    if serve_page_decoder:
        ensure_page_decode_serialization_registered()
    rungs = [
        (bs_per_device, packed_seq_len, decode)
        for bs_per_device in buckets
        for packed_seq_len in _packed_seq_len_rungs(trainer, bs_per_device * mesh_devices, packed)
        for decode in ((False, True) if serve_page_decoder else (False,))
    ]
    for bs_per_device, packed_seq_len, decode in rungs:
        start = time.perf_counter()
        bs = bs_per_device * mesh_devices
        batch_template = _batch_template(
            export_cfg, bs, packed=packed, packed_seq_len=packed_seq_len
        )
        batch_avals = _batch_avals(
            export_cfg, bs, packed=packed, template=batch_template, two_tower=two_tower
        )
        if packed:
            embedding_slices, packed_geometry = _packed_embedding_slices(export_cfg, batch_avals)
            merged_batch = packed_geometry.merged_batch
        else:
            assert dense_slices is not None
            embedding_slices, packed_geometry = dense_slices, None
            merged_batch = bs
        merged_aval = jax.ShapeDtypeStruct(
            (merged_batch, embedding_slices.user_ip_end, model_config.emb_table_width),
            model_config.embedding_dtype,
        )
        retrieval_inputs = retrieval.program_inputs(bs) if retrieval is not None else None
        page_decode_avals = _page_decode_avals(bs, candidate_seq_len) if decode else None
        if retrieval is None:
            forward_fn = _make_forward_fn(export_cfg, embedding_slices, mesh, packed_geometry)
            args = (
                params_avals,
                rng_aval,
                batch_avals,
                merged_aval,
                *([page_decode_avals] if decode else []),
            )
        else:
            forward_fn = _make_retrieval_forward_fn(
                export_cfg, embedding_slices, mesh, retrieval, packed_geometry
            )
            assert retrieval_inputs is not None
            args = (
                params_avals,
                rng_aval,
                batch_avals,
                merged_aval,
                *(aval for _, _, aval, _ in retrieval_inputs),
            )

        with mesh:
            if mesh_export:
                assert retrieval is not None
                in_shardings, out_shardings = _retrieval_shardings(
                    mesh, data_axis, bs, params_avals, batch_avals, retrieval
                )
                jitted = jax.jit(
                    forward_fn.apply, in_shardings=in_shardings, out_shardings=out_shardings
                )
            else:
                jitted = jax.jit(forward_fn.apply)
            lowered_text = jitted.lower(*args).as_text(dialect="stablehlo")
            custom_call_targets = _scan_custom_call_targets(lowered_text)
            exported = jax_export.export(
                jitted,
                platforms=("cuda",),
                disabled_checks=[
                    jax_export.DisabledSafetyCheck.custom_call(t) for t in custom_call_targets
                ],
            )(*args)

        if mesh_export:
            if int(exported.nr_devices) != mesh_devices:
                raise AssertionError(
                    f"bs={bs}: exported for {exported.nr_devices} devices, mesh has {mesh_devices}"
                )
            if ("sdy.mesh" in lowered_text) != use_shardy:
                raise AssertionError(
                    f"bs={bs}: module Shardy attributes do not match "
                    f"jax_use_shardy_partitioner={use_shardy}"
                )

        spec = _input_spec(
            params_avals,
            rng_aval,
            batch_avals,
            merged_aval,
            batch_template,
            retrieval_inputs,
            page_decode_avals,
        )
        if len(spec) != len(exported.in_avals):
            raise AssertionError(
                f"input spec mismatch for bs={bs}: {len(spec)} != {len(exported.in_avals)}"
            )
        for entry, aval in zip(spec, exported.in_avals):
            expected = _aval_entry(aval)
            if entry["shape"] != expected["shape"] or entry["dtype"] != expected["dtype"]:
                raise AssertionError(
                    f"input spec mismatch for bs={bs} {entry['key']}: {entry} vs {expected}"
                )
        names = (*output_names, *DECODE_OUTPUT_NAMES) if decode else output_names
        if len(exported.out_avals) != len(names):
            raise AssertionError(
                f"output count mismatch for bs={bs}: {len(exported.out_avals)} != "
                f"{len(names)} {names}"
            )
        outputs = [
            {"name": name, **_aval_entry(aval)}
            for name, aval in zip(names, exported.out_avals, strict=True)
        ]
        if mesh_export:
            kept = {int(i) for i in exported.module_kept_var_idx}
            for i, (entry, aval, hlo) in enumerate(
                zip(spec, exported.in_avals, exported.in_shardings_hlo, strict=True)
            ):
                if i not in kept:
                    entry["sharding"] = dict(SHARDING_REPLICATED)
                    continue
                entry["sharding"] = _sharding_entry(
                    hlo, len(aval.shape), mesh_devices, f"bs={bs} input {entry['key']}"
                )
                replicated_kinds = (
                    "weight",
                    "rng",
                    "dataset_types",
                    "dataset_ranges",
                    "topic_query",
                )
                expected = (
                    SHARDING_REPLICATED
                    if entry["kind"] in replicated_kinds
                    else {"kind": "tiled", "dim": 0}
                )
                if mesh_devices > 1 and entry["sharding"] != expected:
                    raise AssertionError(
                        f"bs={bs} input {entry['key']}: sharding {entry['sharding']}, "
                        f"expected {expected}"
                    )
            for entry, aval, hlo in zip(
                outputs, exported.out_avals, exported.out_shardings_hlo, strict=True
            ):
                entry["sharding"] = _sharding_entry(
                    hlo, len(aval.shape), mesh_devices, f"bs={bs} output {entry['name']}"
                )

        program_key = str(bs) if packed_seq_len is None else f"{bs}_len{packed_seq_len}"
        file_stem = f"forward_bs{program_key}" + ("_decode" if decode else "")
        mlir_name = f"{BUNDLE_DIR}/{file_stem}.mlirbc"
        jax_export_name = f"{BUNDLE_DIR}/{file_stem}.jax_export"
        mlir_bytes = bytes(exported.mlir_module_serialized)
        jax_export_bytes = bytes(exported.serialize())
        files.append(BundleFile(mlir_name, mlir_bytes))
        files.append(BundleFile(jax_export_name, jax_export_bytes))
        all_custom_call_targets.update(custom_call_targets)
        row_len = int(batch_template["packing_layout"].segment_ids.shape[1]) if packed else None
        if packed_seq_len is not None and row_len != packed_seq_len:
            raise AssertionError(
                f"bs={bs}: template row {row_len} != packed_seq_len {packed_seq_len}"
            )

        (decode_programs if decode else programs)[program_key] = {
            "batch_size": bs,
            "mlir_module": {"file": mlir_name, "adler32": zlib.adler32(mlir_bytes)},
            "jax_export": {"file": jax_export_name, "adler32": zlib.adler32(jax_export_bytes)},
            "calling_convention_version": int(exported.calling_convention_version),
            "module_kept_var_idx": [int(i) for i in exported.module_kept_var_idx],
            "custom_call_targets": custom_call_targets,
            "input_spec": spec,
            "output_spec": outputs,
            "seqpack": (
                {
                    **packed_geometry._asdict(),
                    "merged_slices": embedding_slices._asdict(),
                    "packed_seq_len": row_len,
                    "compact_candidate_slots": compact_candidate_slots,
                }
                if packed_geometry is not None
                else None
            ),
        }
        logger.info(
            "Exported StableHLO %s forward bs=%d (%d/device, packed row %s) in %.1fs "
            "(%d inputs, %d kept, custom_calls=%s)",
            "retrieval" if retrieval is not None else "ranking decode" if decode else "ranking",
            bs,
            bs_per_device,
            row_len,
            time.perf_counter() - start,
            len(spec),
            len(exported.module_kept_var_idx),
            custom_call_targets,
        )

    manifest = _build_manifest(
        trainer,
        export_cfg,
        device,
        programs,
        dense_slices,
        history_seq_len,
        candidate_seq_len,
        sorted(all_custom_call_targets),
        packed=packed,
        retrieval=retrieval,
        mesh=_mesh_block(mesh, use_shardy) if mesh_export else None,
        decode_programs=decode_programs,
    )
    files.insert(0, BundleFile(MANIFEST_NAME, json.dumps(manifest, indent=2).encode()))
    return files


def _leaf_width(tree: Any, key: str, axis: int) -> int:
    leaf = tree.get(key)
    if leaf is None or len(leaf.shape) <= axis:
        return 0
    return int(leaf.shape[axis])


def _prep_spec(
    export_cfg: Any,
    history_seq_len: int,
    candidate_seq_len: int,
    *,
    packed: bool,
    two_tower: bool = False,
) -> dict[str, Any]:
    dataset = export_cfg.dataset
    model_config = export_cfg.model_config
    features = _token_feature_config(model_config, two_tower=two_tower)
    ht = dataset.hash_table
    hk = ht.hash_keys
    batch = _batch_avals(export_cfg, 1, packed=False, two_tower=two_tower)
    hist = batch["history_seq"]
    cand = batch["candidate_seq"]

    if two_tower:
        template_width = _leaf_width(hist, "continuous_actions", 2)
        if template_width != int(features.num_continuous_actions):
            raise ValueError(
                f"two-tower num_continuous_actions mismatch: user tower "
                f"{features.num_continuous_actions} vs batch template {template_width}"
            )
        transformer_candidate_seq_len = 0
    else:
        transformer_candidate_seq_len = (
            (0 if cand.get("post_ids") is not None else candidate_seq_len) if packed else 0
        )

    return {
        "user_id_table_size": int(ht.user_id_table_size),
        "user_hash_scales": [int(x) for x in hk.user_hash_scales],
        "user_biases": [int(x) for x in hk.user_biases],
        "user_modulus": int(hk.user_modulus),
        "item_id_table_size": int(ht.item_id_table_size),
        "item_hash_vocab_size": int(getattr(hk, "item_hash_vocab_size", 0) or 0),
        "item_hash_scales": [int(x) for x in hk.item_hash_scales],
        "item_biases": [int(x) for x in hk.item_biases],
        "item_modulus": int(hk.item_modulus),
        "author_id_table_size": int(ht.author_id_table_size),
        "author_hash_scales": [int(x) for x in hk.author_hash_scales],
        "author_biases": [int(x) for x in hk.author_biases],
        "author_modulus": int(hk.author_modulus),
        "ip_id_table_size": int(ht.ip_id_table_size),
        "ip_hash_scales": [int(x) for x in hk.ip_hash_scales],
        "ip_biases": [int(x) for x in hk.ip_biases],
        "ip_modulus": int(hk.ip_modulus),
        "output_vocab_size": int(dataset.output_vocab_size),
        "num_continuous_actions": int(features.num_continuous_actions),
        "search_query_embedding_dim": _leaf_width(cand, "search_query_embeddings", 2),
        "num_user_categorical_features": _leaf_width(batch, "user_categorical_features", 1),
        "num_user_bool_features": _leaf_width(batch, "user_bool_features", 1),
        "num_user_float_features": _leaf_width(batch, "user_float_features", 1),
        "num_user_int64_features": _leaf_width(batch, "user_int64_features", 1),
        "num_user_installed_apps": _leaf_width(batch, "user_installed_apps_multihot", 1),
        "num_post_categorical_features": _leaf_width(hist, "categorical_features", 2),
        "num_post_bool_features": _leaf_width(hist, "bool_features", 2),
        "num_post_float_features": _leaf_width(hist, "float_features", 2),
        "num_post_int64_features": _leaf_width(hist, "int64_features", 2),
        "enable_stale_post": bool(
            getattr(getattr(features, "feature_prep", None), "enable_stale_post", False)
        ),
        "enable_stale_post_30d": bool(
            getattr(getattr(features, "feature_prep", None), "enable_stale_post_30d", False)
        ),
        "history_seq_len": history_seq_len,
        "candidate_seq_len": candidate_seq_len,
        "sid_num_levels": _leaf_width(hist, "post_sids", 2),
        "multimodal_embedding_dim": _leaf_width(cand, "embedding", 2),
        "num_categorical_features": 0,
        "use_ip": bool(model_config.use_ip_address),
        "use_seqpack": packed,
        "seqpack_block_size": int(export_cfg._seqpack_block_size) if packed else 0,
        "num_user_prefix_tokens": (int(model_config.num_user_prefix_tokens) if packed else 0),
        "transformer_candidate_seq_len": transformer_candidate_seq_len,
    }


def _emb_table_geometry(trainer: RecsysTrainer) -> tuple[int, int]:
    emb_table = getattr(trainer.state_shape, "emb_table", None)
    if emb_table is not None:
        return int(emb_table.x.shape[0]), int(trainer.mesh.shape["expert"])
    ep = max(int(getattr(trainer, "training_ep", 0) or 0), 1)
    return int(trainer.dataset.input_vocab_size), ep


def _build_manifest(
    trainer: RecsysTrainer,
    export_cfg: Any,
    device: Any,
    programs: dict[str, Any],
    dense_slices: EmbeddingSlices | None,
    history_seq_len: int,
    candidate_seq_len: int,
    custom_call_targets: list[str],
    *,
    packed: bool,
    retrieval: RetrievalExport | None = None,
    mesh: dict[str, Any] | None = None,
    decode_programs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import jaxlib

    model_config = export_cfg.model_config
    two_tower = retrieval is not None
    features = _token_feature_config(model_config, two_tower=two_tower)
    dataset = export_cfg.dataset

    emb_rows, ep = _emb_table_geometry(trainer)
    emb_rows_padded = -(-emb_rows // ep) * ep

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": KIND_RETRIEVAL if two_tower else KIND_RANKING,
        "name": trainer.name,
        "model_config_class": type(trainer.model_config).__name__,
        "created_timestamp": time.time(),
        "model_sequence_len": int(model_config.model_config.sequence_len),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platforms": ["cuda"],
        "device_kind": str(device.device_kind),
        "compute_capability": str(getattr(device, "compute_capability", "")),
        "custom_call_targets": custom_call_targets,
        "history_seq_len": history_seq_len,
        "candidate_seq_len": candidate_seq_len,
        "output_vocab_size": int(dataset.output_vocab_size),
        "num_continuous_actions": int(features.num_continuous_actions),
        "use_seqpack": packed,
        "prep_spec": _prep_spec(
            export_cfg, history_seq_len, candidate_seq_len, packed=packed, two_tower=two_tower
        ),
        "programs": programs,
        "embedding": {
            "table_key": "emb_table",
            "rows": emb_rows,
            "rows_padded": emb_rows_padded,
            "num_shards": ep,
            "width": int(model_config.emb_table_width),
            "dtype": _dtype_name(model_config.embedding_dtype),
            "merged_slices": dense_slices._asdict() if dense_slices is not None else None,
            "hash_table": dataset.hash_table.to_dict(),
        },
    }
    if decode_programs:
        manifest["decode_programs"] = decode_programs
    if retrieval is not None:
        manifest["retrieval"] = retrieval.manifest_block()
        if mesh is None:
            raise AssertionError("retrieval manifests are mesh exports")
        manifest["mesh"] = mesh
    elif mesh is not None:
        raise AssertionError("ranking manifests carry no mesh block")
    return manifest
