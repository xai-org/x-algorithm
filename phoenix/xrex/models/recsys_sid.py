# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import functools
import math
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from numpy import typing as npt

from xrex.models.layers import get_parameter
from xrex.models.sharding_context import axis_group_size
from xrex.utils.sharding import in_manual_shard_map, shard_map_unless_manual

_DECODER_PARTS = ("hi", "mid", "lo")


@functools.lru_cache(maxsize=2)
def _load_sid_decoder(
    path: str,
) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    import safetensors.numpy

    tensors = safetensors.numpy.load_file(path)
    return tensors["stages"], (
        tensors["dec_w0"],
        tensors["dec_b0"],
        tensors["dec_w1"],
        tensors["dec_b1"],
    )


def sid_prefix2_lookup(
    sids: jnp.ndarray,
    sid_codebook_size: int,
    embed_dim: int,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    name_prefix: str,
) -> jnp.ndarray:
    sids = sids.astype(jnp.int32)
    first, second = sids[..., 0], sids[..., 1]
    prefix2 = jnp.where((first > 0) & (second > 0), (first - 1) * sid_codebook_size + second, 0)
    table = get_parameter(
        f"{name_prefix}_sid_prefix2",
        [sid_codebook_size * sid_codebook_size + 1, embed_dim],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(embed_dim),
    )
    table = table.at[0].set(0.0)
    return jnp.take(table, prefix2, axis=0)


def sid_prefix3_lookup(
    sids: jnp.ndarray,
    sid_codebook_size: int,
    embed_dim: int,
    rows: int,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    name_prefix: str,
) -> jnp.ndarray:
    codes = sids[..., :3].astype(jnp.uint32)
    exact = ((codes[..., 0] - 1) * sid_codebook_size + (codes[..., 1] - 1)) * sid_codebook_size + (
        codes[..., 2] - 1
    )
    hashed = (exact * jnp.uint32(2654435761)) % jnp.uint32(rows) + 1
    prefix3 = jnp.where((codes > 0).all(axis=-1), hashed, 0).astype(jnp.int32)
    table = get_parameter(
        f"{name_prefix}_sid_prefix3",
        [rows + 1, embed_dim],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(embed_dim),
    )
    table = table.at[0].set(0.0)
    return jnp.take(table, prefix3, axis=0)


def _run_sid_decoder(
    sids: jnp.ndarray,
    stages: jnp.ndarray,
    w0: jnp.ndarray,
    b0: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
) -> jnp.ndarray:
    num_levels, codebook_size, _ = stages.shape
    codes = jnp.clip(sids.astype(jnp.int32) - 1, 0, codebook_size - 1)
    quant = stages[jnp.arange(num_levels), codes].sum(-2)
    recon = jax.nn.relu(quant @ w0 + b0) @ w1 + b1
    recon *= jax.lax.rsqrt((recon**2).sum(-1, keepdims=True) + 1e-12)
    return jnp.where((sids[..., 0] == 0)[..., None], 0.0, recon)


def _decode_distinct_sid_tuples(
    sids: jnp.ndarray,
    run_decoder: Callable[[jnp.ndarray], jnp.ndarray],
    recon_dim: int,
    rows_per_pass: int,
) -> jnp.ndarray:
    flat = sids.reshape(-1, sids.shape[-1])
    key = jnp.zeros(len(flat), jnp.uint32)
    for level in range(flat.shape[1]):
        key = key * jnp.uint32(0x9E3779B1) + flat[:, level].astype(jnp.uint32)
    order = jnp.argsort(key, stable=True)
    sorted_sids = flat[order]
    starts = jnp.concatenate([jnp.ones(1, bool), (sorted_sids[1:] != sorted_sids[:-1]).any(-1)])
    group = jnp.cumsum(starts) - 1
    padded = math.ceil(len(flat) / rows_per_pass) * rows_per_pass
    distinct = sorted_sids[jnp.flatnonzero(starts, size=padded)]

    def decode_one_pass(i: jax.Array, recon: jax.Array) -> jax.Array:
        rows = jax.lax.dynamic_slice_in_dim(distinct, i * rows_per_pass, rows_per_pass)
        return jax.lax.dynamic_update_slice_in_dim(recon, run_decoder(rows), i * rows_per_pass, 0)

    recon = jax.lax.fori_loop(
        0, group[-1] // rows_per_pass + 1, decode_one_pass, jnp.zeros((padded, recon_dim))
    )
    return recon[group][jnp.argsort(order)].reshape(*sids.shape[:-1], recon_dim)


def _frozen_decoder_params(
    decoder: tuple[npt.NDArray, ...], name_prefix: str
) -> tuple[jax.Array, ...]:
    def split(x: npt.NDArray) -> list[npt.NDArray]:
        parts, rest = [], x.astype(np.float32)
        for _ in _DECODER_PARTS:
            parts.append(rest.astype(jnp.bfloat16))
            rest = rest - parts[-1].astype(np.float32)
        assert not rest.any(), "a decoder tensor is not exactly three bf16 parts"
        return parts

    tensors = []
    for key, x in zip(("stages", "w0", "b0", "w1", "b1"), decoder):
        total = None
        for i, suffix in enumerate(_DECODER_PARTS):
            part = get_parameter(
                f"{name_prefix}_sid_decoder_table_{key}_{suffix}",
                list(x.shape),
                dtype=jnp.float32,
                init=lambda shape, dtype, x=x, i=i: jnp.asarray(split(x)[i], dtype),
                pspec=P(*[None] * x.ndim),
                lr_multiplier=0.0,
                dtype_unwrap=jnp.float32,
            )
            total = part if total is None else total + part
        tensors.append(jax.lax.stop_gradient(total))
    return tuple(tensors)


def reconstruct_entity_sid(
    sids: jnp.ndarray,
    target_dim: int,
    decoder_path: str,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    name_prefix: str,
    mesh: Mesh | None = None,
    data_axis: str | tuple[str, ...] | None = None,
    rows_per_decoder_pass: int | None = None,
    decoder_params: bool = False,
    remat: bool = False,
) -> jnp.ndarray:
    if remat:
        return hk.remat(
            lambda s: reconstruct_entity_sid(
                s,
                target_dim,
                decoder_path,
                lr_multiplier_func,
                embed_init_scale,
                fprop_dtype,
                name_prefix,
                mesh=mesh,
                data_axis=data_axis,
                rows_per_decoder_pass=rows_per_decoder_pass,
                decoder_params=decoder_params,
            )
        )(sids)
    assert mesh is None or (data_axis is not None and rows_per_decoder_pass is not None), (
        "a mesh needs the data axis sids are sharded on and rows_per_decoder_pass"
    )
    stages_np, (w0_np, b0_np, w1_np, b1_np) = _load_sid_decoder(decoder_path)
    num_levels = stages_np.shape[0]
    assert sids.shape[-1] == num_levels, (
        f"post_sids have {sids.shape[-1]} levels but the decoder at {decoder_path!r} "
        f"expects {num_levels}; set sid_num_levels to match the artifact."
    )
    if decoder_params:
        stages, w0, b0, w1, b1 = _frozen_decoder_params(
            (stages_np, w0_np, b0_np, w1_np, b1_np), name_prefix
        )
    else:
        stages, (w0, b0, w1, b1) = jax.tree.map(
            jnp.asarray, (stages_np, (w0_np, b0_np, w1_np, b1_np))
        )
    run_decoder = functools.partial(_run_sid_decoder, stages=stages, w0=w0, b0=b0, w1=w1, b1=b1)
    recon_dim = w1.shape[1]

    posts_per_device = math.prod(sids.shape[:-1])
    if mesh is not None and not in_manual_shard_map():
        posts_per_device //= axis_group_size(data_axis, mesh)

    if mesh is None or posts_per_device < rows_per_decoder_pass:
        recon = run_decoder(sids)
    else:
        recon = shard_map_unless_manual(
            functools.partial(
                _decode_distinct_sid_tuples,
                run_decoder=run_decoder,
                recon_dim=recon_dim,
                rows_per_pass=rows_per_decoder_pass,
            ),
            mesh=mesh,
            in_specs=P(data_axis),
            out_specs=P(data_axis),
            check_vma=False,
        )(sids)
    recon = jax.lax.stop_gradient(recon)

    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    proj = get_parameter(
        f"{name_prefix}_sid_recon_proj",
        [recon_dim, target_dim],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(recon_dim),
    )
    return jnp.dot(recon, proj).astype(fprop_dtype)
