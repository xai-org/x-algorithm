# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import math

import jax
import jax.numpy as jnp

_log = logging.getLogger(__name__)

TARGET = "xrex_mol_epilogue"
FUSED_TARGET = "xrex_mol_fused_int8"
GATE_BWD_TARGET = "xrex_mol_gate_bwd"
SUPPORTED_KL_H = ((4, 32), (8, 16), (8, 32), (16, 16))
GATE_BWD_KL_H = ((4, 32), (8, 16), (8, 32))
MAX_USERS = 512
FUSED_QUERY_ROWS = (32, 64, 96, 128)

try:
    import xrex_cuda_kernels.mol_epilogue_api as mol_epilogue_api
except ModuleNotFoundError:
    mol_epilogue_api = None
    _log.warning(
        "xrex.cuda.mol_epilogue: xrex_cuda_kernels.mol_epilogue_api not installed; the MoL "
        "serving kernel is unavailable"
    )
else:
    jax.ffi.register_ffi_target(TARGET, fn=mol_epilogue_api.mol_epilogue(), platform="CUDA")
    if hasattr(mol_epilogue_api, "mol_fused_int8"):
        jax.ffi.register_ffi_target(
            FUSED_TARGET, fn=mol_epilogue_api.mol_fused_int8(), platform="CUDA"
        )
    if hasattr(mol_epilogue_api, "mol_gate_bwd"):
        jax.ffi.register_ffi_target(
            GATE_BWD_TARGET, fn=mol_epilogue_api.mol_gate_bwd(), platform="CUDA"
        )


def available() -> bool:
    return mol_epilogue_api is not None and jax.default_backend() == "gpu"


def fused_col_offset_supported() -> bool:
    return getattr(mol_epilogue_api, "mol_fused_int8_col_offset", False) is True


def fused_query_rows(num_rows: int) -> int | None:
    return next((r for r in FUSED_QUERY_ROWS if r >= num_rows), None)


def fused_supported(kl: int, num_components: int, hidden: int, num_users: int, dim: int) -> bool:
    return (
        available()
        and hasattr(mol_epilogue_api, "mol_fused_int8")
        and kl == 4
        and num_components == 1
        and hidden == 32
        and dim % 64 == 0
        and fused_query_rows(kl * num_users) is not None
    )


def mol_epilogue(
    dots: jax.Array,
    norms: jax.Array,
    ubias_t: jax.Array,
    ibias: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
    *,
    num_users: int,
    num_components: int,
) -> jax.Array:
    if not available():
        raise RuntimeError(
            f"CUDA FFI target {TARGET!r} is unavailable (xrex_cuda_kernels.mol_epilogue_api "
            f"installed: {mol_epilogue_api is not None}, backend: {jax.default_backend()})"
        )
    kl, hidden = w1.shape
    if dots.shape != (kl * num_users, dots.shape[1]):
        raise ValueError(f"dots must be [KL*B, M] with KL={kl}, B={num_users}; got {dots.shape}")
    if (kl, hidden) not in SUPPORTED_KL_H:
        raise ValueError(f"unsupported (KL, H)=({kl}, {hidden}); compiled: {SUPPORTED_KL_H}")
    if num_users > MAX_USERS:
        raise ValueError(f"num_users {num_users} exceeds the kernel limit {MAX_USERS}")
    call = jax.ffi.ffi_call(
        TARGET,
        jax.ShapeDtypeStruct((num_users, dots.shape[1]), jnp.float32),
        vmap_method="sequential",
    )
    return call(
        dots.astype(jnp.bfloat16),
        norms.astype(jnp.float32),
        ubias_t.astype(jnp.float32),
        ibias.astype(jnp.float32),
        w1.astype(jnp.float32),
        b1.reshape(-1).astype(jnp.float32),
        w2.astype(jnp.float32),
        b2.reshape(-1).astype(jnp.float32),
        num_users=num_users,
        num_components=num_components,
    )


def mol_fused_int8(
    posts_q8: jax.Array,
    post_scales: jax.Array,
    queries_q8: jax.Array,
    query_scales: jax.Array,
    norms: jax.Array,
    ubias_t: jax.Array,
    ibias: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
    *,
    num_users: int,
    col_offset: int = 0,
) -> jax.Array:
    kl, hidden = w1.shape
    dim = queries_q8.shape[1]
    strided = col_offset != 0 or posts_q8.shape[1] != dim
    if strided and not fused_col_offset_supported():
        raise RuntimeError(
            f"{FUSED_TARGET!r} in this xrex_cuda_kernels cannot read a column offset"
        )
    if not fused_supported(kl, norms.shape[0], hidden, num_users, dim):
        raise RuntimeError(
            f"{FUSED_TARGET!r} unsupported for KL={kl}, L={norms.shape[0]}, H={hidden}, "
            f"B={num_users}, D={dim} (or xrex_cuda_kernels lacks it)"
        )
    if queries_q8.shape[0] not in FUSED_QUERY_ROWS or queries_q8.shape[0] < kl * num_users:
        raise ValueError(
            f"queries must have {FUSED_QUERY_ROWS} rows >= KL*B; got {queries_q8.shape}"
        )
    call = jax.ffi.ffi_call(
        FUSED_TARGET,
        jax.ShapeDtypeStruct((num_users, posts_q8.shape[0]), jnp.bfloat16),
        vmap_method="sequential",
    )
    return call(
        posts_q8.astype(jnp.int8),
        post_scales.astype(jnp.float32),
        queries_q8.astype(jnp.int8),
        query_scales.astype(jnp.float32),
        norms.astype(jnp.float32),
        ubias_t.astype(jnp.float32),
        ibias.astype(jnp.float32),
        w1.astype(jnp.float32),
        b1.reshape(-1).astype(jnp.float32),
        w2.astype(jnp.float32),
        b2.reshape(-1).astype(jnp.float32),
        num_users=num_users,
        **({"col_offset": col_offset} if fused_col_offset_supported() else {}),
    )


def gate_training_supported(kl: int, hidden: int, num_users: int) -> bool:
    return (
        available()
        and hasattr(mol_epilogue_api, "mol_gate_bwd")
        and (kl, hidden) in GATE_BWD_KL_H
        and num_users <= min(MAX_USERS, mol_epilogue_api.mol_gate_bwd_max_users)
    )


def _gate_forward(cos, user_bias, item_bias, w1, b1, w2, b2):
    kl, num_users, num_items = cos.shape
    return mol_epilogue(
        cos.reshape(kl * num_users, num_items),
        jnp.ones((1, num_items), jnp.float32),
        jnp.transpose(user_bias),
        jnp.transpose(item_bias),
        w1,
        b1,
        w2,
        b2,
        num_users=num_users,
        num_components=1,
    )


@jax.custom_vjp
def mol_gate_scores(cos, user_bias, item_bias, w1, b1, w2, b2):
    return _gate_forward(cos, user_bias, item_bias, w1, b1, w2, b2)


def _gate_vjp_fwd(cos, user_bias, item_bias, w1, b1, w2, b2):
    out = _gate_forward(cos, user_bias, item_bias, w1, b1, w2, b2)
    return out, (cos, user_bias, item_bias, w1, b1, w2, b2)


def _gate_vjp_bwd(res, g):
    cos, user_bias, item_bias, w1, b1, w2, b2 = res
    kl, num_users, num_items = cos.shape
    hidden = w1.shape[1]
    blocks = math.ceil(num_items / mol_epilogue_api.mol_gate_bwd_items_per_block)
    f32 = jnp.float32
    call = jax.ffi.ffi_call(
        GATE_BWD_TARGET,
        (
            jax.ShapeDtypeStruct((kl * num_users, num_items), jnp.bfloat16),
            jax.ShapeDtypeStruct((kl, num_items), f32),
            jax.ShapeDtypeStruct((blocks, kl, num_users), f32),
            jax.ShapeDtypeStruct((blocks, 2 * kl * hidden + hidden), f32),
        ),
        vmap_method="sequential",
    )
    dcos, ditem_bias, duser_bias, dw = call(
        cos.reshape(kl * num_users, num_items).astype(jnp.bfloat16),
        jnp.transpose(user_bias).astype(f32),
        jnp.transpose(item_bias).astype(f32),
        w1.astype(f32),
        b1.reshape(-1).astype(f32),
        w2.astype(f32),
        b2.reshape(-1).astype(f32),
        g.astype(f32),
        num_users=num_users,
    )
    duser_bias = jnp.sum(duser_bias, axis=0)
    dw = jnp.sum(dw, axis=0)
    return (
        dcos.reshape(cos.shape).astype(cos.dtype),
        jnp.transpose(duser_bias).astype(user_bias.dtype),
        jnp.transpose(ditem_bias).astype(item_bias.dtype),
        dw[: kl * hidden].reshape(kl, hidden).astype(w1.dtype),
        dw[kl * hidden : kl * hidden + hidden].astype(b1.dtype),
        dw[kl * hidden + hidden :].reshape(hidden, kl).astype(w2.dtype),
        jnp.sum(duser_bias, axis=1).astype(b2.dtype),
    )


mol_gate_scores.defvjp(_gate_vjp_fwd, _gate_vjp_bwd)
