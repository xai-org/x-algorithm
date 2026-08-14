# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import jax
import jax.numpy as jnp


def gather_selected_validity(eligibility: jax.Array, selected_indices: jax.Array) -> jax.Array:
    return jnp.take_along_axis(eligibility, selected_indices, axis=1)


try:
    from xrex.cuda.top_k_by_key.src import top_k_by_key_api
except ImportError:
    top_k_by_key_api = None
else:
    jax.ffi.register_ffi_target(
        "xrex_top_k_by_key", fn=top_k_by_key_api.top_k_by_key(), platform="CUDA"
    )

try:
    from xrex.cuda.top_k_by_key.src import top_k_by_key_async_api
except ImportError:
    top_k_by_key_async_api = None
else:
    jax.ffi.register_ffi_target(
        "xrex_top_k_by_key_async",
        fn=top_k_by_key_async_api.top_k_by_key_async(),
        platform="CUDA",
    )

try:
    from xrex.cuda.top_k_by_key.src import top_k_by_key_radix_select_api
except ImportError:
    top_k_by_key_radix_select_api = None
else:
    jax.ffi.register_ffi_target(
        "xrex_top_k_by_key_radix_select",
        fn=top_k_by_key_radix_select_api.top_k_by_key_radix_select(),
        platform="CUDA",
    )


def top_k_by_key(
    keys: jax.Array,
    k: int,
    heuristic_pivot_ratio: float,
    use_async: bool = False,
    use_radix_select: bool = False,
):
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Only bfloat16 is supported for keys.")
    if keys.ndim > 2:
        raise ValueError("keys must be 1D or 2D.")

    n = keys.shape[-1]
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")

    if use_radix_select:
        api = top_k_by_key_radix_select_api
    elif use_async:
        api = top_k_by_key_async_api
    else:
        api = top_k_by_key_api
    if api is None or jax.default_backend() != "gpu":
        sorted_keys, sorted_indices = jax.lax.top_k(keys, k)
        return sorted_keys, sorted_indices.astype(jnp.int32)

    out_shape = (k,) if keys.ndim == 1 else (keys.shape[0], k)
    out_types = [
        jax.ShapeDtypeStruct(shape=out_shape, dtype=keys.dtype),
        jax.ShapeDtypeStruct(shape=out_shape, dtype=jnp.int32),
    ]
    if use_radix_select:
        call = jax.ffi.ffi_call(
            "xrex_top_k_by_key_radix_select", out_types, vmap_method="broadcast_all"
        )
        return call(keys, k=k)
    if use_async:
        call = jax.ffi.ffi_call("xrex_top_k_by_key_async", out_types, vmap_method="broadcast_all")
        return call(keys, k=k)
    call = jax.ffi.ffi_call("xrex_top_k_by_key", out_types, vmap_method="broadcast_all")
    return call(keys, k=k, heuristic_pivot_ratio=heuristic_pivot_ratio)
