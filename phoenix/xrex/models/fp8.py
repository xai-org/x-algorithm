# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import dataclasses
import enum
import functools
import math
import re

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import AxisType
from jax.typing import DTypeLike

from xrex.cutedsl import fp8_transpose
from xrex.utils import gpu


class Fp8Gemms(enum.StrEnum):
    OFF = "off"
    BWD = "bwd"
    ALL = "all"


_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2
_FWD_PRECISION = lax.Precision.DEFAULT
_GRAD_PRECISION = lax.Precision.HIGHEST
_NN = (((1,), (0,)), ((), ()))
_NT = (((1,), (1,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))
_CUBLASLT_FP8_GEMM = 'custom_call_target="__cublas$lt$matmul$f8"'
_OPERAND_PRECISION = re.compile(r'"operand_precision":\[([^]]*)\]')


@dataclasses.dataclass(frozen=True)
class Fp8Dot:
    gemms: Fp8Gemms = Fp8Gemms.OFF
    input_is_materialized: bool = False
    transposed_input_checkpoint_name: str | None = None
    fuse_gradient_amax: bool = False

    def __call__(self, x: jax.Array, w: jax.Array) -> jax.Array:
        if self.gemms == Fp8Gemms.OFF:
            return jnp.dot(x, w)
        assert x.dtype == w.dtype, (x.dtype, w.dtype)
        use_transpose_kernels = _use_transpose_kernels(self.gemms, x.shape, w.shape[1], x.dtype)
        assert not (
            use_transpose_kernels
            and jax.device_count() > 1
            and (jax.typeof(x).vma or jax.typeof(w).vma)
        ), "Fp8Dot's transpose kernels need check_vma=False in their shard_map"
        return _fp8_dot(x, w, self, use_transpose_kernels)


def count_compiled_fp8_gemms(hlo: str) -> tuple[int, int]:
    precisions = collections.Counter(
        match[1] if (match := _OPERAND_PRECISION.search(line)) else None
        for line in hlo.splitlines()
        if _CUBLASLT_FP8_GEMM in line
    )
    forward, gradient = (
        precisions.pop(f'"{precision.name}","{precision.name}"', 0)
        for precision in (_FWD_PRECISION, _GRAD_PRECISION)
    )
    assert not precisions, f"FP8 GEMMs of other operand precisions: {precisions}"
    return forward, gradient


def check_compiled_fp8_gemms(program_name: str, hlo: str, gemms: Fp8Gemms) -> None:
    forward, gradient = count_compiled_fp8_gemms(hlo)
    min_forward = gradient // 2 if gemms == Fp8Gemms.ALL else 0
    if not gradient or forward < min_forward:
        raise ValueError(
            f"{program_name} compiled {forward} forward and {gradient} gradient FP8 GEMMs "
            f"(at least {min_forward} and 1 expected): check its XLA flags"
        )


def _use_transpose_kernels(
    gemms: Fp8Gemms, x_shape: tuple[int, ...], output_size: int, dtype: DTypeLike
) -> bool:
    num_rows = math.prod(x_shape[:-1])
    x_2d_shape, g_2d_shape = (num_rows, x_shape[-1]), (num_rows, output_size)
    mesh = jax.sharding.get_abstract_mesh()
    per_device_operands = jax.device_count() == 1 or (
        not mesh.empty
        and all(
            kind == AxisType.Manual or size == 1
            for kind, size in zip(mesh.axis_types, mesh.axis_sizes, strict=True)
        )
    )
    return (
        gemms == Fp8Gemms.ALL
        and gpu.cuda_compute_capability(jax.devices()[0]) == (9, 0)
        and dtype == jnp.bfloat16
        and fp8_transpose.transpose_supported(x_2d_shape)
        and fp8_transpose.quantize_transposed_supported(x_2d_shape)
        and fp8_transpose.quantize_transposed_supported(g_2d_shape)
        and per_device_operands
    )


def _scale(x: jax.Array, dtype: DTypeLike) -> jax.Array:
    amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
    largest_mantissa, largest_exponent = math.frexp(float(jnp.finfo(dtype).max))
    mantissa, exponent = jnp.frexp(amax)
    scale_exponent = exponent - largest_exponent + (mantissa > largest_mantissa).astype(jnp.int32)
    finfo = jnp.finfo(jnp.float32)
    scale_exponent = jnp.clip(scale_exponent, finfo.minexp, finfo.maxexp - 1)
    scale = jnp.ldexp(jnp.float32(1), scale_exponent)
    return jnp.where(amax == 0, 1.0, jnp.where(jnp.isfinite(amax), scale, jnp.nan))


def _quantize(x: jax.Array, scale: jax.Array, dtype: DTypeLike) -> jax.Array:
    limit = float(jnp.finfo(dtype).max)
    return jnp.clip(x.astype(jnp.float32) / scale, -limit, limit).astype(dtype)


def _dequantize(q: jax.Array, scale: jax.Array, dtype: DTypeLike) -> jax.Array:
    return q.astype(dtype) * scale.astype(dtype)


def _quantize_dequantize(x: jax.Array, scale: jax.Array, dtype: DTypeLike) -> jax.Array:
    return _dequantize(_quantize(x, scale, dtype), scale, x.dtype)


def _matmul(
    a: jax.Array, b: jax.Array, dims: lax.DotDimensionNumbers, precision: lax.Precision
) -> jax.Array:
    return lax.dot_general(a, b, dims, precision=precision, preferred_element_type=a.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _fp8_dot(x: jax.Array, w: jax.Array, dot: Fp8Dot, use_transpose_kernels: bool) -> jax.Array:
    return jnp.dot(x, w)


def _fp8_dot_fwd(x, w, dot: Fp8Dot, use_transpose_kernels: bool):
    w_scale = checkpoint_name(_scale(w, _E4M3), "scalar_stats")
    if dot.gemms != Fp8Gemms.ALL:
        return jnp.dot(x, w), (x, w, None, w_scale, None)
    x_2d = x.reshape(-1, x.shape[-1])
    x_scale = _scale(x_2d, _E4M3)
    x_q_t_bits = None
    if use_transpose_kernels and dot.transposed_input_checkpoint_name is not None:
        x_q, x_q_t = fp8_transpose.quantize_transposed(x_2d, 1.0 / x_scale, _E4M3)
        x_fp8 = _dequantize(x_q, x_scale, x_2d.dtype)
        x_q_t_bits = checkpoint_name(
            lax.bitcast_convert_type(x_q_t, jnp.uint8), dot.transposed_input_checkpoint_name
        )
    else:
        x_fp8 = _quantize_dequantize(x_2d, x_scale, _E4M3)
    y = _matmul(x_fp8, _quantize_dequantize(w, w_scale, _E4M3), _NN, _FWD_PRECISION)
    x_scale = checkpoint_name(x_scale, "scalar_stats")
    x_residual = None if x_q_t_bits is not None else x
    return y.reshape(*x.shape[:-1], w.shape[1]), (x_residual, w, x_scale, w_scale, x_q_t_bits)


def _fp8_dot_bwd(dot: Fp8Dot, use_transpose_kernels: bool, residuals, g):
    x, w, x_scale, w_scale, x_q_t_bits = residuals
    g_2d = g.reshape(-1, g.shape[-1])
    x_2d = None if x is None else x.reshape(-1, x.shape[-1])
    if x_scale is None:
        x_scale = _scale(x_2d, _E4M3)
    w_fp8 = _quantize_dequantize(w, w_scale, _E4M3)
    if not use_transpose_kernels:
        g_fp8 = _quantize_dequantize(g_2d, _scale(g_2d, _E5M2), _E5M2)
        dx = _matmul(g_fp8, w_fp8, _NT, _GRAD_PRECISION)
        dw = _matmul(_quantize_dequantize(x_2d, x_scale, _E4M3), g_fp8, _TN, _GRAD_PRECISION)
        return dx.reshape(*g.shape[:-1], w.shape[0]), dw

    if dot.fuse_gradient_amax:
        finfo = jnp.finfo(g_2d.dtype)
        g_2d = lax.reduce_precision(g_2d, exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
    else:
        g_2d = lax.optimization_barrier(g_2d)
    g_scale = _scale(g_2d, _E5M2)
    g_q, g_q_t = fp8_transpose.quantize_transposed(g_2d, 1.0 / g_scale, _E5M2)
    dx = _matmul(_dequantize(g_q, g_scale, g_2d.dtype), w_fp8, _NT, _GRAD_PRECISION)
    if x_q_t_bits is not None:
        x_q_t = lax.bitcast_convert_type(x_q_t_bits, _E4M3)
    elif dot.input_is_materialized:
        _, x_q_t = fp8_transpose.quantize_transposed(
            x_2d, 1.0 / x_scale, _E4M3, with_untransposed=False
        )
    else:
        x_q_t = fp8_transpose.transpose(_quantize(x_2d, x_scale, _E4M3))
    dw = _matmul(
        _dequantize(x_q_t, x_scale, g_2d.dtype),
        _dequantize(g_q_t, g_scale, g_2d.dtype),
        _NT,
        _GRAD_PRECISION,
    )
    return dx.reshape(*g.shape[:-1], w.shape[0]), dw


_fp8_dot.defvjp(_fp8_dot_fwd, _fp8_dot_bwd)
