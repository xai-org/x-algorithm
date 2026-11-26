# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

_ROWS = 16
_LANES_M = 4
_WARPS = 8
_WARP_SIZE = 32
_LANES_N = _WARP_SIZE // _LANES_M
_QUANTIZE_COLS = 8
_TRANSPOSE_COLS = 16
_PTX_CVT_TYPE = {"float8_e4m3fn": "e4m3x2", "float8_e5m2": "e5m2x2"}


def quantize_transposed_supported(shape: tuple[int, ...]) -> bool:
    return _supported(shape, _QUANTIZE_COLS)


def transpose_supported(shape: tuple[int, ...]) -> bool:
    return _supported(shape, _TRANSPOSE_COLS)


def _supported(shape: tuple[int, ...], cols: int) -> bool:
    return (
        len(shape) == 2
        and 0 < shape[0] * shape[1] < 2**31
        and shape[0] % (_ROWS * _LANES_M) == 0
        and shape[0] <= 65535 * _ROWS * _LANES_M * _WARPS
        and shape[1] % (cols * _LANES_N) == 0
    )


def quantize_transposed(
    x: jax.Array, inv_scale: jax.Array, dtype: DTypeLike, *, with_untransposed: bool = True
) -> tuple[jax.Array | None, jax.Array]:
    m, n = x.shape
    assert x.dtype == jnp.bfloat16 and quantize_transposed_supported(x.shape), (x.dtype, x.shape)
    call = _compiled_kernel(m, n, _PTX_CVT_TYPE[jnp.dtype(dtype).name], with_untransposed)
    x_words = jax.lax.bitcast_convert_type(x.reshape(m, n // 2, 2), jnp.int32)
    outputs = call(x_words, inv_scale.astype(jnp.float32).reshape(1))
    q_t = jax.lax.bitcast_convert_type(outputs[-1], dtype).reshape(n, m)
    q = jax.lax.bitcast_convert_type(outputs[0], dtype).reshape(m, n) if with_untransposed else None
    return q, q_t


def transpose(q: jax.Array) -> jax.Array:
    m, n = q.shape
    assert q.dtype.itemsize == 1 and transpose_supported(q.shape), (q.dtype, q.shape)
    call = _compiled_kernel(m, n, ptx_cvt_type=None, with_untransposed=False)
    (q_t,) = call(jax.lax.bitcast_convert_type(q.reshape(m, n // 4, 4), jnp.int32))
    return jax.lax.bitcast_convert_type(q_t, q.dtype).reshape(n, m)


def _store_transposed(words, num_words, m_q_t, col, row0):
    import cutlass.cute as cute
    from cutlass import Int32

    for word in range(num_words):
        outs = [cute.make_rmem_tensor((1, _ROWS // 4), Int32) for _ in range(4)]
        for block in range(_ROWS // 4):
            a0, a1, a2, a3 = (words[4 * block + i][0, word] for i in range(4))
            b0, b1 = cute.arch.prmt(a0, a1, 0x5140), cute.arch.prmt(a0, a1, 0x7362)
            b2, b3 = cute.arch.prmt(a2, a3, 0x5140), cute.arch.prmt(a2, a3, 0x7362)
            outs[0][0, block] = cute.arch.prmt(b0, b2, 0x5410)
            outs[1][0, block] = cute.arch.prmt(b0, b2, 0x7632)
            outs[2][0, block] = cute.arch.prmt(b1, b3, 0x5410)
            outs[3][0, block] = cute.arch.prmt(b1, b3, 0x7632)
        for j in range(4):
            cute.autovec_copy(
                outs[j],
                cute.local_tile(m_q_t, (1, _ROWS // 4), (col + 4 * word + j, row0 // _ROWS)),
            )


@functools.cache
def _compiled_kernel(m: int, n: int, ptx_cvt_type: str | None, with_untransposed: bool):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, const_expr
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.jax import cutlass_call

    quantize = ptx_cvt_type is not None
    cols = _QUANTIZE_COLS if quantize else _TRANSPOSE_COLS
    num_warp_rows = m // (_ROWS * _LANES_M)
    grid = [n // (cols * _LANES_N), -(-num_warp_rows // _WARPS), 1]
    block = [_WARP_SIZE * _WARPS, 1, 1]

    @dsl_user_op
    def quantize4(lo: Int32, hi: Int32, inv: Float32, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [arg.ir_value(loc=loc, ip=ip) for arg in (Int32(lo), Int32(hi), Float32(inv))],
                "{\n"
                ".reg .b32 a0, a1, a2, a3;\n"
                ".reg .b16 l, h;\n"
                "shl.b32 a0, $1, 16;\n"
                "and.b32 a1, $1, 0xFFFF0000;\n"
                "shl.b32 a2, $2, 16;\n"
                "and.b32 a3, $2, 0xFFFF0000;\n"
                "mul.rn.f32 a0, a0, $3;\n"
                "mul.rn.f32 a1, a1, $3;\n"
                "mul.rn.f32 a2, a2, $3;\n"
                "mul.rn.f32 a3, a3, $3;\n"
                f"cvt.rn.satfinite.{ptx_cvt_type}.f32 l, a1, a0;\n"
                f"cvt.rn.satfinite.{ptx_cvt_type}.f32 h, a3, a2;\n"
                "mov.b32 $0, {l, h};\n"
                "}",
                "=r,r,r,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.kernel
    def kernel(mIn, mInv, mQ, mQT):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp, lane = tidx // _WARP_SIZE, tidx % _WARP_SIZE
        warp_row = by * _WARPS + warp
        if warp_row < num_warp_rows:
            col = (bx * _LANES_N + lane % _LANES_N) * cols
            row0 = (warp_row * _LANES_M + lane // _LANES_N) * _ROWS
            if const_expr(quantize):
                inv = mInv[0]
            words = [cute.make_rmem_tensor((1, 4), Int32) for _ in range(_ROWS)]
            for r in cutlass.range_constexpr(_ROWS):
                cute.autovec_copy(cute.local_tile(mIn, (1, 4), (row0 + r, col // cols)), words[r])
            if const_expr(quantize):
                q = [cute.make_rmem_tensor((1, 2), Int32) for _ in range(_ROWS)]
                for r in cutlass.range_constexpr(_ROWS):
                    q[r][0, 0] = quantize4(words[r][0, 0], words[r][0, 1], inv)
                    q[r][0, 1] = quantize4(words[r][0, 2], words[r][0, 3], inv)
                    if const_expr(mQ is not None):
                        cute.autovec_copy(
                            q[r], cute.local_tile(mQ, (1, 2), (row0 + r, col // cols))
                        )
                words = q
            _store_transposed(words, cols // 4, mQT, col, row0)

    @cute.jit
    def launch(stream: cuda_driver.CUstream, mIn, mInv, mQ, mQT):
        kernel(mIn, mInv, mQ, mQT).launch(grid=grid, block=block, stream=stream)

    @cute.jit
    def launch_quantize_transposed_only(stream: cuda_driver.CUstream, mIn, mInv, mQT):
        launch(stream, mIn, mInv, None, mQT)

    @cute.jit
    def launch_transpose(stream: cuda_driver.CUstream, mIn, mQT):
        launch(stream, mIn, None, None, mQT)

    q_t_spec = jax.ShapeDtypeStruct((n, m // 4), jnp.int32)
    if not quantize:
        return cutlass_call(
            launch_transpose, output_shape_dtype=[q_t_spec], use_static_tensors=True
        )
    if not with_untransposed:
        return cutlass_call(
            launch_quantize_transposed_only, output_shape_dtype=[q_t_spec], use_static_tensors=True
        )
    q_spec = jax.ShapeDtypeStruct((m, n // 4), jnp.int32)
    return cutlass_call(launch, output_shape_dtype=[q_spec, q_t_spec], use_static_tensors=True)
