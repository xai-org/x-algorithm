# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Derived from flash-attention (https://github.com/Dao-AILab/flash-attention);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def cvt_copy(
    atom: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type, loc=loc, ip=ip)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    cute.copy(atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_fragment_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def get_copy_atom(
    dtype: type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom, num_wg: int = 1, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_dp, num_bits, num_rep, _ = sm100_utils.get_tmem_copy_properties(tmem_copy_atom)
    assert num_dp == 32
    assert num_bits == 32
    tiler_mn = (cute.make_layout((128 * num_rep * num_wg // 32, 32), stride=(32, 1)),)
    layout_tv = cute.make_layout(
        ((32, 4, num_wg), (num_rep, 32)), stride=((0, 1, 4 * num_rep), (4, 4 * num_rep * num_wg))
    )
    return cute.make_tiled_copy(tmem_copy_atom, layout_tv, tiler_mn)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_1d(
    dtype: type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: type[cutlass.Numeric], major_mode_size: int, num_threads: int, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@dsl_user_op
def atomic_add_fp32x4(
    a: Float32, b: Float32, c: Float32, d: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            gmem_ptr_i64,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "mov.f32 abcd.x, $1;\n\t"
        "mov.f32 abcd.y, $2;\n\t"
        "mov.f32 abcd.z, $3;\n\t"
        "mov.f32 abcd.w, $4;\n\t"
        "red.global.add.v4.f32 [$0], abcd;\n\t"
        "}\n",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote_fp32x4(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "mov.f32 abcd.x, $2;\n\t"
        "mov.f32 abcd.y, $3;\n\t"
        "mov.f32 abcd.z, $4;\n\t"
        "mov.f32 abcd.w, $5;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], abcd, [$1];\n\t"
        "}\n",
        "r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_src_ptr_i32 = smem_src_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_dst_ptr_i32 = set_block_rank(
        smem_dst_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_dst_ptr_i32,
            smem_src_ptr_i32,
            mbar_ptr_i32,
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    tma_bar_ptr: cute.Pointer,
    size: int | Int32,
    *,
    loc=None,
    ip=None,
):
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = tma_bar_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, smem_ptr_i32, mbar_ptr_i32, Int32(size).ir_value()],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$0], $3, [$2];",
        "l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    group_rank_src = const_expr(cute.rank(src_tensor) - (1 if not single_stage else 0))
    group_rank_dst = const_expr(cute.rank(dst_tensor) - (1 if not single_stage else 0))
    src = cute.group_modes(src_tensor, 0, group_rank_src)
    dst = cute.group_modes(dst_tensor, 0, group_rank_dst)

    def copy_bulk(src_idx, dst_idx, **new_kwargs):
        size = const_expr(cute.size(src.shape[:-1]) * src.element_type.width // 8)
        cpasync_bulk_g2s(
            src[None, src_idx].iterator,
            dst[None, dst_idx].iterator,
            size=size,
            **new_kwargs,
            **kwargs,
        )

    def copy_bulk_single_stage(**new_kwargs):
        size = const_expr(cute.size(src.shape) * src.element_type.width // 8)
        cpasync_bulk_g2s(src.iterator, dst.iterator, size=size, **new_kwargs, **kwargs)

    return copy_bulk if const_expr(not single_stage) else copy_bulk_single_stage


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn
