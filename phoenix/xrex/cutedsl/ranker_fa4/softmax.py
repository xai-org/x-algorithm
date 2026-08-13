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
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

import xrex.cutedsl.ranker_fa4.utils as utils
from xrex.cutedsl.ranker_fa4.seqlen_info import SeqlenInfoQK


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                row_sum[r] += cute.math.exp2(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
                )

            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0
    max_offset: cutlass.Constexpr[int] = 0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        max_offset: cutlass.Constexpr[int] = 0,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
            max_offset=max_offset,
        )

    @cute.jit
    def compute_row_max_local(self, acc_S_row: cute.TensorSSA, is_first: Boolean) -> Float32:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        return row_max_new

    @cute.jit
    def update_row_max_from_local(
        self,
        row_max_new: Float32,
        is_first: Boolean,
    ) -> tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int) -> tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        row_max_scaled = row_max * self.scale_log2
        max_offset = Float32(self.max_offset)
        bias = max_offset - row_max_scaled
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (bias, bias),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                if cutlass.const_expr(ex2_emu_freq == 0):
                    acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                else:
                    if cutlass.const_expr(
                        k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(
                            acc_S_row_frg[k + 1, j], fastmath=True
                        )
                    else:
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)

    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))

    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to((vec_size,))

        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        post_mod_scores = score_mod(
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            seqlen_info=seqlen_info,
            aux_tensors=aux_args,
        )

        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]


@cute.jit
def apply_score_mod_bwd_inner(
    grad_tensor,
    score_tensor,
    index_tensor,
    score_mod_bwd: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)
    n_vals = cutlass.const_expr(cute.size(grad_tensor.shape))
    grad_vec = cute.make_fragment(vec_size, qk_acc_dtype)
    score_vec = cute.make_fragment(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))
    q_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)

    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_vec[j] = grad_tensor[i + j]
            score_vec[j] = score_tensor[i + j] * softmax_scale

            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        grad_ssa = grad_vec.load()
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()

        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            q_idx_ssa = utils.scalar_to_ssa(constant_q_idx, cutlass.Int32).broadcast_to((vec_size,))

        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        grad_out_ssa = score_mod_bwd(
            grad_ssa,
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            seqlen_info=seqlen_info,
            aux_tensors=aux_args,
        )

        grad_vec.store(grad_out_ssa)
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_tensor[i + j] = grad_vec[j]
