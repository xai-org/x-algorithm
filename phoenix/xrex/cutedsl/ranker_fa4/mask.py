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
import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import min as dsl_min
from quack import layout_utils

import xrex.cutedsl.ranker_fa4.utils as utils
from xrex.cutedsl.ranker_fa4.block_info import BlockInfo
from xrex.cutedsl.ranker_fa4.seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Int32 | None = None
    window_size_right: Int32 | None = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                r2p = const_expr(not self.swap_AB)
                if const_expr(not r2p):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                    mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            wrap_aux_indices = const_expr(
                has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
            )

            for r in cutlass.range_constexpr(nrow):
                local_row = tScS_mn[r, 0][ROW]
                global_row_idx = local_row + m_block * self.tile_m
                row_for_mod = global_row_idx
                head_idx_for_mod = head_idx
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
                row_for_seqlen = row_for_mod
                if const_expr(wrap_aux_indices):
                    _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][COL]
                    global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
                    col_for_mod = global_col_idx
                    if const_expr(wrap_aux_indices):
                        _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

                    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                    head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                    q_idx_ssa = utils.scalar_to_ssa(row_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(col_for_mod, cutlass.Int32)
                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    if const_expr(mask_seqlen):
                        out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                    else:
                        acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf

        else:
            if const_expr(not self.swap_AB):
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(not self.swap_AB)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            mask_r2p_lambda(
                                acc_S_mn[r, None],
                                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                                rank1=True,
                            )
                else:
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    r2p_local = const_expr(not self.swap_AB)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        if const_expr(not r2p_local):
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                col_idx = t0ScS_mn[0, c][1]
                                if col_idx >= col_limit_right or col_idx < col_limit_left:
                                    acc_S_mn[r, c] = -Float32.inf
                        else:
                            col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

                            def mask_gen_fn(s: int) -> Uint32:
                                return r2p_bitmask_below(
                                    col_limit_right_r2p,
                                    s,
                                ) & r2p_bitmask_above(col_limit_left_r2p, s)

                            mask_r2p_lambda(acc_S_mn[r, None], mask_gen_fn, rank1=True)
            else:
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else (
                                col0 - causal_row_offset - self.window_size_right
                                if const_expr(self.window_size_right is not None)
                                else 0
                            )
                        )
                        row_limit_bot = (
                            col0 - causal_row_offset + self.window_size_left
                            if const_expr(self.window_size_left is not None)
                            else self.tile_m
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        mask_diagonal: cutlass.Constexpr[bool] = False,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
        r2p: bool = True,
        rBitmask: cute.Tensor | None = None,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        assert not (mask_diagonal and (mask_causal or mask_local or mask_mod is not None)), (
            "mask_diagonal is mutually exclusive with mask_causal/mask_local/mask_mod"
        )
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n

        if const_expr(rBitmask is not None):
            ncol_packed = const_expr(cute.size(rBitmask.shape[0]))
            for i in cutlass.range_constexpr(ncol_packed):
                col_start = 32 * i
                curr_mask_val = rBitmask[i]
                for j in cutlass.range_constexpr(32):
                    curr_col = col_start + j
                    mask = (curr_mask_val >> j) & 1
                    acc_S[curr_col] = acc_S[curr_col] if cutlass.Boolean(mask) else -Float32.inf

        elif const_expr(mask_diagonal):
            ROW = 0 if const_expr(not self.swap_AB) else 1
            COL = 1 if const_expr(not self.swap_AB) else 0
            qpk = const_expr(self.qhead_per_kvhead_packgqa)
            per_packed_pos = const_expr(self.tile_m // qpk)
            diag_offset = m_block * per_packed_pos - n_block * self.tile_n
            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][ROW]
                col_coord = tScS_t2r[i][COL]
                if const_expr(qpk != 1):
                    q_pos_in_tile = row_coord // qpk
                else:
                    q_pos_in_tile = row_coord
                on_diag = col_coord == q_pos_in_tile + diag_offset
                acc_S[i] = acc_S[i] if on_diag else -Float32.inf
                if const_expr(mask_seqlen):
                    global_col = col_coord + n_block * self.tile_n
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    global_row = row_coord + m_block * self.tile_m
                    if const_expr(qpk != 1):
                        assert head_divmod is not None
                        mask_row, _ = divmod(global_row, head_divmod)
                    else:
                        mask_row = global_row
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

        elif const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
                        rank1=True,
                    )

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)

            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
                col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
                global_row = row_coord + m_block * self.tile_m
                global_col = col_coord + n_block * self.tile_n

                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert head_divmod is not None
                    mask_row, head_offset = divmod(global_row, head_divmod)
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                else:
                    head_idx_for_mod = head_idx
                    mask_row = global_row

                mask_row_for_mod = mask_row
                if const_expr(has_fastdiv and aux_tensors is not None):
                    if check_q_boundary:
                        _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
                global_col_for_mod = global_col
                if const_expr(has_fastdiv and mask_seqlen and aux_tensors is not None):
                    _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

                head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
                kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
                mask_value = mask_mod(
                    batch_idx_ssa,
                    head_idx_ssa,
                    mask_row_ssa,
                    kv_idx_ssa,
                    self.seqlen_info,
                    aux_tensors,
                )
                cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                acc_S[i] = acc_S[i] if cond else -Float32.inf
                if const_expr(mask_seqlen):
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

        else:
            causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset + 1
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(col_limit_right, s),
                        rank1=True,
                    )
            else:
                local_row_offset_right = (
                    causal_row_offset + 1 + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:

                    def mask_gen_fn(s: int) -> Uint32:
                        return r2p_bitmask_below(col_limit_right, s) & r2p_bitmask_above(
                            col_limit_left, s
                        )

                    mask_r2p_lambda(acc_S, mask_gen_fn, rank1=True)

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        is_diagonal_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if is_diagonal_block:
            qpk = const_expr(self.qhead_per_kvhead_packgqa)
            per_packed_pos = const_expr(self.tile_m // qpk)
            diag_offset = m_block * per_packed_pos - n_block * self.tile_n
            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][ROW]
                col_coord = tScS_t2r[i][COL]
                if const_expr(qpk != 1):
                    q_pos_in_tile = row_coord // qpk
                else:
                    q_pos_in_tile = row_coord
                on_diag = col_coord == q_pos_in_tile + diag_offset
                acc_S[i] = acc_S[i] if on_diag else -cutlass.Float32.inf
                if const_expr(mask_seqlen):
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n
                    q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
                    kv_out_of_bounds = global_kv >= self.seqlen_k
                    out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                    acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            if is_full_block:
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                            acc_S[i] = -cutlass.Float32.inf
                    elif check_m_boundary:
                        ncol = const_expr(cute.size(tScS_t2r.shape))
                        for i in cutlass.range_constexpr(ncol):
                            row_coord = tScS_t2r[i][ROW]
                            col_coord = tScS_t2r[i][COL]
                            global_q = row_coord + m_block * self.tile_m
                            global_kv = col_coord + n_block * self.tile_n
                            q_out_of_bounds = global_q >= self.seqlen_q
                            kv_out_of_bounds = global_kv >= self.seqlen_k
                            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                            acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]
            else:
                has_fastdiv = const_expr(
                    fastdiv_mods is not None
                    and fastdiv_mods[0] is not None
                    and fastdiv_mods[1] is not None
                )
                wrap_aux_indices = const_expr(
                    has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
                )
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)

                ncol = const_expr(cute.size(tScS_t2r.shape))
                for i in cutlass.range_constexpr(ncol):
                    row_coord = tScS_t2r[i][ROW]
                    col_coord = tScS_t2r[i][COL]
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n

                    q_idx_for_mod = global_q
                    kv_idx_for_mod = global_kv
                    if const_expr(wrap_aux_indices):
                        _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
                        _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

                    q_idx_ssa = utils.scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

                    if const_expr(mask_seqlen):
                        q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
                        kv_out_of_bounds = global_kv >= self.seqlen_k
                        out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                        acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])
                    num_wg = 2
                    row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_above(row_limit, s),
                        rank1=True,
                    )
            else:
                if const_expr(self.window_size_right is not None):
                    row_limit_top = causal_offset - self.window_size_right
                else:
                    row_limit_top = 0
                if const_expr(self.window_size_left is not None):
                    row_limit_bot = causal_offset + self.window_size_left
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        row_idx = t0ScS_t2r[i][ROW]
                        local_mask = row_idx < row_limit_top
                        if const_expr(self.window_size_left is not None):
                            local_mask |= row_idx > row_limit_bot
                        acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
                else:

                    def mask_gen_fn(s: int) -> Uint32:
                        num_rep = cute.size(tScS_t2r, mode=[0])
                        num_wg = 2

                        row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                        mask = r2p_bitmask_above(row_limit, s)

                        if const_expr(self.window_size_left is not None):
                            row_limit_bottom = row_to_r2p_idx(row_limit_bot + 1, num_rep, num_wg)
                            mask = mask & r2p_bitmask_below(row_limit_bottom, s)

                        return mask

                    mask_r2p_lambda(
                        acc_S,
                        mask_gen_fn,
                        rank1=True,
                    )


class Sm100MaskEnum(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()


class Sm100FusedMask:
    def get_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> Int32:
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == Sm100MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is Sm100MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK
            or mask_type == Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = dsl_min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type == Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = dsl_min(max_blocks_q, tmp_blocks_q)
        start_block = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start_count_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> tuple[Int32, Int32]:
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        return n_block_min, n_block_max - n_block_min

    @cute.jit
    def get_trip_mask_bounds_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> tuple[Int32, Int32]:
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
        )
        n_block_min, _ = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        return n_block_min_causal_local_mask, n_block_min_before_local_mask

    @cute.jit
    def get_trip_start(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> Int32:
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> tuple[Int32, Int32]:
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                leading_mask_end = dsl_min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> tuple[Int32 | None, Int32 | None]:
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = dsl_min(
                    min_idx_q // tile_shape[1], trip_count + trip_start - 1
                )
                trailing_mask_end = trip_count + trip_start - 1
            else:
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                trailing_mask_begin = dsl_min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = dsl_min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> Int32:
        result = 0
        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        rem_count: Int32 | None = 0,
    ) -> Int32:
        result = 0

        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                trailing_mask_begin, trailing_mask_end = Sm100FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(
                    trailing_mask_begin is not None and trailing_mask_end is not None
                ):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
    ) -> Int32:
        result = (
            Sm100FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def apply_mask(
        mask_type: Sm100MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        offset = 0
        if cutlass.const_expr(window_size_left is None and window_size_right is not None) or cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(
                mask_type == Sm100MaskEnum.RESIDUAL_MASK
                or mask_type == Sm100MaskEnum.RESIDUAL_MASK_BWD
            ):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf

    @cute.jit
    def apply_mask_via_causal_local(
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        apply_semantic_window: cutlass.Constexpr[bool] = True,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        offset = 0
        if cutlass.const_expr(apply_semantic_window):
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(apply_semantic_window):
                if cutlass.const_expr(is_causal and not is_local):
                    right = 0 if const_expr(window_size_right is None) else window_size_right
                    if index_q + offset + right < index_k:
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(
                    is_local or window_size_left is not None or window_size_right is not None
                ):
                    if cutlass.const_expr(window_size_left is None):
                        if index_q + offset + window_size_right < index_k:
                            acc_qk[i] = -Float32.inf
                    elif cutlass.const_expr(window_size_right is None):
                        if index_q + offset - window_size_left > index_k:
                            acc_qk[i] = -Float32.inf
                    else:
                        max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                        min_K_index = max(0, index_q + offset - window_size_left)
                        if index_k > max_K_index or index_k < min_K_index:
                            acc_qk[i] = -Float32.inf
            if index_k >= seqlen_k or index_q >= seqlen_q:
                acc_qk[i] = -Float32.inf
