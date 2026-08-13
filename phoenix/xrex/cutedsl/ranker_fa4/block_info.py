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
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from xrex.cutedsl.ranker_fa4.seqlen_info import SeqlenInfoQK, SeqlenInfoQKNewK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Int32 | None = None
    window_size_right: Int32 | None = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        if cutlass.const_expr(self.is_split_kv):
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_m_block_min_max(self, seqlen_info: SeqlenInfoQK, n_block: Int32) -> tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        m_block_min = 0
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_right = m_idx if const_expr(self.is_causal) else m_idx - self.window_size_right
            m_block_min = max(m_block_min, m_idx_right // self.tile_m)
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_left = m_idx + self.window_size_left
            m_block_max = min(m_block_max, cute.ceil_div(m_idx_left, self.tile_m))
        return m_block_min, m_block_max

    @cute.jit
    def get_n_block_k_new_min_max(
        self,
        seqlen_info: SeqlenInfoQKNewK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> tuple[Int32, Int32]:
        n_block_min, n_block_max = self.get_n_block_min_max(
            seqlen_info,
            m_block,
            split_idx,
            num_splits,
        )
        idx_k_new_min = cutlass.max(n_block_min * self.tile_n - seqlen_info.seqlen_k_og, 0)
        idx_k_new_max = cutlass.min(
            n_block_max * self.tile_n - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new
        )
        n_block_new_min = idx_k_new_min // self.tile_n
        n_block_new_max = (
            cute.ceil_div(idx_k_new_max, self.tile_n)
            if idx_k_new_max > idx_k_new_min
            else n_block_new_min
        )
        return n_block_new_min, n_block_new_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        if const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))

    @cute.jit
    def get_n_block_max_for_m_block(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Int32:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or self.window_size_right is not None):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            if const_expr(self.window_size_right is not None):
                n_idx_right += self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        return n_block_max
