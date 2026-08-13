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
from quack import copy_utils


@dataclass(frozen=True)
class SeqlenInfo:
    offset: Int32
    offset_padded: Int32
    seqlen: Int32
    has_cu_seqlens: cutlass.Constexpr[bool] = False

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_static: Int32,
        cu_seqlens: cute.Tensor | None = None,
        seqused: cute.Tensor | None = None,
        tile: cutlass.Constexpr[int] = 128,
    ):
        offset = 0 if const_expr(cu_seqlens is None) else cu_seqlens[batch_idx]
        offset_padded = (
            0
            if const_expr(cu_seqlens is None)
            else cute.assume((offset + batch_idx * tile) // tile * tile, divby=tile)
        )
        if const_expr(seqused is not None):
            seqlen = seqused[batch_idx]
        elif const_expr(cu_seqlens is not None):
            seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            seqlen = seqlen_static
        return SeqlenInfo(offset, offset_padded, seqlen, has_cu_seqlens=cu_seqlens is not None)

    def offset_batch(
        self,
        mT: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        if const_expr(not self.has_cu_seqlens):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mT) - 1 - dim)
            return mT[idx]
        else:
            off = multiple * (self.offset if const_expr(not padded) else self.offset_padded)
            offset = off if const_expr(cute.rank(mT.shape[0]) == 1) else (0, off)
            idx = (offset,) + (None,) * (cute.rank(mT) - 1)
            return cute.domain_offset(idx, mT)


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32
    offset_k: Int32
    padded_offset_q: Int32
    padded_offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32
    m_block_offset: Int32
    block_idx_offset: Int32
    num_n_blocks: Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]
    has_seqused_q: cutlass.Constexpr[bool]
    has_seqused_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        mCuTotalMBlocks: cute.Tensor | None = None,
        mCuBlockIdxOffsets: cute.Tensor | None = None,
        tile_m: cutlass.Constexpr[Int32] = 128,
        tile_n: cutlass.Constexpr[Int32] = 128,
    ):
        offset_q = 0 if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        offset_k = 0 if const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        padded_offset_q = (
            0
            if const_expr(mCuSeqlensQ is None)
            else cute.assume((offset_q + batch_idx * tile_m) // tile_m * tile_m, divby=tile_m)
        )
        padded_offset_k = (
            0
            if const_expr(mCuSeqlensK is None)
            else cute.assume((offset_k + batch_idx * tile_n) // tile_n * tile_n, divby=tile_n)
        )
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = mSeqUsedQ[batch_idx]
        else:
            seqlen_q = (
                seqlen_q_static
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ[batch_idx + 1] - offset_q
            )
        if const_expr(mSeqUsedK is not None):
            seqlen_k = mSeqUsedK[batch_idx]
        else:
            seqlen_k = (
                seqlen_k_static
                if const_expr(mCuSeqlensK is None)
                else mCuSeqlensK[batch_idx + 1] - offset_k
            )
        m_block_offset = 0 if const_expr(mCuTotalMBlocks is None) else mCuTotalMBlocks[batch_idx]
        num_n_blocks = (seqlen_k + tile_n - 1) // tile_n
        block_idx_offset = (
            mCuBlockIdxOffsets[batch_idx]
            if const_expr(mCuBlockIdxOffsets is not None)
            else m_block_offset * num_n_blocks
        )
        return SeqlenInfoQK(
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            seqlen_q,
            seqlen_k,
            m_block_offset,
            block_idx_offset,
            num_n_blocks,
            has_cu_seqlens_q=mCuSeqlensQ is not None,
            has_cu_seqlens_k=mCuSeqlensK is not None,
            has_seqused_q=mSeqUsedQ is not None,
            has_seqused_k=mSeqUsedK is not None,
        )

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        if const_expr(not ragged):
            if const_expr(not self.has_cu_seqlens_q):
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
                return mQ[idx]
            else:
                offset_q = self.offset_q if const_expr(not padded) else self.padded_offset_q
                offset_q = offset_q if const_expr(cute.rank(mQ.shape[0]) == 1) else (None, offset_q)
                idx = (offset_q,) + (None,) * (cute.rank(mQ) - 1)
                return cute.domain_offset(idx, mQ)
        else:
            if const_expr(not self.has_cu_seqlens_q):
                offset_q = 0
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
                mQ = mQ[idx]
            else:
                offset_q = self.offset_q if const_expr(not padded) else self.padded_offset_q
            if const_expr(cute.rank(mQ.shape[0]) == 1):
                return copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=0, ptr_shift=True
                )
            else:
                assert cute.rank(mQ.shape[0]) == 2
                idx = ((None, None),) + (None,) * (cute.rank(mQ) - 1)
                mQ = mQ[idx]
                mQ = copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=1, ptr_shift=True
                )
                return cute.group_modes(mQ, 0, 2)

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        if const_expr(not ragged):
            if const_expr(not self.has_cu_seqlens_k):
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
                return mK[idx]
            else:
                offset_k = self.offset_k if const_expr(not padded) else self.padded_offset_k
                offset_k *= multiple
                idx = (offset_k,) + (None,) * (cute.rank(mK) - 1)
                return cute.domain_offset(idx, mK)
        else:
            if const_expr(not self.has_cu_seqlens_k):
                offset_k = 0
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
                mK = mK[idx]
            else:
                offset_k = self.offset_k if const_expr(not padded) else self.padded_offset_k
                offset_k *= multiple
            return copy_utils.offset_ragged_tensor(
                mK, offset_k, self.seqlen_k, ragged_dim=0, ptr_shift=True
            )


@dataclass(frozen=True)
class SeqlenInfoQKNewK:
    leftpad_k: Int32
    offset_q: Int32
    offset_k: Int32
    offset_k_new: Int32
    seqlen_q: Int32
    seqlen_k_og: Int32
    seqlen_k_new: Int32
    seqlen_k: Int32
    seqlen_rotary: Int32

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        shape_K_new_0: Int32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mCuSeqlensKNew: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        mLeftpadK: cute.Tensor | None = None,
        mSeqlensRotary: cute.Tensor | None = None,
    ):
        leftpad_k = 0 if const_expr(mLeftpadK is None) else mLeftpadK[batch_idx]
        offset_q = 0 if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        if const_expr(mCuSeqlensK is not None):
            offset_k = mCuSeqlensK[batch_idx] + leftpad_k
        else:
            offset_k = leftpad_k if const_expr(mCuSeqlensQ is not None) else 0
        offset_k_new = 0 if const_expr(mCuSeqlensKNew is None) else mCuSeqlensKNew[batch_idx]
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = mSeqUsedQ[batch_idx]
        elif const_expr(mCuSeqlensQ is not None):
            seqlen_q = mCuSeqlensQ[batch_idx + 1] - mCuSeqlensQ[batch_idx]
        else:
            seqlen_q = seqlen_q_static
        if const_expr(mSeqUsedK is not None):
            seqlen_k_og = mSeqUsedK[batch_idx] - leftpad_k
        elif const_expr(mCuSeqlensK is not None):
            seqlen_k_og = mCuSeqlensK[batch_idx + 1] - mCuSeqlensK[batch_idx] - leftpad_k
        else:
            seqlen_k_og = (
                seqlen_k_static - leftpad_k
                if const_expr(mCuSeqlensQ is not None)
                else seqlen_k_static
            )
        if const_expr(mCuSeqlensKNew is None):
            seqlen_k_new = 0 if const_expr(mCuSeqlensQ is None) else shape_K_new_0
        else:
            seqlen_k_new = mCuSeqlensKNew[batch_idx + 1] - mCuSeqlensKNew[batch_idx]
        seqlen_k = seqlen_k_og if const_expr(mCuSeqlensQ is None) else seqlen_k_og + seqlen_k_new

        if const_expr(mSeqlensRotary is not None):
            seqlen_rotary = mSeqlensRotary[batch_idx]
        else:
            seqlen_rotary = seqlen_k_og + leftpad_k
        return SeqlenInfoQKNewK(
            leftpad_k,
            offset_q,
            offset_k,
            offset_k_new,
            seqlen_q,
            seqlen_k_og,
            seqlen_k_new,
            seqlen_k,
            seqlen_rotary,
        )
