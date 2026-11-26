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
from functools import partial
from typing import Callable, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from xrex.cutedsl.quack import copy_utils, layout_utils, sm90_utils
from xrex.cutedsl.quack.cute_dsl_utils import ParamsBase
from xrex.cutedsl.quack.sm90_utils import gemm_w_idx, gemm_zero_init
from xrex.cutedsl.ranker_fa4 import barrier, pipeline, utils
from xrex.cutedsl.ranker_fa4.block_info import BlockInfo
from xrex.cutedsl.ranker_fa4.block_sparse_utils import (
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
)
from xrex.cutedsl.ranker_fa4.block_sparsity import BlockSparseTensors
from xrex.cutedsl.ranker_fa4.cute_dsl_utils import assume_tensor_aligned
from xrex.cutedsl.ranker_fa4.mask import AttentionMask
from xrex.cutedsl.ranker_fa4.named_barrier import NamedBarrierBwd
from xrex.cutedsl.ranker_fa4.seqlen_info import SeqlenInfoQK
from xrex.cutedsl.ranker_fa4.softmax import apply_score_mod_bwd_inner, apply_score_mod_inner
from xrex.cutedsl.ranker_fa4.tile_scheduler import (
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


def _load_lse_and_softmax_stats(m_block, producer_state, load_lse, load_row_max, load_inv_sum):
    load_lse(m_block, producer_state=producer_state)
    load_row_max(m_block, producer_state=producer_state)
    load_inv_sum(m_block, producer_state=producer_state)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        deterministic: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
        dQ_single_wg: bool = False,
        qhead_loop: bool = False,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.deterministic = deterministic
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_wg_mma = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_wg_mma
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        if qhead_per_kvhead > 1:
            assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
            assert self.num_wg_mma == 2, "GQA backward assumes 2 warp groups"
        self.qhead_loop = qhead_loop
        self.q_heads_per_cta = qhead_per_kvhead if qhead_loop else 1
        self.dKV_postprocess = qhead_per_kvhead > 1 and not qhead_loop
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024

        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32
        if dQ_single_wg:
            assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"
        self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(not self.dKV_postprocess):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage, mms)
            for shape, stage, mms in [
                ((self.tile_m, self.tile_hdim), self.Q_stage, self.tile_hdim // wg_d_dKV),
                ((self.tile_m, self.tile_hdimv), self.dO_stage, self.tile_hdim // wg_d_dKV),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),
        )

    def _get_tiled_mma(self):
        maybe_swap_mn = lambda shape, swap: (shape[1], shape[0], *shape[2:]) if swap else shape
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[1] if not self.SdP_swapAB else tiler_mn_SdP[0]),
        )
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.tile_hdim // atom_layout_dKV[1])
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.tile_hdimv // atom_layout_dKV[1])
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                cute.nvgpu.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1] if not self.dKV_swapAB else tiler_mn_d[0]),
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K if not self.dQ_swapAB else cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN if not self.dQ_swapAB else cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1] if not self.dQ_swapAB else tiler_mn_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]
        stat_elems = cute.round_up(self.tile_m, 64) * self.Q_stage if self.has_softmax_stats else 0
        sStat_struct = cute.struct.Align[cute.struct.MemRange[Float32, stat_elems], 128]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sRowMax: sStat_struct
            sInvSum: sStat_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mSoftmaxStats: Optional[Tuple[cute.Tensor, cute.Tensor]] = None,
        stream: cuda.CUstream = None,
    ):
        self.has_softmax_stats = mSoftmaxStats is not None
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        if const_expr(self.qhead_loop):
            assert blocksparse_tensors is not None, "qhead_loop needs block sparsity"
            assert not (self.is_varlen_q or self.varlen_k), (
                "qhead_loop does not support cu_seqlens or seqused"
            )

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(not self.dKV_postprocess):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]
        if const_expr(mSoftmaxStats is not None):
            mSoftmaxStats = tuple(
                layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in mSoftmaxStats
            )

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
        if const_expr(self.num_wg_mma == 2):
            if const_expr(self.num_wg_dQ == 1):
                self.num_mma_regs_wg0 = 256
                self.num_mma_regs_wg1 = 224
            else:
                self.num_mma_regs_wg0 = 240
                self.num_mma_regs_wg1 = 240
            self.num_mma_regs = self.num_mma_regs_wg0
            self.num_producer_regs = 24
            assert (
                self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= REG_LIMIT
            )
        else:
            self.num_mma_regs_wg0 = 160
            self.num_mma_regs_wg1 = 160
            self.num_mma_regs = 160
            self.num_producer_regs = 32
            assert self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs <= REG_LIMIT

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(not self.dKV_postprocess):
            mdK_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdK
            )
            mdV_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdV
            )
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK_tma,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mK.shape[2]) if const_expr(self.qhead_loop) else cute.size(mQ.shape[2]),
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),
            1,
            cute.size(mQ.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mK.shape[0])
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(not self.dKV_postprocess) else mdK,
            tma_tensor_dV if const_expr(not self.dKV_postprocess) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mSoftmaxStats,
            mdPsum,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mSoftmaxStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"]
            + self.tma_copy_bytes["LSE"] * (3 if const_expr(mSoftmaxStats is not None) else 1),
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sStats = None
        if const_expr(mSoftmaxStats is not None):
            sStats = tuple(
                st.get_tensor(
                    cute.make_layout(
                        (self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
                    )
                )
                for st in (storage.sRowMax, storage.sInvSum)
            )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    mSoftmaxStats,
                    sStats,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    mdQ_semaphore,
                )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sStats,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_tensors,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )
            if const_expr(self.num_wg_dQ == self.num_wg_mma):
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mSoftmaxStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        sStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        q_head_loads = partial(
            self.get_q_head_loads,
            mQ=mQ,
            mdO=mdO,
            mLSE=mLSE,
            mdPsum=mdPsum,
            sQ=sQ,
            sdO=sdO,
            sLSE=sLSE,
            sdPsum=sdPsum,
            mSoftmaxStats=mSoftmaxStats,
            sStats=sStats,
            tma_atom_Q=tma_atom_Q,
            tma_atom_dO=tma_atom_dO,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
        )

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1 or self.qhead_loop)
                    else head_idx // qhead_per_kvhead_divmod
                )
                first_q_head = head_idx * self.q_heads_per_cta
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local and not self.is_varlen_q)
                        or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = self.total_q_block_count(
                        blocksparse_tensors, batch_idx, first_q_head, n_block, m_block_max
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        load_Q, load_dO, load_LSE, load_dPsum = q_head_loads(
                            seqlen, batch_idx, head_idx
                        )
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(
                            producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        cute.arch.griddepcontrol_wait()
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(
                            producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:
                        kv_loaded = False
                        for q_head in cutlass.range(
                            first_q_head, first_q_head + self.q_heads_per_cta, unroll=1
                        ):
                            producer_state_Q, producer_state_dO, kv_loaded = (
                                produce_block_sparse_q_loads_bwd_sm90(
                                    blocksparse_tensors,
                                    batch_idx,
                                    q_head,
                                    n_block,
                                    producer_state_Q,
                                    producer_state_dO,
                                    pipeline_Q,
                                    pipeline_dO,
                                    load_K,
                                    load_V,
                                    *q_head_loads(seqlen, batch_idx, q_head),
                                    self.tma_copy_bytes["K"],
                                    self.tma_copy_bytes["V"],
                                    Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                                    subtile_factor=self.subtile_factor,
                                    m_block_max=m_block_max,
                                    kv_loaded=kv_loaded,
                                )
                            )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def get_q_head_loads(
        self,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        q_head: Int32,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mSoftmaxStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        sStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
    ) -> tuple[Callable, Callable, Callable, Callable]:
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, q_head]
        mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, q_head]
        mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, q_head]
        mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, q_head]
        gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
        gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
        gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
        load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ)
        load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
        load_dO, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO)
        load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
        load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
        load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
        if const_expr(mSoftmaxStats is not None):
            mRowMax_cur = seqlen.offset_batch_Q(mSoftmaxStats[0], batch_idx, dim=2, padded=True)[
                None, q_head
            ]
            mInvSum_cur = seqlen.offset_batch_Q(mSoftmaxStats[1], batch_idx, dim=2, padded=True)[
                None, q_head
            ]
            gRowMax = cute.local_tile(mRowMax_cur, (self.tile_m,), (None,))
            gInvSum = cute.local_tile(mInvSum_cur, (self.tile_m,), (None,))
            load_LSE = partial(
                _load_lse_and_softmax_stats,
                load_lse=load_LSE,
                load_row_max=copy_utils.tma_producer_copy_fn(
                    copy_utils.cpasync_bulk_get_copy_fn(gRowMax, sStats[0]), pipeline_Q
                ),
                load_inv_sum=copy_utils.tma_producer_copy_fn(
                    copy_utils.cpasync_bulk_get_copy_fn(gInvSum, sStats[1]), pipeline_Q
                ),
            )
        load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
        load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)
        return load_Q, load_dO, load_LSE, load_dPsum

    @cute.jit
    def total_q_block_count(
        self,
        blocksparse_tensors: BlockSparseTensors,
        batch_idx: Int32,
        first_q_head: Int32,
        n_block: Int32,
        m_block_max: Int32,
    ) -> Int32:
        total = Int32(0)
        for q_head_in_cta in cutlass.range_constexpr(self.q_heads_per_cta):
            total += get_total_q_block_count_bwd(
                blocksparse_tensors,
                batch_idx,
                first_q_head + q_head_in_cta,
                n_block,
                subtile_factor=self.subtile_factor,
                m_block_max=m_block_max,
            )
        return total

    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            tScS,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB
        )
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB
        )
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(
                gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )

        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        shfl_copy = copy_utils.tiled_copy_1d(sLSE.element_type, num_threads=8, num_copy_elems=2)
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsLSE)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsdPsum)
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)
        tLSEsStats = None
        if const_expr(sStats is not None):
            tLSEsStats = []
            for sStat in sStats:
                tStat = layout_utils.mma_partition_C_vec(
                    sStat, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
                )
                if const_expr(self.shuffle_LSE):
                    tStat = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tStat)
                    tStat = cute.group_modes(tStat, 0, 2)
                tLSEsStats.append(tStat)
            tLSEsStats = tuple(tLSEsStats)

        tdQsdQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )
        score_mod_fn = partial(
            self.apply_score_mod,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
        score_mod_bwd_fn = partial(
            self.apply_score_mod_bwd,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tLSEsStats=tLSEsStats,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            is_dQ_wg=is_dQ_wg,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            first_q_head = head_idx * self.q_heads_per_cta

            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = self.total_q_block_count(
                    blocksparse_tensors, batch_idx, first_q_head, n_block, m_block_max
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_tensors=aux_tensors,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            score_mod_fn=partial(score_mod_fn_cur, head_idx=head_idx),
                            score_mod_bwd_fn=partial(score_mod_bwd_fn_cur, head_idx=head_idx),
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
                else:
                    dKV_accumulate = False
                    for q_head in cutlass.range(
                        first_q_head, first_q_head + self.q_heads_per_cta, unroll=1
                    ):
                        consumer_state_Q, consumer_state_dO, dKV_accumulate = (
                            consume_block_sparse_mma_bwd_sm90(
                                blocksparse_tensors,
                                batch_idx,
                                q_head,
                                n_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mma_one_m_block_all,
                                mask,
                                self.mask_mod,
                                is_causal=self.is_causal,
                                is_local=self.is_local,
                                thr_mma_SdP=thr_mma_SdP,
                                score_mod_fn=partial(score_mod_fn_cur, head_idx=q_head),
                                score_mod_bwd_fn=partial(score_mod_bwd_fn_cur, head_idx=q_head),
                                subtile_factor=self.subtile_factor,
                                m_block_max=m_block_max,
                                aux_tensors=aux_tensors,
                                fastdiv_mods=fastdiv_mods,
                                dKV_accumulate=dKV_accumulate,
                            )
                        )

                if const_expr(not self.dKV_postprocess):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                    mdK_semaphore,
                    mdV_semaphore,
                )
            else:
                if const_expr(self.use_block_sparsity or self.is_local or self.is_varlen_q):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        if const_expr(not shuffle):
            return tSrS[row]
        vecsize = cute.size(tSrS, mode=[0, 0])
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tLSEsStats: Optional[Tuple[cute.Tensor, cute.Tensor]],
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        score_mod_fn: Optional[Callable] = None,
        score_mod_bwd_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
    ):
        consumer_state_dO_cur = (
            consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        if const_expr(tLSEsStats is not None):
            tLSErRowMax = copy_utils.load_s2r(tLSEsStats[0][None, smem_idx_Q])
            tLSErInvSum = copy_utils.load_s2r(tLSEsStats[1][None, smem_idx_Q])
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        if const_expr(self.score_mod_bwd is not None):
            acc_S_pre = cute.make_fragment_like(acc_S)
            cute.autovec_copy(acc_S, acc_S_pre)

        if const_expr(self.score_mod is not None):
            score_mod_fn(acc_S, m_block=m_block)

        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            if const_expr(tLSEsStats is not None):
                row_max = self._get_stat(tLSErRowMax, r, lane_idx, shuffle=self.shuffle_LSE)
                inv_sum = self._get_stat(tLSErInvSum, r, lane_idx, shuffle=self.shuffle_LSE)
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    acc_S_mn[r, c] = (
                        cute.math.exp2(
                            (acc_S_mn[r, c] - row_max) * softmax_scale_log2, fastmath=True
                        )
                        * inv_sum
                    )
            else:
                lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
                row_masked = lse_val == -Float32.inf
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    acc_S_mn[r, c] = (
                        Float32(0.0)
                        if row_masked
                        else cute.math.exp2(
                            acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                        )
                    )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        if const_expr(not self.mma_dkv_is_rs):
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        if const_expr(self.score_mod_bwd is not None):
            score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )
        else:
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(is_dQ_wg):
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)

            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            tdQrdQaccum_flat = cute.make_tensor(
                acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
            )
            cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )

            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(not self.dKV_postprocess):
            mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )
            sdV = sV if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sV)
            sdK = sK if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sK)
            copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dV,
                sdV,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dK,
                sdK,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dV_r2s(acc_dV, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
        else:
            deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
            sdKaccum_shape0 = self.tile_n * self.tile_hdim // self.num_wg_mma
            sdVaccum_shape0 = self.tile_n * self.tile_hdimv // self.num_wg_mma
            sdKaccum_layout = cute.make_layout((sdKaccum_shape0, self.num_wg_mma))
            sdVaccum_layout = cute.make_layout((sdVaccum_shape0, self.num_wg_mma))
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            if const_expr(deterministic_KV):
                assert mdK_semaphore is not None
                assert mdV_semaphore is not None
                mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                lock_value = head_idx % self.qhead_per_kvhead
            mdKaccum_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim
            )[None, head_idx_kv]
            mdVaccum_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv
            )[None, head_idx_kv]
            gdKaccum_ = cute.local_tile(mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,))
            gdKaccum = cute.flat_divide(gdKaccum_, (sdKaccum_shape0,))
            gdVaccum_ = cute.local_tile(mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,))
            gdVaccum = cute.flat_divide(gdVaccum_, (sdVaccum_shape0,))
            sVaccum_ptr = cute.recast_ptr(sV.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sVaccum_ptr, sdKaccum_layout)
            sdVaccum = cute.make_tensor(sVaccum_ptr, sdVaccum_layout)
            tiled_copy_dKVaccum_r2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((self.num_threads_per_warp_group, self.num_wg_mma)),
                cute.make_layout(128 // Float32.width),
            )
            thr_copy_dKVaccum_r2s = tiled_copy_dKVaccum_r2s.get_slice(tidx)
            tdKsdKaccum = thr_copy_dKVaccum_r2s.partition_D(sdKaccum)
            tdVsdVaccum = thr_copy_dKVaccum_r2s.partition_D(sdVaccum)

            read_flag = const_expr(not deterministic_KV)
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.wait_eq(mdK_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, tdKsdKaccum.shape)
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.arrive_inc(mdK_semaphore_cur.iterator, tidx, 0, 1)
                barrier.wait_eq(mdV_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, tdVsdVaccum.shape)
            cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()
            if const_expr(deterministic_KV):
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                barrier.arrive_inc(mdV_semaphore_cur.iterator, tidx, 0, 1)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_local_tidx = tidx % cute.arch.WARP_SIZE
        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            first_q_head = head_idx * self.q_heads_per_cta

            if const_expr(mdQ_semaphore is not None):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = self.total_q_block_count(
                    blocksparse_tensors, batch_idx, first_q_head, n_block, m_block_max
                )
                process_tile = total_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    gdQaccum = self.get_q_head_dQaccum(mdQaccum, seqlen, batch_idx, head_idx)
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block = m_block_min + iter_idx
                        m_block_safe = m_block

                        num_dQ_chunks = self.num_wg_dQ
                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            if const_expr(not self.deterministic):
                                cute.arch.cp_async_bulk_wait_group(
                                    num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                )
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )

                        if const_expr(self.deterministic):
                            if const_expr(self.spt):
                                _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                                    seqlen, m_block_safe
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block
                            else:
                                lock_value = n_block
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,
                                lock_value,
                            )

                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, warp_group_idx].iterator,
                                    gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            cute.arch.cp_async_bulk_commit_group()

                        if const_expr(self.deterministic):
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,
                                1,
                            )
                else:
                    assert not self.deterministic, (
                        "Deterministic not implemented for block-sparse backward"
                    )
                    for q_head in cutlass.range(
                        first_q_head, first_q_head + self.q_heads_per_cta, unroll=1
                    ):
                        dQaccum_store_block_sparse_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            q_head,
                            n_block,
                            sdQaccum,
                            self.get_q_head_dQaccum(mdQaccum, seqlen, batch_idx, q_head),
                            subtile_factor=self.subtile_factor,
                            m_block_max=m_block_max,
                            num_dQ_warp_groups=self.num_wg_dQ,
                            num_threads_per_warp_group=self.num_threads_per_warp_group,
                            tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                        )

            if const_expr(
                self.deterministic and not self.spt and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        warp_local_tidx,
                        0,
                        1,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def get_q_head_dQaccum(
        self, mdQaccum: cute.Tensor, seqlen: SeqlenInfoQK, batch_idx: Int32, q_head: Int32
    ) -> cute.Tensor:
        if const_expr(not seqlen.has_cu_seqlens_q):
            mdQaccum_cur = mdQaccum[None, q_head, batch_idx]
        else:
            mdQaccum_cur = cute.domain_offset(
                (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, q_head]
            )
        return cute.local_tile(
            mdQaccum_cur,
            (cute.make_layout((self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)),),
            (None,),
        )
