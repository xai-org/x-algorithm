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
#
# BSD 3-Clause License
#
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
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
# Portions ported/derived from NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from enum import IntEnum

import cutlass
import cutlass.cute as cute


class Major(IntEnum):
    K = 0
    MN = 1


class ScaleIn(IntEnum):
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


def to_UMMA_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}")


def make_instr_desc(
    a_type,
    b_type,
    c_type,
    M: int,
    N: int,
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4
    n_dim = N >> 3

    desc = 0
    desc |= (0 & 0x3) << 0
    desc |= (int(is_sparse) & 0x1) << 2
    desc |= (int(c_sat) & 0x1) << 3
    desc |= (c_fmt & 0x3) << 4
    desc |= (a_fmt & 0x7) << 7
    desc |= (b_fmt & 0x7) << 10
    desc |= (int(a_neg) & 0x1) << 13
    desc |= (int(b_neg) & 0x1) << 14
    desc |= (int(a_major) & 0x1) << 15
    desc |= (int(b_major) & 0x1) << 16
    desc |= (n_dim & 0x3F) << 17
    desc |= (m_dim & 0x1F) << 24
    desc |= (int(max_shift) & 0x3) << 30

    return desc & 0xFFFF_FFFF


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K if op.a_major_mode == cute.nvgpu.OperandMajorMode.K else Major.MN,
        Major.K if op.b_major_mode == cute.nvgpu.OperandMajorMode.K else Major.MN,
    )


class LayoutType(IntEnum):
    SWIZZLE_NONE = 0
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[B]
    if M == 5:
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(layout: cute.Layout, swizzle: cute.Swizzle, major: Major) -> int:
    layout_type = _layout_type(swizzle)

    VERSION = 1
    LBO_MODE = 0
    BASE_OFFSET = 0

    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(layout, (swizzle_atom_mn_size, swizzle_atom_k_size))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_MN Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = canonical_layout.stride[0][1], canonical_layout.stride[1][1]
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError("Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.")
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    desc = 0
    desc |= (leading_byte_offset & 0x3FFF) << 16
    desc |= (stride_byte_offset & 0x3FFF) << 32
    desc |= (VERSION & 0x3) << 46
    desc |= (BASE_OFFSET & 0x7) << 49
    desc |= (LBO_MODE & 0x1) << 52
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    return (start_addr.toint() & 0x3FFFF) >> 4


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    sA_swizzle = sA.iterator.type.swizzle_type
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        sA_swizzle,
        major,
    )
