# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from quack-kernels (https://github.com/Dao-AILab/quack);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def concat_to_interleave(a: cute.Tensor, dim: int) -> cute.Tensor:
    half = cute.size(a, mode=[dim]) // 2
    shape = (*a.shape[:dim], (2, half), *a.shape[dim + 1 :])
    stride = (*a.stride[:dim], (half * a.stride[dim], a.stride[dim]), *a.stride[dim + 1 :])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = cute.arch.prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = cute.arch.prmt(upper0, lower0, selector_lower)


@cute.jit
def permute_Cregs_b32_for_stsm(t: cute.Tensor) -> None:
    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b10

    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            left0 = left if quad_idx < 2 else right
            right0 = right if quad_idx < 2 else left
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            t[i * 4 + r * 2 + 0] = left0 if quad_idx % 2 == 0 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx % 2 == 0 else left0
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]


@cute.jit
def permute_Cregs_b32_for_ldsm(t: cute.Tensor) -> None:
    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b01

    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            left0 = left if quad_idx % 2 == 0 else right
            right0 = right if quad_idx % 2 == 0 else left
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            t[i * 4 + r * 2 + 0] = left0 if quad_idx < 2 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx < 2 else left0


@cute.jit
def concat_layout(*layouts: cute.Layout) -> cute.Layout:
    return cute.make_layout(
        tuple(l.shape for l in layouts),
        stride=tuple(l.stride for l in layouts),
    )


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(acc_layout, ((None, None, div), None, None))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:
        assert acc_layout.shape[2] % 2 == 0
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def convert_layout_zero_stride(
    input: cute.Tensor | cute.Layout, ref_layout: cute.Layout
) -> cute.Layout:
    layout = input.layout if const_expr(isinstance(input, cute.Tensor)) else input
    layout_flat = cute.flatten(layout)
    ref_layout_flat = cute.flatten(ref_layout)
    nonzero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride != 0]
    zero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride == 0]
    new_shape = (
        tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    out_layout = cute.make_layout(new_shape, stride=new_stride)
    if const_expr(isinstance(input, cute.Tensor)):
        return cute.make_tensor(input.iterator, out_layout)
    else:
        return out_layout


def mma_partition_C_vec(
    sVec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def mma_partition_A_vec(
    sVec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_A(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_S_vec(
    sVec: cute.Tensor, thr_copy: cute.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_S(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_D_vec(
    sVec: cute.Tensor, thr_copy: cute.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_D(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def tile_atom_to_shape_SF_strided(
    shape: cute.Shape,
    sf_vec_size: int,
    sf_strides,
) -> cute.Layout:
    from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk

    atom = BlockScaledBasicChunk(sf_vec_size).layout
    rmn = cute.ceil_div(shape[0], 128)
    rk = cute.ceil_div(shape[1], sf_vec_size * 4)
    outer = cute.make_layout((rmn, rk), stride=(sf_strides[1] // 512, sf_strides[2] // 512))
    sf_layout = cute.blocked_product(atom, outer)
    if const_expr(len(shape) == 3):
        sf_layout = cute.append(sf_layout, cute.make_layout(shape[2], stride=sf_strides[0]))
    return sf_layout
