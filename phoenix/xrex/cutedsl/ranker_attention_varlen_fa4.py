# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    block_q: int
    block_kv: int


BLACKWELL_CONFIG = TuningConfig(block_q=128, block_kv=128)


@functools.cache
def get_device_tuning_config() -> TuningConfig:
    return BLACKWELL_CONFIG


def _build_packed_block_sparse(
    hist_sizes,
    transformer_candidate_seq_len=128,
    block_size=128,
    max_hist_blocks=None,
    num_blocks=None,
    real_history_starts=None,
    num_user_prefix_tokens=0,
    cand_sizes=None,
    max_cand_blocks=None,
):
    hist_sizes = np.asarray(hist_sizes, dtype=np.int32)
    assert np.all(hist_sizes % block_size == 0), (
        f"hist_sizes must be multiples of block_size={block_size}, got {hist_sizes}"
    )
    assert transformer_candidate_seq_len % block_size == 0
    assert 0 <= num_user_prefix_tokens <= block_size
    if cand_sizes is None:
        cand_sizes = np.full_like(hist_sizes, transformer_candidate_seq_len)
    else:
        cand_sizes = np.asarray(cand_sizes, dtype=np.int32)
        assert cand_sizes.shape == hist_sizes.shape
        assert np.all(cand_sizes % block_size == 0), (
            f"cand_sizes must be multiples of block_size={block_size}, got {cand_sizes}"
        )
        assert np.all(cand_sizes <= transformer_candidate_seq_len)
    if real_history_starts is None:
        real_history_starts = np.zeros_like(hist_sizes)
        num_user_prefix_tokens = 0
    else:
        real_history_starts = np.asarray(real_history_starts, dtype=np.int32)
        assert real_history_starts.shape == hist_sizes.shape
        assert np.all(real_history_starts >= num_user_prefix_tokens)
        assert np.all(real_history_starts <= hist_sizes)

    h_blocks = (hist_sizes // block_size).tolist()
    c_blocks = (cand_sizes // block_size).tolist()
    actual_num_blocks = sum(hb + cb for hb, cb in zip(h_blocks, c_blocks))
    if num_blocks is None:
        num_blocks = actual_num_blocks
    else:
        assert num_blocks >= actual_num_blocks, (
            f"num_blocks={num_blocks} too small for hist_sizes (need {actual_num_blocks})"
        )

    actual_max_hist_blocks = max(h_blocks) if h_blocks else 1
    if max_hist_blocks is None:
        max_hist_blocks = actual_max_hist_blocks
    else:
        assert max_hist_blocks >= actual_max_hist_blocks, (
            f"max_hist_blocks={max_hist_blocks} too small (max h={actual_max_hist_blocks})"
        )
    max_hist_blocks = max(max_hist_blocks, 1)
    actual_max_cand_blocks = max(c_blocks) if c_blocks else 1
    if max_cand_blocks is None:
        max_cand_blocks = actual_max_cand_blocks
    else:
        assert max_cand_blocks >= actual_max_cand_blocks, (
            f"max_cand_blocks={max_cand_blocks} too small (max c={actual_max_cand_blocks})"
        )
    max_q_per_n = max_hist_blocks + max_cand_blocks

    valid_block_upper = np.zeros(num_blocks, dtype=np.int32)
    valid_block_lower = np.full(num_blocks, block_size, dtype=np.int32)
    is_partial_per_block = np.zeros(num_blocks, dtype=bool)
    is_empty_per_block = np.ones(num_blocks, dtype=bool)
    example_id_per_block = np.full(num_blocks, -1, dtype=np.int32)
    is_cand_per_block = np.zeros(num_blocks, dtype=bool)
    hist_block_start_per_user: list[int] = []

    cur = 0
    for user_idx, hb in enumerate(h_blocks):
        hist_block_start_per_user.append(cur)
        example_id_per_block[cur : cur + hb] = user_idx
        real_start = int(real_history_starts[user_idx])
        for local_block in range(hb):
            physical_block = cur + local_block
            tile_start = local_block * block_size
            upper = int(np.clip(num_user_prefix_tokens - tile_start, 0, block_size))
            lower = int(np.clip(real_start - tile_start, 0, block_size))
            if upper >= lower:
                upper = lower = 0
                is_empty = False
                is_partial = False
            elif upper == 0 and lower == block_size:
                is_empty = True
                is_partial = False
            else:
                is_empty = False
                is_partial = True
            valid_block_upper[physical_block] = upper
            valid_block_lower[physical_block] = lower
            is_empty_per_block[physical_block] = is_empty
            is_partial_per_block[physical_block] = is_partial
        cur += hb

        for j in range(c_blocks[user_idx]):
            physical_block = cur + j
            example_id_per_block[physical_block] = user_idx
            is_cand_per_block[physical_block] = True
            is_empty_per_block[physical_block] = False
            valid_block_upper[physical_block] = 0
            valid_block_lower[physical_block] = 0
        cur += c_blocks[user_idx]

    fwd_mask_cnt = np.zeros(num_blocks, dtype=np.int32)
    fwd_mask_idx = np.zeros((num_blocks, max_hist_blocks), dtype=np.int32)
    fwd_full_cnt = np.zeros(num_blocks, dtype=np.int32)
    fwd_full_idx = np.zeros((num_blocks, max_hist_blocks), dtype=np.int32)
    fwd_diag_cnt = np.zeros(num_blocks, dtype=np.int32)
    fwd_diag_idx = np.zeros((num_blocks, 1), dtype=np.int32)

    for m_block in range(num_blocks):
        user_idx = int(example_id_per_block[m_block])
        if user_idx < 0 or (not is_cand_per_block[m_block] and is_empty_per_block[m_block]):
            continue
        hstart = hist_block_start_per_user[user_idx]
        for n_block in range(hstart, hstart + h_blocks[user_idx]):
            if is_empty_per_block[n_block]:
                continue
            is_mask_edge = is_partial_per_block[m_block] or is_partial_per_block[n_block]
            if is_mask_edge:
                idx = int(fwd_mask_cnt[m_block])
                fwd_mask_idx[m_block, idx] = n_block
                fwd_mask_cnt[m_block] += 1
            else:
                idx = int(fwd_full_cnt[m_block])
                fwd_full_idx[m_block, idx] = n_block
                fwd_full_cnt[m_block] += 1
        if is_cand_per_block[m_block]:
            fwd_diag_cnt[m_block] = 1
            fwd_diag_idx[m_block, 0] = m_block

    bwd_mask_lists: list[list[int]] = [[] for _ in range(num_blocks)]
    bwd_full_lists: list[list[int]] = [[] for _ in range(num_blocks)]
    bwd_diag_lists: list[list[int]] = [[] for _ in range(num_blocks)]
    for m_block in range(num_blocks):
        for i in range(int(fwd_mask_cnt[m_block])):
            bwd_mask_lists[int(fwd_mask_idx[m_block, i])].append(m_block)
        for i in range(int(fwd_full_cnt[m_block])):
            bwd_full_lists[int(fwd_full_idx[m_block, i])].append(m_block)
        for i in range(int(fwd_diag_cnt[m_block])):
            bwd_diag_lists[int(fwd_diag_idx[m_block, i])].append(m_block)

    bwd_mask_cnt = np.asarray([len(xs) for xs in bwd_mask_lists], dtype=np.int32)
    bwd_mask_idx = np.zeros((num_blocks, max_q_per_n), dtype=np.int32)
    bwd_full_cnt = np.asarray([len(xs) for xs in bwd_full_lists], dtype=np.int32)
    bwd_full_idx = np.zeros((num_blocks, max_q_per_n), dtype=np.int32)
    bwd_diag_cnt = np.asarray([len(xs) for xs in bwd_diag_lists], dtype=np.int32)
    bwd_diag_idx = np.zeros((num_blocks, 1), dtype=np.int32)
    for n_block in range(num_blocks):
        bwd_mask_idx[n_block, : len(bwd_mask_lists[n_block])] = bwd_mask_lists[n_block]
        bwd_full_idx[n_block, : len(bwd_full_lists[n_block])] = bwd_full_lists[n_block]
        bwd_diag_idx[n_block, : len(bwd_diag_lists[n_block])] = bwd_diag_lists[n_block]

    return dict(
        total_seq_len=num_blocks * block_size,
        num_blocks=num_blocks,
        fwd=(fwd_mask_cnt, fwd_mask_idx, fwd_full_cnt, fwd_full_idx, fwd_diag_cnt, fwd_diag_idx),
        bwd=(bwd_mask_cnt, bwd_mask_idx, bwd_full_cnt, bwd_full_idx, bwd_diag_cnt, bwd_diag_idx),
        valid_block_upper=valid_block_upper,
        valid_block_lower=valid_block_lower,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class BlockSparseLayout:
    fwd_mask_cnt: np.ndarray
    fwd_mask_idx: np.ndarray
    fwd_full_cnt: np.ndarray
    fwd_full_idx: np.ndarray
    fwd_diag_cnt: np.ndarray
    fwd_diag_idx: np.ndarray
    bwd_mask_cnt: np.ndarray
    bwd_mask_idx: np.ndarray
    bwd_full_cnt: np.ndarray
    bwd_full_idx: np.ndarray
    bwd_diag_cnt: np.ndarray
    bwd_diag_idx: np.ndarray
    valid_block_upper: np.ndarray
    valid_block_lower: np.ndarray


def build_block_sparse_layout(
    cu_seqlens: np.ndarray,
    transformer_candidate_seq_len: int,
    max_history_seq_len: int,
    packed_seq_len: int | None = None,
    padding_mask: np.ndarray | None = None,
    num_user_prefix_tokens: int = 0,
    candidate_slot_lens: np.ndarray | None = None,
) -> BlockSparseLayout:
    per_user_lens = np.diff(cu_seqlens, axis=-1)
    block_size = get_device_tuning_config().block_q
    assert np.all(per_user_lens % block_size == 0), "All per-user lengths must be block-aligned"
    assert transformer_candidate_seq_len % block_size == 0, (
        "Candidate sequence length must be block-aligned"
    )
    assert 0 <= num_user_prefix_tokens <= block_size
    if candidate_slot_lens is not None:
        candidate_slot_lens = np.asarray(candidate_slot_lens, dtype=np.int32)
        assert candidate_slot_lens.shape == per_user_lens.shape

    num_devices, bs_per_device = per_user_lens.shape
    if padding_mask is not None:
        padding_mask = np.asarray(padding_mask, dtype=np.bool_)
        assert padding_mask.ndim == 2 and padding_mask.shape[0] == num_devices
        if packed_seq_len is not None:
            assert padding_mask.shape[1] == packed_seq_len

    if packed_seq_len is not None:
        assert packed_seq_len % block_size == 0, (
            f"packed_seq_len={packed_seq_len} must be a multiple of block_size={block_size}"
        )
        used = per_user_lens.sum(axis=-1)
        assert int(used.max()) <= packed_seq_len, (
            f"cu_seqlens claim more tokens than the physical row: {used.max()} > {packed_seq_len}"
        )
        total_blocks_per_device = packed_seq_len // block_size
    else:
        total_blocks_per_device = int(per_user_lens[0].sum()) // block_size
    max_hist_blocks = (max_history_seq_len + block_size - 1) // block_size

    fwd_arrays: list[list[np.ndarray]] = [[] for _ in range(6)]
    bwd_arrays: list[list[np.ndarray]] = [[] for _ in range(6)]
    valid_upper_arrays: list[np.ndarray] = []
    valid_lower_arrays: list[np.ndarray] = []
    max_cand_blocks = transformer_candidate_seq_len // block_size
    for d in range(num_devices):
        cand_sizes = (
            candidate_slot_lens[d]
            if candidate_slot_lens is not None
            else np.full(bs_per_device, transformer_candidate_seq_len, dtype=np.int32)
        )
        hist_sizes = per_user_lens[d] - cand_sizes
        if padding_mask is None:
            real_history_starts = None
        else:
            real_history_starts = np.zeros(bs_per_device, dtype=np.int32)
            for user_idx in range(bs_per_device):
                seq_start = int(cu_seqlens[d, user_idx])
                hist_size = int(hist_sizes[user_idx])
                hist_valid = padding_mask[
                    d,
                    seq_start + num_user_prefix_tokens : seq_start + hist_size,
                ]
                valid_offsets = np.flatnonzero(hist_valid)
                real_start = (
                    num_user_prefix_tokens + int(valid_offsets[0])
                    if valid_offsets.size
                    else hist_size
                )
                assert np.all(hist_valid[: real_start - num_user_prefix_tokens] == 0)
                assert np.all(hist_valid[real_start - num_user_prefix_tokens :] == 1)
                real_history_starts[user_idx] = real_start
        info = _build_packed_block_sparse(
            hist_sizes,
            transformer_candidate_seq_len=transformer_candidate_seq_len,
            max_hist_blocks=max_hist_blocks,
            num_blocks=total_blocks_per_device,
            real_history_starts=real_history_starts,
            num_user_prefix_tokens=num_user_prefix_tokens,
            cand_sizes=cand_sizes,
            max_cand_blocks=max_cand_blocks,
        )
        for i, arr in enumerate(info["fwd"]):
            fwd_arrays[i].append(arr)
        for i, arr in enumerate(info["bwd"]):
            bwd_arrays[i].append(arr)
        valid_upper_arrays.append(info["valid_block_upper"])
        valid_lower_arrays.append(info["valid_block_lower"])

    fwd = [np.stack(arrs, axis=0) for arrs in fwd_arrays]
    bwd = [np.stack(arrs, axis=0) for arrs in bwd_arrays]
    return BlockSparseLayout(
        fwd_mask_cnt=fwd[0],
        fwd_mask_idx=fwd[1],
        fwd_full_cnt=fwd[2],
        fwd_full_idx=fwd[3],
        fwd_diag_cnt=fwd[4],
        fwd_diag_idx=fwd[5],
        bwd_mask_cnt=bwd[0],
        bwd_mask_idx=bwd[1],
        bwd_full_cnt=bwd[2],
        bwd_full_idx=bwd[3],
        bwd_diag_cnt=bwd[4],
        bwd_diag_idx=bwd[5],
        valid_block_upper=np.stack(valid_upper_arrays, axis=0),
        valid_block_lower=np.stack(valid_lower_arrays, axis=0),
    )


_FA4_PACKED_CACHE = {}


def varlen_kernels(
    q,
    k,
    v,
    sm_scale,
    block_sparse_layout,
    valid_block_upper,
    valid_block_lower,
    cap=-1.0,
    cap_method="soft_sign",
):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.jax import cutlass_call

    from xrex.cutedsl.ranker_attention_fa4 import cutedsl_arch
    from xrex.cutedsl.ranker_fa4.block_sparsity import BlockSparseTensors
    from xrex.cutedsl.ranker_fa4.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
    from xrex.cutedsl.ranker_fa4.utils import create_cap_scoremods

    arch = cutedsl_arch()
    score_mod, score_mod_bwd = create_cap_scoremods(cap, cap_method)
    cap_key = (float(cap), cap_method) if score_mod is not None else None
    centered_stats = arch in (90, 100)
    batch_size, packed_S, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    qpk = num_q_heads // num_kv_heads
    block_size = 128
    hdr = ((head_dim + 31) // 32) * 32
    sr_q = ((packed_S + block_size - 1) // block_size) * block_size
    sr_k = sr_q
    qhead_loop = arch == 90 and qpk > 1
    dKV_postprocess = qpk > 1 and not qhead_loop
    if arch == 80:
        assert packed_S % block_size == 0, (
            "SM80 cutedsl ranker varlen attention requires packed_S % 128 == 0"
        )
        assert score_mod is None, "SM80 rejects score_mod with block sparsity; use cap <= 0"
        bwd_tile_m = 64
    elif arch == 90:
        bwd_tile_m = 64
    else:
        bwd_tile_m = block_size
    use_pack_gqa = qpk > 1 and (block_size % qpk == 0)

    fwd_bs, _ = block_sparse_layout
    if valid_block_upper is None or valid_block_lower is None:
        if valid_block_upper is not None or valid_block_lower is not None:
            raise ValueError("valid_block_upper and valid_block_lower must be provided together")
        valid_block_upper = jnp.zeros(fwd_bs[2].shape, dtype=jnp.int32)
        valid_block_lower = jnp.zeros(fwd_bs[2].shape, dtype=jnp.int32)
    valid_block_upper = jnp.broadcast_to(valid_block_upper, fwd_bs[2].shape)
    valid_block_lower = jnp.broadcast_to(valid_block_lower, fwd_bs[2].shape)
    bs_max_hist_blocks = int(fwd_bs[3].shape[-1])
    bs_num_blocks = int(fwd_bs[3].shape[-2])

    _expected_m_blocks = (packed_S + block_size - 1) // block_size
    if bs_num_blocks != _expected_m_blocks:
        raise ValueError(
            f"block-sparse arrays cover {bs_num_blocks} m-tiles but the kernel "
            f"will iterate {_expected_m_blocks} (packed_S={packed_S}). "
            "Pass packed_seq_len (the physical packed row length) to "
            "build_block_sparse_layout so every physical tile has an entry."
        )

    cache_key = (
        "packed",
        arch,
        head_dim,
        num_q_heads,
        num_kv_heads,
        batch_size,
        packed_S,
        bs_num_blocks,
        bs_max_hist_blocks,
        use_pack_gqa,
        cap_key,
    )

    if cache_key not in _FA4_PACKED_CACHE:
        if arch == 80:
            from xrex.cutedsl.ranker_fa4.flash_fwd import FlashAttentionForwardSm80

            fa_fwd = FlashAttentionForwardSm80(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qpk,
                is_causal=False,
                is_local=False,
                pack_gqa=False,
                tile_m=64,
                tile_n=block_size,
                num_stages=1,
                num_threads=128,
                Q_in_regs=False,
                q_subtile_factor=block_size // 64,
            )
        elif arch == 90:
            from xrex.cutedsl.ranker_fa4.flash_fwd_sm90 import FlashAttentionForwardSm90

            fa_fwd = FlashAttentionForwardSm90(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qpk,
                is_causal=False,
                is_local=False,
                pack_gqa=use_pack_gqa,
                tile_m=block_size,
                tile_n=block_size,
                num_stages=2,
                num_threads=384,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=None,
                score_mod=score_mod,
            )
        else:
            from xrex.cutedsl.ranker_fa4.flash_fwd_sm100 import FlashAttentionForwardSm100

            fa_fwd = FlashAttentionForwardSm100(
                head_dim=head_dim,
                head_dim_v=head_dim,
                qhead_per_kvhead=qpk,
                is_causal=False,
                is_local=False,
                pack_gqa=use_pack_gqa,
                is_persistent=True,
                mask_mod=None,
                score_mod=score_mod,
                q_stage=1,
            )
        q_shape = (batch_size, packed_S, num_q_heads, head_dim)
        k_shape = (batch_size, packed_S, num_kv_heads, head_dim)
        lse_shape = (batch_size, num_q_heads, packed_S)

        @cute.jit
        def launch_fwd(
            stream: cuda_driver.CUstream,
            mQ,
            mK,
            mV,
            mMaskCnt,
            mMaskIdx,
            mFullCnt,
            mFullIdx,
            mDiagCnt,
            mDiagIdx,
            mValidUpper,
            mValidLower,
            mO,
            mLSE,
            *softmax_stats,
            softmax_scale: cutlass.Float32,
        ):
            bs = BlockSparseTensors(
                mask_block_cnt=mMaskCnt,
                mask_block_idx=mMaskIdx,
                full_block_cnt=mFullCnt,
                full_block_idx=mFullIdx,
                cu_total_m_blocks=None,
                cu_block_idx_offsets=None,
                dq_write_order=None,
                dq_write_order_full=None,
                diag_block_cnt=mDiagCnt,
                diag_block_idx=mDiagIdx,
                dq_write_order_diag=None,
                valid_block_upper=mValidUpper,
                valid_block_lower=mValidLower,
            )
            if cutlass.const_expr(centered_stats):
                fa_fwd(
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    softmax_scale,
                    blocksparse_tensors=bs,
                    mSoftmaxStats=softmax_stats,
                    stream=stream,
                )
            else:
                fa_fwd(
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    softmax_scale,
                    blocksparse_tensors=bs,
                    stream=stream,
                )

        fwd_call = cutlass_call(
            launch_fwd,
            output_shape_dtype=[
                jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            ]
            + ([jax.ShapeDtypeStruct(lse_shape, jnp.float32)] * 2 if centered_stats else []),
            use_static_tensors=False,
            softmax_scale=cutlass.Float32(sm_scale),
        )

        if arch == 80:
            from xrex.cutedsl.ranker_fa4.flash_bwd import FlashAttentionBackwardSm80

            bwd_atom_layout_dkv = 2
            fa_bwd = FlashAttentionBackwardSm80(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qpk,
                m_block_size=bwd_tile_m,
                n_block_size=block_size,
                num_stages_Q=2,
                num_stages_dO=2,
                num_threads=256,
                pack_gqa=False,
                is_causal=False,
                SdP_swapAB=False,
                dKV_swapAB=False,
                dQ_swapAB=False,
                AtomLayoutMSdP=2,
                AtomLayoutNdKV=bwd_atom_layout_dkv,
                AtomLayoutMdQ=2,
                V_in_regs=False,
                q_subtile_factor=block_size // bwd_tile_m,
            )
            post_threads = 256
            post_dq_atom_layout = 2
        elif arch == 90:
            from xrex.cutedsl.ranker_fa4.flash_bwd_sm90 import FlashAttentionBackwardSm90

            bwd_atom_layout_dkv = 2
            fa_bwd = FlashAttentionBackwardSm90(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qpk,
                False,
                is_local=False,
                deterministic=False,
                tile_m=bwd_tile_m,
                tile_n=block_size,
                Q_stage=2,
                dO_stage=2,
                PdS_stage=2,
                SdP_swapAB=True,
                dKV_swapAB=False,
                dQ_swapAB=False,
                AtomLayoutMSdP=1,
                AtomLayoutNdKV=bwd_atom_layout_dkv,
                AtomLayoutMdQ=1,
                num_threads=384,
                mask_mod=None,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                subtile_factor=block_size // bwd_tile_m,
                qhead_loop=qhead_loop,
            )
            post_threads = 256
            post_dq_atom_layout = 1
        else:
            from xrex.cutedsl.ranker_fa4.flash_bwd_sm100 import FlashAttentionBackwardSm100

            bwd_atom_layout_dkv = 1
            fa_bwd = FlashAttentionBackwardSm100(
                head_dim=head_dim,
                head_dim_v=head_dim,
                qhead_per_kvhead=qpk,
                is_causal=False,
                is_local=False,
                mask_mod=None,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
            post_threads = 128
            post_dq_atom_layout = 1
        dq_accum_shape = (batch_size, num_q_heads, sr_q * hdr)
        dk_accum_shape = (batch_size, num_kv_heads, sr_k * hdr)
        dkv_out = jax.ShapeDtypeStruct(
            dk_accum_shape if dKV_postprocess else k_shape,
            jnp.float32 if dKV_postprocess else jnp.bfloat16,
        )
        bwd_accumulators = [jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32), dkv_out, dkv_out]

        @cute.jit
        def launch_bwd(
            stream: cuda_driver.CUstream,
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mMaskCnt,
            mMaskIdx,
            mFullCnt,
            mFullIdx,
            mDiagCnt,
            mDiagIdx,
            mValidUpper,
            mValidLower,
            *rest,
            softmax_scale: cutlass.Float32,
        ):
            mdQa, mdKa, mdVa = rest[-3], rest[-2], rest[-1]
            bs = BlockSparseTensors(
                mask_block_cnt=mMaskCnt,
                mask_block_idx=mMaskIdx,
                full_block_cnt=mFullCnt,
                full_block_idx=mFullIdx,
                cu_total_m_blocks=None,
                cu_block_idx_offsets=None,
                dq_write_order=None,
                dq_write_order_full=None,
                diag_block_cnt=mDiagCnt,
                diag_block_idx=mDiagIdx,
                dq_write_order_diag=None,
                valid_block_upper=mValidUpper,
                valid_block_lower=mValidLower,
            )
            if cutlass.const_expr(centered_stats):
                fa_bwd(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    mdQa,
                    mdKa,
                    mdVa,
                    softmax_scale,
                    blocksparse_tensors=bs,
                    mSoftmaxStats=(rest[0], rest[1]),
                    stream=stream,
                )
            else:
                fa_bwd(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    mdQa,
                    mdKa,
                    mdVa,
                    softmax_scale,
                    blocksparse_tensors=bs,
                    stream=stream,
                )

        bwd_call = cutlass_call(
            launch_bwd,
            output_shape_dtype=bwd_accumulators,
            input_output_aliases=(
                {16: 0, 17: 1, 18: 2} if centered_stats else {14: 0, 15: 1, 16: 2}
            ),
            use_static_tensors=False,
            softmax_scale=cutlass.Float32(sm_scale),
        )

        fa_post_dq = FlashAttentionBackwardPostprocess(
            cutlass.BFloat16,
            head_dim,
            arch,
            tile_m=bwd_tile_m,
            num_threads=post_threads,
            AtomLayoutMdQ=post_dq_atom_layout,
        )

        @cute.jit
        def launch_post_dq(stream, mAccum, mOut, scale: cutlass.Float32):
            fa_post_dq(mAccum, mOut, scale, None, None, stream)

        post_dq_call = cutlass_call(
            launch_post_dq,
            output_shape_dtype=[jax.ShapeDtypeStruct(q_shape, jnp.bfloat16)],
            use_static_tensors=False,
            scale=cutlass.Float32(sm_scale),
        )

        post_dk_call = post_dv_call = None
        if dKV_postprocess:
            fa_post_dk = FlashAttentionBackwardPostprocess(
                cutlass.BFloat16,
                head_dim,
                arch,
                tile_m=block_size,
                num_threads=post_threads,
                AtomLayoutMdQ=bwd_atom_layout_dkv,
            )

            @cute.jit
            def launch_post_dk(stream, mAccum, mOut, scale: cutlass.Float32):
                fa_post_dk(mAccum, mOut, scale, None, None, stream)

            post_dk_call = cutlass_call(
                launch_post_dk,
                output_shape_dtype=[jax.ShapeDtypeStruct(k_shape, jnp.bfloat16)],
                use_static_tensors=False,
                scale=cutlass.Float32(sm_scale),
            )

            fa_post_dv = FlashAttentionBackwardPostprocess(
                cutlass.BFloat16,
                head_dim,
                arch,
                tile_m=block_size,
                num_threads=post_threads,
                AtomLayoutMdQ=bwd_atom_layout_dkv,
            )

            @cute.jit
            def launch_post_dv(stream, mAccum, mOut, scale: cutlass.Float32):
                fa_post_dv(mAccum, mOut, scale, None, None, stream)

            post_dv_call = cutlass_call(
                launch_post_dv,
                output_shape_dtype=[jax.ShapeDtypeStruct(k_shape, jnp.bfloat16)],
                use_static_tensors=False,
                scale=cutlass.Float32(1.0),
            )

        _FA4_PACKED_CACHE[cache_key] = dict(
            fwd_call=fwd_call,
            bwd_call=bwd_call,
            post_dq_call=post_dq_call,
            post_dk_call=post_dk_call,
            post_dv_call=post_dv_call,
            dKV_postprocess=dKV_postprocess,
            bwd_accumulators=bwd_accumulators,
            sr_q=sr_q,
            hdr=hdr,
        )

    fwd_bs, bwd_bs = block_sparse_layout
    return _FA4_PACKED_CACHE[cache_key], (
        *fwd_bs,
        *bwd_bs,
        valid_block_upper,
        valid_block_lower,
    )


def zeros_after(anchor: jax.Array, *structs: jax.ShapeDtypeStruct) -> list[jax.Array]:
    zeros = []
    for i, struct in enumerate(structs):
        zero, _ = jax.lax.optimization_barrier((jnp.zeros(i + 1, struct.dtype), anchor))
        zeros.append(jnp.broadcast_to(zero[0], struct.shape))
    return zeros


def varlen_forward(kernels, q, k, v, bs_args):
    out, lse, *softmax_stats = kernels["fwd_call"](q, k, v, *bs_args[:6], *bs_args[12:])
    return out, lse, tuple(softmax_stats)


def varlen_backward(kernels, q, k, v, out, lse, softmax_stats, bs_args, g):
    from xrex.cutedsl.ranker_attention_fa4 import attention_delta, pad_softmax_stats

    bbs = bs_args[6:12]
    valid_bounds = bs_args[12:]
    dpsum = attention_delta(out, g, kernels["sr_q"])
    softmax_stats = pad_softmax_stats(softmax_stats, kernels["sr_q"])
    lse_log2 = lse * jnp.float32(math.log2(math.e))
    if lse_log2.shape[-1] < kernels["sr_q"]:
        lse_log2 = jnp.pad(lse_log2, ((0, 0), (0, 0), (0, kernels["sr_q"] - lse_log2.shape[-1])))

    dq_accum_init, dk_init, dv_init = zeros_after(g, *kernels["bwd_accumulators"])
    dq_accum, dk, dv = kernels["bwd_call"](
        q,
        k,
        v,
        g,
        lse_log2,
        dpsum,
        *bbs,
        *valid_bounds,
        *softmax_stats,
        dq_accum_init,
        dk_init,
        dv_init,
    )
    (dq,) = kernels["post_dq_call"](dq_accum)
    if kernels["dKV_postprocess"]:
        (dk,) = kernels["post_dk_call"](dk)
        (dv,) = kernels["post_dv_call"](dv)
    return dq, dk, dv


def ranker_attention_varlen_fa4(
    q,
    k,
    v,
    sm_scale,
    block_sparse_layout,
    valid_block_upper=None,
    valid_block_lower=None,
    cap=-1.0,
    cap_method="soft_sign",
):
    kernels, bs_args = varlen_kernels(
        q,
        k,
        v,
        sm_scale,
        block_sparse_layout,
        valid_block_upper,
        valid_block_lower,
        cap=cap,
        cap_method=cap_method,
    )

    @jax.custom_vjp
    def _attention(q, k, v, *bs_args):
        return varlen_forward(kernels, q, k, v, bs_args)[0]

    def _attention_fwd(q, k, v, *bs_args):
        out, lse, softmax_stats = varlen_forward(kernels, q, k, v, bs_args)
        out = checkpoint_name(out, "cutedsl_attn_outputs")
        lse = checkpoint_name(lse, "cutedsl_attn_outputs")
        softmax_stats = tuple(checkpoint_name(t, "cutedsl_attn_outputs") for t in softmax_stats)
        return out, (q, k, v, out, lse, softmax_stats, bs_args)

    def _attention_bwd(res, g):
        q, k, v, out, lse, softmax_stats, bs_args = res
        return varlen_backward(kernels, q, k, v, out, lse, softmax_stats, bs_args, g) + (
            None,
        ) * len(bs_args)

    _attention.defvjp(_attention_fwd, _attention_bwd)
    return _attention(q, k, v, *bs_args)
