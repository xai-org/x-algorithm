# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from xrex.pallas import ranker_attention_utils

HISTORY_SEGMENT_ID = 1
CANDIDATE_SEGMENT_ID = -1
PADDING_SEGMENT_ID = 0


def tanh(x):
    return 2 * jax.lax.logistic(2 * x) - 1


def _abs(x):
    return jnp.where(x >= 0, x, -x)


def _apply_cap(qk, cap: float, cap_method: str):
    if cap <= 0.0 or cap_method == "none":
        return qk, None, None
    if cap_method == "tanh":
        qk_tanh = tanh(qk / cap)
        return cap * qk_tanh, qk_tanh, None
    if cap_method == "soft_sign":
        soft_sign = 1.0 / (1.0 + _abs(qk) / cap)
        return qk * soft_sign, None, soft_sign
    raise ValueError(f"cap_method must be in [tanh, soft_sign, none], got {cap_method}")


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    block_q: int
    block_kv: int
    max_concurrent_steps: int
    use_schedule_barrier: bool = True
    causal: bool = False
    compute_wgs_bwd: int = 1

    block_q_dkv: int | None = None
    block_kv_dkv: int | None = None
    block_q_dq: int | None = None
    block_kv_dq: int | None = None

    def __post_init__(self):
        if self.block_q % 64:
            raise ValueError(f"{self.block_q=} must be a multiple of 64")
        if self.block_kv % 64:
            raise ValueError(f"{self.block_kv=} must be a multiple of 64")
        if self.max_concurrent_steps < 2:
            raise ValueError(f"{self.max_concurrent_steps=} must be at least 2")

        backward_blocks = [self.block_q_dkv, self.block_kv_dkv, self.block_q_dq, self.block_kv_dq]
        block_is_set = [blk is not None for blk in backward_blocks]
        if any(block_is_set) and not all(block_is_set):
            raise ValueError(
                "Backward block sizes (block_q_dkv, block_kv_dkv, block_q_dq, "
                "block_kv_dq) must either all be specified or all be None."
            )

    @property
    def has_backward_blocks(self) -> bool:
        return self.block_q_dkv is not None


def _attention_forward(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals: bool = False,
    bound=None,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
    batch_size, q_seq_len, num_q_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    if k.shape != kv_shape:
        raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
    if v.shape != kv_shape:
        raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
    if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
        raise ValueError(
            f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    if head_dim % 64:
        raise ValueError(f"{head_dim=} must be divisible by 64")
    if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
        raise NotImplementedError(f"Only f16 and bf16 are supported, got dtype: {dtype}")

    max_concurrent_steps = min(config.max_concurrent_steps, kv_seq_len // config.block_kv)
    block_q, block_kv = config.block_q, config.block_kv
    if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

    def kernel(q_ref, k_ref, v_ref, bound_ref, out_ref, lse_ref, scoped):
        batch = lax.axis_index("batch")
        q_head = lax.axis_index("heads")
        q_seq = lax.axis_index("q_seq")
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        wg_idx = lax.axis_index("wg")
        qo_smem2, k_smem, v_smem, lse_smem2 = smem_buffers
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed_barriers, v_consumed_barriers = consumed_barriers
        history_lower_bound = plgpu.load(bound_ref, (batch, 0))
        history_upper_bound = plgpu.load(bound_ref, (batch, 1))
        candidate_lower_bound = plgpu.load(bound_ref, (batch, 2))
        candidate_upper_bound = plgpu.load(bound_ref, (batch, 3))

        def perform_schedule_barrier():
            plgpu.barrier_arrive(schedule_barrier)
            plgpu.barrier_wait(schedule_barrier)

        q_tile_base = q_seq * (2 * block_q)
        q_tile_end = q_tile_base + (2 * block_q)

        def tile_has_tokens(lower, upper) -> bool:
            return (q_tile_base < upper) & (q_tile_end > lower)

        valid_program = tile_has_tokens(history_lower_bound, history_upper_bound) | tile_has_tokens(
            candidate_lower_bound, candidate_upper_bound
        )

        hist_k_start = lax.div(history_lower_bound, block_kv)
        hist_k_end = pl.cdiv(history_upper_bound, block_kv)
        hist_steps = jnp.maximum(hist_k_end - hist_k_start, 0)

        cand_start = jnp.maximum(candidate_lower_bound, q_tile_base)
        cand_end = jnp.minimum(candidate_upper_bound, q_tile_end)
        cand_has = cand_start < cand_end
        cand_k_start = lax.div(cand_start, block_kv)
        cand_k_end = pl.cdiv(cand_end, block_kv)
        cand_steps = jnp.where(cand_has, cand_k_end - cand_k_start, 0)

        block_max_kv_steps = hist_steps + cand_steps

        @pl.when((wg_idx < 2) & (~valid_program))
        def _zero_out_wg():
            qo_smem = qo_smem2.at[wg_idx]
            lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
            q_seq_base = q_seq * (2 * block_q) + wg_idx * block_q
            zero_acc = plgpu.layout_cast(
                jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            qo_smem[...] = zero_acc.astype(dtype)
            if lse_smem is not None:
                lse_zero = plgpu.layout_cast(
                    jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                    plgpu.Layout.WGMMA_ROW,
                )
                lse_smem[...] = lse_zero
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                qo_smem,
                out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
            )
            if lse_smem is not None:
                plgpu.copy_smem_to_gmem(
                    lse_smem,
                    lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
                )
            plgpu.wait_smem_to_gmem(0)

        @pl.when((wg_idx < 2) & valid_program)
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")
            qo_smem = qo_smem2.at[wg_idx]
            lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
            q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q

            kv_steps = block_max_kv_steps

            plgpu.copy_gmem_to_smem(
                q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
                qo_smem,
                q_barriers.at[wg_idx],
            )
            plgpu.barrier_wait(q_barriers.at[wg_idx])

            m_i = plgpu.layout_cast(
                jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            l_i = plgpu.layout_cast(
                jnp.full((block_q,), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            acc = plgpu.layout_cast(
                jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )

            @pl.when(kv_steps > 0)
            def _():
                plgpu.barrier_wait(k_barriers.at[0])

            pl.when(wg_idx == 1)(perform_schedule_barrier)

            def kv_loop(kv_step, carry):
                acc, m_i, l_i = carry
                slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                kv_block_idx = ranker_attention_utils.kv_block_for_step(
                    kv_step,
                    hist_k_start,
                    hist_steps,
                    cand_k_start,
                )

                def compute_qk(acc_ref):
                    plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    perform_schedule_barrier()
                    return acc_ref[...]

                qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
                plgpu.barrier_arrive(k_consumed_barriers.at[slot])

                if sm_scale != 1.0:
                    qk *= sm_scale
                qk, _, _ = _apply_cap(qk, cap, cap_method)

                ranker_attention_mask = ranker_attention_utils.history_candidate_mask(
                    q_seq_base,
                    block_q,
                    kv_block_idx * block_kv,
                    block_kv,
                    history_lower_bound,
                    history_upper_bound,
                    candidate_lower_bound,
                    candidate_upper_bound,
                    broadcasted_iota_fn=plgpu.broadcasted_iota,
                    layout=plgpu.Layout.WGMMA,
                )
                qk = jnp.where(ranker_attention_mask, qk, -jnp.inf)

                log2e = math.log2(math.e)
                m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
                alpha = jnp.exp2(m_i - m_ij)
                m_i = m_ij
                p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i *= alpha
                p16 = p.astype(dtype)

                def end_softmax_barriers():
                    plgpu.barrier_arrive(schedule_barrier)
                    plgpu.barrier_wait(v_barriers.at[slot])
                    plgpu.barrier_wait(schedule_barrier)

                if head_dim <= 128:
                    l_i += p.sum(axis=1)
                    acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                    end_softmax_barriers()
                else:
                    end_softmax_barriers()
                    l_i += p.sum(axis=1)

                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])

                    wait_step = kv_step + 1
                    wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                    @pl.when(wait_step < kv_steps)
                    def _wait():
                        plgpu.barrier_wait(k_barriers.at[wait_slot])

                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_consumed_barriers.at[slot])
                return acc, m_i, l_i

            acc, m_i, l_i = lax.fori_loop(0, block_max_kv_steps, kv_loop, (acc, m_i, l_i))
            pl.when(wg_idx == 0)(perform_schedule_barrier)

            acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
            qo_smem[...] = acc.astype(dtype)
            if lse_smem is not None:
                RCP_LN2 = 1.4426950408889634
                log2 = lambda x: jnp.log(x) * RCP_LN2
                lse_smem[...] = m_i + log2(l_i)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                qo_smem,
                out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
            )
            if lse_smem is not None:
                plgpu.copy_smem_to_gmem(
                    lse_smem,
                    lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
                )
            plgpu.wait_smem_to_gmem(0)

        @pl.when((wg_idx == 2) & valid_program)
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")
            kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
            for i in range(max_concurrent_steps):

                @pl.when(i < block_max_kv_steps)
                def _prefetch(i=i):
                    kv_block_idx = ranker_attention_utils.kv_block_for_step(
                        jnp.array(i, jnp.int32),
                        hist_k_start,
                        hist_steps,
                        cand_k_start,
                    )
                    s = (batch, pl.ds(kv_block_idx * block_kv, block_kv), kv_head)
                    plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
                    plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])

            @pl.loop(0, jnp.maximum(block_max_kv_steps - max_concurrent_steps, 0))
            def _kv_loop(kv_step):
                tma_step = kv_step + max_concurrent_steps
                tma_slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                kv_block_idx = ranker_attention_utils.kv_block_for_step(
                    tma_step,
                    hist_k_start,
                    hist_steps,
                    cand_k_start,
                )
                s = (batch, pl.ds(kv_block_idx * block_kv, block_kv), kv_head)
                plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
                plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])

    def entry(q_ref, k_ref, v_ref, bound_ref, out_ref, lse_ref):
        compute_wgs = 2
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)
        qo_scratch = plgpu.SMEM(
            (compute_wgs, block_q, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )
        k_scratch = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )
        v_scratch = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )
        scratch = [qo_scratch, k_scratch, v_scratch, None]
        if save_residuals:
            scratch[3] = plgpu.SMEM((compute_wgs, block_q), jnp.float32)

        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, bound_ref, out_ref, lse_ref, args),
            scratch,
            (
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=compute_wgs),
            ),
            (plgpu.Barrier(num_arrivals=compute_wgs, num_barriers=max_concurrent_steps),) * 2,
            plgpu.Barrier(num_arrivals=compute_wgs),
            collective_axes="wg",
        )

    num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
    if rem:
        raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")

    out_shape = [q, None]
    if save_residuals:
        out_shape[1] = jax.ShapeDtypeStruct((batch_size, num_q_heads, q_seq_len), jnp.float32)
    out, lse = plgpu.kernel(
        entry,
        out_shape=out_shape,
        grid=(num_q_heads, num_q_tiles, batch_size),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, bound)

    if save_residuals:
        assert lse is not None
        return out, (lse,)

    return out


def _normalize_bound(bound, batch_size: int, seq_len: int):
    if bound is None:
        bound = (0, seq_len + 1, seq_len + 1, seq_len + 1)
    bound_arr = jnp.asarray(bound, dtype=jnp.int32)
    if bound_arr.shape != (4,):
        raise ValueError(f"Expected bound to have shape (4,), got {bound_arr.shape}")
    return jnp.broadcast_to(bound_arr, (batch_size, 4))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
@functools.partial(
    jax.jit,
    static_argnames=[
        "config",
        "save_residuals",
        "bound",
        "sm_scale",
        "cap",
        "cap_method",
        "z_loss_weight",
    ],
)
def attention(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals: bool = False,
    bound: tuple[int, int, int, int] | None = None,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
    z_loss_weight: float = 0.0,
):
    batch_size, q_seq_len, _, _ = q.shape
    bound_arr = _normalize_bound(bound, batch_size, q_seq_len)
    return _attention_forward(
        q,
        k,
        v,
        config,
        save_residuals,
        bound_arr,
        sm_scale=sm_scale,
        cap=cap,
        cap_method=cap_method,
    )


def _attention_fwd(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals: bool,
    bound,
    sm_scale: float,
    cap: float,
    cap_method: str,
    z_loss_weight: float,
):
    del save_residuals, z_loss_weight

    batch_size, q_seq_len, _, _ = q.shape
    bound_arr = _normalize_bound(bound, batch_size, q_seq_len)
    out, (lse,) = _attention_forward(
        q,
        k,
        v,
        config,
        save_residuals=True,
        bound=bound_arr,
        sm_scale=sm_scale,
        cap=cap,
        cap_method=cap_method,
    )
    return out, (q, k, v, out, lse)


def _attention_bwd(
    config: TuningConfig,
    save_residuals: bool,
    bound,
    sm_scale: float,
    cap: float,
    cap_method: str,
    z_loss_weight: float,
    res,
    do,
):
    del save_residuals
    q, k, v, out, lse = res

    if config.causal:
        raise NotImplementedError("Causal attention not supported in the backwards pass yet.")

    if not config.has_backward_blocks:
        raise ValueError("Need to specify backward blocks.")

    assert config.block_q_dq is not None
    assert config.block_kv_dq is not None
    assert config.block_q_dkv is not None
    assert config.block_kv_dkv is not None

    batch_size, q_seq_len, num_q_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    dtype = q.dtype
    compute_wgs = config.compute_wgs_bwd

    num_q_tiles, rem = divmod(q_seq_len, config.block_q_dq * compute_wgs)
    if rem:
        raise NotImplementedError(
            f"{q_seq_len=} must be a multiple of {config.block_q_dq=} * {compute_wgs=}"
        )

    num_kv_tiles, rem = divmod(kv_seq_len, config.block_kv_dkv * compute_wgs)
    if rem:
        raise NotImplementedError(
            f"{kv_seq_len=} must be a multiple of {config.block_kv_dkv=} * {compute_wgs=}"
        )

    num_q_tiles_in_dkv, rem = divmod(q_seq_len, config.block_q_dkv)
    if rem:
        raise NotImplementedError(f"{q_seq_len=} must be a multiple of {config.block_q_dkv=}")

    num_kv_tiles_in_dq, rem = divmod(kv_seq_len, config.block_kv_dq)
    if rem:
        raise NotImplementedError(f"{kv_seq_len=} must be a multiple of {config.block_kv_dq=}")

    bound_arr = _normalize_bound(bound, batch_size, q_seq_len)
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)

    delta = jnp.einsum("bqhd,bqhd->bhq", out.astype(jnp.float32), do.astype(jnp.float32))
    del out

    def kernel_dq(
        q_ref,
        k_ref,
        v_ref,
        do_ref,
        lse_ref,
        delta_ref,
        bound_ref,
        dq_ref,
        smem_buffers,
        buffer_barriers,
        block_q,
        block_kv,
    ):
        batch = lax.axis_index("batch")
        q_head = lax.axis_index("heads")
        wg_idx = lax.axis_index("wg")
        kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
        q_seq = lax.axis_index("q_seq")
        q_smem2, do_smem2, lse_smem2, delta_smem2 = smem_buffers
        q_barriers, do_barriers, lse_barriers, delta_barriers = buffer_barriers
        history_lower_bound = plgpu.load(bound_ref, (batch, 0))
        history_upper_bound = plgpu.load(bound_ref, (batch, 1))
        candidate_lower_bound = plgpu.load(bound_ref, (batch, 2))
        candidate_upper_bound = plgpu.load(bound_ref, (batch, 3))
        q_tile_base = q_seq * (compute_wgs * block_q)
        q_tile_end = q_tile_base + (compute_wgs * block_q)
        hist_k_start = lax.div(history_lower_bound, block_kv)
        hist_k_end = pl.cdiv(history_upper_bound, block_kv)
        hist_steps = jnp.maximum(hist_k_end - hist_k_start, 0)
        cand_start = jnp.maximum(candidate_lower_bound, q_tile_base)
        cand_end = jnp.minimum(candidate_upper_bound, q_tile_end)
        cand_has = cand_start < cand_end
        cand_k_start = lax.div(cand_start, block_kv)
        cand_k_end = pl.cdiv(cand_end, block_kv)
        cand_steps = jnp.where(cand_has, cand_k_end - cand_k_start, 0)
        block_max_kv_steps = hist_steps + cand_steps
        q_seq_base = q_seq * (compute_wgs * block_q) + wg_idx * block_q

        def _compute_thread(pipeline_callback):
            q_smem, do_smem, lse_smem, delta_smem = (
                q_smem2.at[wg_idx],
                do_smem2.at[wg_idx],
                lse_smem2.at[wg_idx],
                delta_smem2.at[wg_idx],
            )
            q_slice = (batch, pl.ds(q_seq_base, block_q), q_head)
            plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, q_barriers.at[wg_idx])
            plgpu.copy_gmem_to_smem(do_ref.at[q_slice], do_smem, do_barriers.at[wg_idx])
            plgpu.copy_gmem_to_smem(
                delta_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
                delta_smem,
                delta_barriers.at[wg_idx],
            )
            plgpu.copy_gmem_to_smem(
                lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
                lse_smem,
                lse_barriers.at[wg_idx],
            )
            for buffer in buffer_barriers:
                plgpu.barrier_wait(buffer.at[wg_idx])

            delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_ROW)
            lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_ROW)
            dq_acc = plgpu.layout_cast(
                jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            dq, _, _ = pipeline_callback((dq_acc, lse, delta))
            q_smem[...] = dq.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
            plgpu.wait_smem_to_gmem(0)

        def kv_pipeline(kv_step, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry):
            kv_step = kv_step[0]
            q_smem, do_smem = q_smem2.at[wg_idx], do_smem2.at[wg_idx]
            (dq_acc, lse, delta) = carry
            kv_block_idx = ranker_attention_utils.kv_block_for_step(
                kv_step,
                hist_k_start,
                hist_steps,
                cand_k_start,
            )

            def compute_s(acc_ref):
                plgpu.wgmma(acc_ref, q_smem, plgpu.transpose_ref(k_smem, (1, 0)))
                return acc_ref[...]

            s = pl.run_scoped(compute_s, plgpu.ACC((block_q, block_kv), jnp.float32))
            if sm_scale != 1.0:
                s *= sm_scale
            s, qk_tanh, soft_sign = _apply_cap(s, cap, cap_method)
            s *= math.log2(math.e)
            p = jnp.exp2(s - lax.broadcast_in_dim(lse, (block_q, block_kv), [0]))
            kv_seq_base = kv_block_idx * block_kv
            ranker_attention_mask = ranker_attention_utils.history_candidate_mask(
                q_seq_base,
                block_q,
                kv_seq_base,
                block_kv,
                history_lower_bound,
                history_upper_bound,
                candidate_lower_bound,
                candidate_upper_bound,
                broadcasted_iota_fn=plgpu.broadcasted_iota,
                layout=plgpu.Layout.WGMMA,
            )
            p = jnp.where(ranker_attention_mask, p, 0)

            def compute_dp(acc_ref):
                plgpu.wgmma(acc_ref, do_smem, plgpu.transpose_ref(v_smem, (1, 0)))
                return acc_ref[...]

            dp = pl.run_scoped(compute_dp, plgpu.ACC((block_q, block_kv), jnp.float32))
            plgpu.barrier_arrive(v_consumed_barrier)

            ds = p * (dp - lax.broadcast_in_dim(delta, (block_q, block_kv), [0]))
            if z_loss_weight > 0.0:
                logsumexp = lse * math.log(2.0)
                ds += p * lax.broadcast_in_dim(z_loss_weight * logsumexp, (block_q, block_kv), [0])
            if cap > 0.0 and cap_method != "none":
                if cap_method == "tanh":
                    ds = ds * (1 - qk_tanh**2)
                elif cap_method == "soft_sign":
                    ds = ds * soft_sign**2
                else:
                    raise ValueError(
                        f"cap_method must be in [tanh, soft_sign, none], got {cap_method}"
                    )
            if sm_scale != 1.0:
                ds = ds * sm_scale

            def compute_dq(acc_ref):
                plgpu.wgmma(acc_ref, ds.astype(k_ref.dtype), k_smem)

            dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
            plgpu.barrier_arrive(k_consumed_barrier)

            return (dq_acc, lse, delta)

        pipeline = plgpu.emit_pipeline_warp_specialized(
            kv_pipeline,
            grid=(block_max_kv_steps,),
            max_concurrent_steps=min([config.max_concurrent_steps, num_q_tiles]),
            num_compute_wgs=compute_wgs,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=_compute_thread,
            in_specs=[
                plgpu.BlockSpec(
                    block_shape=(block_kv, head_dim),
                    index_map=lambda i: (
                        ranker_attention_utils.kv_block_for_step(
                            i,
                            hist_k_start,
                            hist_steps,
                            cand_k_start,
                        ),
                        0,
                    ),
                    transforms=[tiling, swizzle],
                ),
                plgpu.BlockSpec(
                    block_shape=(block_kv, head_dim),
                    index_map=lambda i: (
                        ranker_attention_utils.kv_block_for_step(
                            i,
                            hist_k_start,
                            hist_steps,
                            cand_k_start,
                        ),
                        0,
                    ),
                    transforms=[tiling, swizzle],
                ),
            ],
        )
        k_ref = k_ref.at[batch, :, kv_head, :]
        v_ref = v_ref.at[batch, :, kv_head, :]
        pipeline(k_ref, v_ref)

    def kernel_dkv(
        q_ref,
        k_ref,
        v_ref,
        do_ref,
        lse_ref,
        delta_ref,
        bound_ref,
        dk_ref,
        dv_ref,
        smem_buffers,
        buffer_barriers,
        block_q: int,
        block_kv: int,
    ):
        batch = lax.axis_index("batch")
        q_head = lax.axis_index("heads")
        wg_idx = lax.axis_index("wg")
        kv_seq = lax.axis_index("kv_seq")
        (k_smem2, v_smem2) = smem_buffers
        (k_barriers, v_barriers) = buffer_barriers
        history_lower_bound = plgpu.load(bound_ref, (batch, 0))
        history_upper_bound = plgpu.load(bound_ref, (batch, 1))
        candidate_lower_bound = plgpu.load(bound_ref, (batch, 2))
        candidate_upper_bound = plgpu.load(bound_ref, (batch, 3))
        kv_group_base = kv_seq * (compute_wgs * block_kv)
        kv_group_end = kv_group_base + (compute_wgs * block_kv)
        kv_group_is_hist = (kv_group_base < history_upper_bound) & (
            kv_group_end > history_lower_bound
        )
        kv_group_is_cand = (kv_group_base < candidate_upper_bound) & (
            kv_group_end > candidate_lower_bound
        )
        hist_q_start = lax.div(history_lower_bound, block_q)
        hist_q_end = pl.cdiv(history_upper_bound, block_q)
        hist_q_steps = jnp.maximum(hist_q_end - hist_q_start, 0)
        cand_has = candidate_lower_bound < candidate_upper_bound
        cand_q_start = lax.div(candidate_lower_bound, block_q)
        cand_q_end = pl.cdiv(candidate_upper_bound, block_q)
        cand_q_steps = jnp.where(cand_has, cand_q_end - cand_q_start, 0)
        cand_overlap_start = jnp.maximum(candidate_lower_bound, kv_group_base)
        cand_overlap_end = jnp.minimum(candidate_upper_bound, kv_group_end)
        cand_overlap_has = cand_overlap_start < cand_overlap_end
        cand_q_overlap_start = lax.div(cand_overlap_start, block_q)
        cand_q_overlap_end = pl.cdiv(cand_overlap_end, block_q)
        cand_q_overlap_steps = jnp.where(
            cand_overlap_has, cand_q_overlap_end - cand_q_overlap_start, 0
        )
        block_max_q_steps = jnp.where(
            kv_group_is_hist,
            hist_q_steps + cand_q_steps,
            jnp.where(kv_group_is_cand, cand_q_overlap_steps, 0),
        )

        def q_block_for_step(q_step):
            q_step = jnp.asarray(q_step, dtype=jnp.int32)
            hist_cand_block = jnp.where(
                q_step < hist_q_steps,
                hist_q_start + q_step,
                cand_q_start + (q_step - hist_q_steps),
            )
            cand_only_block = cand_q_overlap_start + q_step
            return jnp.where(kv_group_is_hist, hist_cand_block, cand_only_block)

        def _compute_thread(pipeline_callback):
            k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
            kv_seq_base = lax.axis_index("kv_seq") * (compute_wgs * block_kv) + wg_idx * block_kv
            kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
            plgpu.copy_gmem_to_smem(
                k_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)],
                k_smem,
                k_barriers.at[wg_idx],
            )
            plgpu.copy_gmem_to_smem(
                v_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)],
                v_smem,
                v_barriers.at[wg_idx],
            )
            plgpu.barrier_wait(k_barriers.at[wg_idx])
            plgpu.barrier_wait(v_barriers.at[wg_idx])
            dk_acc = plgpu.layout_cast(
                jnp.full((block_kv, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            dv_acc = plgpu.layout_cast(
                jnp.full((block_kv, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            (dk, dv) = pipeline_callback((dv_acc, dk_acc))
            k_smem[...] = dk.astype(dtype)
            v_smem[...] = dv.astype(dtype)

            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                k_smem, dk_ref.at[(batch, pl.ds(kv_seq_base, block_kv), q_head)], commit_group=False
            )
            plgpu.copy_smem_to_gmem(
                v_smem, dv_ref.at[(batch, pl.ds(kv_seq_base, block_kv), q_head)], commit_group=False
            )
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        def q_pipeline(
            q_step,
            q_smem,
            do_smem,
            lse_smem,
            delta_smem,
            q_consumed_barrier,
            do_consumed_barrier,
            lse_consumed_barrier,
            delta_consumed_barrier,
            carry,
        ):
            q_step = q_step[0]
            k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
            dk_acc, dv_acc = carry
            q_block_idx = q_block_for_step(q_step)

            def _compute_sT(acc_ref):
                plgpu.wgmma(acc_ref, k_smem, plgpu.transpose_ref(q_smem, (1, 0)))
                return acc_ref[...]

            sT = pl.run_scoped(_compute_sT, plgpu.ACC((block_kv, block_q), jnp.float32))
            if sm_scale != 1.0:
                sT *= sm_scale
            sT, qk_tanh, soft_sign = _apply_cap(sT, cap, cap_method)
            sT *= math.log2(math.e)

            lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_COL)
            plgpu.barrier_arrive(lse_consumed_barrier)
            pT = jnp.exp2(sT - lax.broadcast_in_dim(lse, (block_kv, block_q), [1]))
            q_seq_base = q_block_idx * block_q
            kv_seq_base = kv_seq * (compute_wgs * block_kv) + wg_idx * block_kv
            ranker_attention_mask = ranker_attention_utils.history_candidate_mask_kv_q(
                q_seq_base,
                block_q,
                kv_seq_base,
                block_kv,
                history_lower_bound,
                history_upper_bound,
                candidate_lower_bound,
                candidate_upper_bound,
                broadcasted_iota_fn=plgpu.broadcasted_iota,
                layout=plgpu.Layout.WGMMA,
            )
            pT = jnp.where(ranker_attention_mask, pT, 0)

            def _compute(refs):
                dv_acc_ref, dpT_acc_ref = refs
                plgpu.wgmma(dv_acc_ref, pT.astype(dtype), do_smem)
                plgpu.wgmma(dpT_acc_ref, v_smem, plgpu.transpose_ref(do_smem, (1, 0)))

            zeros = plgpu.layout_cast(
                jnp.full((block_kv, block_q), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            dv_acc, dpT = pl.run_state(_compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
            plgpu.barrier_arrive(do_consumed_barrier)

            delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_COL)
            plgpu.barrier_arrive(delta_consumed_barrier)

            dsT = pT * (dpT - lax.broadcast_in_dim(delta, (block_kv, block_q), [1]))
            if z_loss_weight > 0.0:
                logsumexp = lse * math.log(2.0)
                dsT += pT * lax.broadcast_in_dim(
                    z_loss_weight * logsumexp, (block_kv, block_q), [1]
                )
            if cap > 0.0 and cap_method != "none":
                if cap_method == "tanh":
                    dsT = dsT * (1 - qk_tanh**2)
                elif cap_method == "soft_sign":
                    dsT = dsT * soft_sign**2
                else:
                    raise ValueError(
                        f"cap_method must be in [tanh, soft_sign, none], got {cap_method}"
                    )
            if sm_scale != 1.0:
                dsT = dsT * sm_scale

            def compute_dk(acc_ref):
                plgpu.wgmma(acc_ref, dsT.astype(dtype), q_smem)

            dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
            plgpu.barrier_arrive(q_consumed_barrier)

            return (dk_acc, dv_acc)

        pipeline = plgpu.emit_pipeline_warp_specialized(
            q_pipeline,
            grid=(block_max_q_steps,),
            max_concurrent_steps=min([config.max_concurrent_steps, num_kv_tiles]),
            num_compute_wgs=compute_wgs,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=_compute_thread,
            in_specs=[
                plgpu.BlockSpec(
                    block_shape=(block_q, head_dim),
                    index_map=lambda i: (q_block_for_step(i), 0),
                    transforms=[tiling, swizzle],
                ),
                plgpu.BlockSpec(
                    block_shape=(block_q, head_dim),
                    index_map=lambda i: (q_block_for_step(i), 0),
                    transforms=[tiling, swizzle],
                ),
                plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (q_block_for_step(i),)),
                plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (q_block_for_step(i),)),
            ],
        )
        q_ref = q_ref.at[batch, :, q_head, :]
        do_ref = do_ref.at[batch, :, q_head, :]
        lse_ref = lse_ref.at[batch, q_head, :]
        delta_ref = delta_ref.at[batch, q_head, :]
        pipeline(q_ref, do_ref, lse_ref, delta_ref)

    q_scratch = plgpu.SMEM(
        (compute_wgs, config.block_q_dq, head_dim),
        dtype,
        transforms=(tiling, swizzle),
    )
    do_scratch = q_scratch
    lse_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
    delta_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
    dq = plgpu.kernel(
        functools.partial(kernel_dq, block_q=config.block_q_dq, block_kv=config.block_kv_dq),
        out_shape=q,
        scratch_shapes=[
            (q_scratch, do_scratch, lse_scratch, delta_scratch),
            (plgpu.Barrier(num_barriers=compute_wgs),) * 4,
        ],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(num_q_heads, num_q_tiles, batch_size),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=compute_wgs + 1,
        thread_name="wg",
    )(q, k, v, do, lse, delta, bound_arr)

    k_scratch = plgpu.SMEM(
        (compute_wgs, config.block_kv_dkv, head_dim),
        dtype,
        transforms=(tiling, swizzle),
    )
    v_scratch = k_scratch
    out_shape_kv = jax.ShapeDtypeStruct(
        (batch_size, kv_seq_len, num_q_heads, head_dim), dtype=dtype
    )
    dk, dv = plgpu.kernel(
        functools.partial(kernel_dkv, block_q=config.block_q_dkv, block_kv=config.block_kv_dkv),
        out_shape=[out_shape_kv, out_shape_kv],
        scratch_shapes=[
            (k_scratch, v_scratch),
            (plgpu.Barrier(num_barriers=compute_wgs),) * 2,
        ],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(num_q_heads, num_kv_tiles, batch_size),
        grid_names=("heads", "kv_seq", "batch"),
        num_threads=compute_wgs + 1,
        thread_name="wg",
    )(q, k, v, do, lse, delta, bound_arr)

    if q_heads_per_kv_head > 1:
        sum_shape = (*k.shape[:-1], q_heads_per_kv_head, head_dim)
        dk = dk.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dk.dtype)
        dv = dv.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dv.dtype)

    return dq, dk, dv


attention.defvjp(_attention_fwd, _attention_bwd)


@functools.partial(
    jax.jit,
    static_argnames=["config", "save_residuals", "sm_scale", "cap", "cap_method"],
)
def attention_with_pipeline_emitter(
    q,
    k,
    v,
    config: TuningConfig,
    save_residuals=False,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
):
    if config.causal:
        raise NotImplementedError(
            "Causal attention is not supported with the pipeline emitter yet."
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
    batch_size, q_seq_len, num_q_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    if k.shape != kv_shape:
        raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
    if v.shape != kv_shape:
        raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
    if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
        raise ValueError(
            f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    if head_dim % 64:
        raise ValueError(f"{head_dim=} must be divisible by 64")
    if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
        raise NotImplementedError(f"Only f16 and bf16 are supported, got dtype: {dtype}")

    max_concurrent_steps = min(config.max_concurrent_steps, kv_seq_len // config.block_kv)
    compute_wgs = 2
    block_q, block_kv = config.block_q, config.block_kv
    num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
    if rem:
        raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")

    def fa3_kernel(
        q_ref, k_ref, v_ref, out_ref, lse_ref, smem_buffers, q_barriers, schedule_barrier
    ):
        batch = lax.axis_index("batch")
        wg_idx = lax.axis_index("wg")
        qo_smem2, lse_smem2 = smem_buffers
        q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
        q_head = lax.axis_index("heads")
        kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))

        def perform_schedule_barrier():
            if config.use_schedule_barrier:
                plgpu.barrier_arrive(schedule_barrier)
                plgpu.barrier_wait(schedule_barrier)

        def _compute_thread(pipeline_callback):
            qo_smem = qo_smem2.at[wg_idx]
            lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
            m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
            l_i = jnp.full((block_q,), 0, dtype=jnp.float32)
            acc = jnp.full((block_q, head_dim), 0, dtype=jnp.float32)
            plgpu.copy_gmem_to_smem(
                q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
                qo_smem,
                q_barriers.at[wg_idx],
            )
            plgpu.barrier_wait(q_barriers.at[wg_idx])
            pl.when(wg_idx == 1)(perform_schedule_barrier)
            final_carry = pipeline_callback((acc, m_i, l_i))
            pl.when(wg_idx == 0)(perform_schedule_barrier)
            acc, m_i, l_i = final_carry
            acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
            qo_smem[...] = acc.astype(dtype)
            if lse_smem is not None:
                RCP_LN2 = 1.4426950408889634
                log2 = lambda x: jnp.log(x) * RCP_LN2
                lse_smem[...] = m_i + log2(l_i)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                qo_smem,
                out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
            )
            if lse_smem is not None:
                plgpu.copy_smem_to_gmem(
                    lse_smem,
                    lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
                )
            plgpu.wait_smem_to_gmem(0)

        def kv_pipeline(_, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry):
            acc, m_i, l_i = carry
            qo_smem = qo_smem2.at[wg_idx]

            def compute_qk(acc_ref):
                plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem, (1, 0)))
                perform_schedule_barrier()
                return acc_ref[...]

            qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
            plgpu.barrier_arrive(k_consumed_barrier)

            if sm_scale != 1.0:
                qk *= sm_scale
            qk, _, _ = _apply_cap(qk, cap, cap_method)
            log2e = math.log2(math.e)
            m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
            alpha = jnp.exp2(m_i - m_ij)
            m_i = m_ij
            p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
            acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
            l_i *= alpha
            p16 = p.astype(dtype)
            perform_schedule_barrier()
            l_i += p.sum(axis=1)

            def compute_pv(acc_ref):
                plgpu.wgmma(acc_ref, p16, v_smem)

            acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
            plgpu.barrier_arrive(v_consumed_barrier)
            return acc, m_i, l_i

        pipeline = plgpu.emit_pipeline_warp_specialized(
            kv_pipeline,
            grid=(kv_seq_len // block_kv,),
            max_concurrent_steps=max_concurrent_steps,
            num_compute_wgs=compute_wgs,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=_compute_thread,
            in_specs=[
                plgpu.BlockSpec(block_shape=(block_kv, head_dim), index_map=lambda i: (i, 0)),
                plgpu.BlockSpec(block_shape=(block_kv, head_dim), index_map=lambda i: (i, 0)),
            ],
            out_specs=[],
        )
        k_ref = k_ref.at[batch, :, kv_head, :]
        v_ref = v_ref.at[batch, :, kv_head, :]
        pipeline(k_ref, v_ref)

    out_shape = [q, None]
    if save_residuals:
        out_shape[1] = jax.ShapeDtypeStruct((batch_size, num_q_heads, q_seq_len), jnp.float32)

    qo_scratch = plgpu.SMEM((compute_wgs, block_q, head_dim), dtype)
    smem_scratch = [qo_scratch, None]
    if save_residuals:
        smem_scratch[1] = plgpu.SMEM((compute_wgs, block_q), jnp.float32)

    out, lse = plgpu.kernel(
        fa3_kernel,
        grid=(num_q_heads, num_q_tiles, batch_size),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=3,
        thread_name="wg",
        out_shape=out_shape,
        scratch_shapes=(
            tuple(smem_scratch),
            plgpu.Barrier(num_barriers=compute_wgs),
            plgpu.Barrier(num_arrivals=compute_wgs),
        ),
        compiler_params=plgpu.CompilerParams(
            approx_math=True,
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(q, k, v)

    if save_residuals:
        assert lse is not None
        return out, (lse,)

    return out


@functools.partial(
    jax.jit, static_argnames=["bound", "save_residuals", "sm_scale", "cap", "cap_method"]
)
def attention_reference(
    q,
    k,
    v,
    bound: tuple[int, int, int, int] | None = None,
    save_residuals=False,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
):
    batch_size, q_seq_len, num_q_heads, head_dim = q.shape
    kv_seq_len, num_kv_heads = k.shape[1], k.shape[2]
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    q_reshaped = q.reshape(
        batch_size, q_seq_len, num_kv_heads, num_q_heads // num_kv_heads, head_dim
    )
    logits = jnp.einsum("bqHhc,bkHc->bqHhk", q_reshaped, k)
    if sm_scale != 1.0:
        logits *= sm_scale
    logits, _, _ = _apply_cap(logits, cap, cap_method)

    if bound is None:
        bound = (0, kv_seq_len + 1, kv_seq_len + 1, kv_seq_len + 1)

    mask = ranker_attention_utils.history_candidate_mask(
        q_seq_base=0,
        block_q=q_seq_len,
        kv_seq_base=0,
        block_kv=kv_seq_len,
        history_lower_bound=bound[0],
        history_upper_bound=bound[1],
        candidate_lower_bound=bound[2],
        candidate_upper_bound=bound[3],
    )
    mask = mask[None, :, None, None, :]
    row_has_tokens = jnp.any(mask, axis=-1, keepdims=True)
    logits = jnp.where(mask, logits, -jnp.inf)

    m = logits.max(axis=-1, keepdims=True)
    m_safe = jnp.where(row_has_tokens, m, 0.0)
    unnormalized = jnp.exp(logits - m_safe)
    unnormalized = jnp.where(row_has_tokens, unnormalized, 0.0)
    l = unnormalized.sum(axis=-1, keepdims=True)
    l_safe = jnp.where(row_has_tokens, l, 1.0)
    weights = unnormalized / l_safe
    out = jnp.einsum("bqHhk,bkHc->bqHhc", weights, v).reshape(*q.shape)

    if save_residuals:
        log2e = math.log2(math.e)
        l = l.reshape(*q.shape[:-1])
        m = m.reshape(*q.shape[:-1])
        lse = m * log2e + jnp.log2(l)
        return out, (lse.swapaxes(-1, -2),)
    else:
        return out


def _attention_forward_inference(
    q,
    k,
    v,
    config: TuningConfig,
    bound=None,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
):
    batch_size, q_seq_len, num_q_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    dtype = q.dtype

    block_q, block_kv = config.block_q, config.block_kv
    max_concurrent_steps = min(config.max_concurrent_steps, kv_seq_len // block_kv)

    if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

    def kernel(q_ref, k_ref, v_ref, bound_ref, out_ref, scoped):
        batch = lax.axis_index("batch")
        q_head = lax.axis_index("heads")
        q_seq = lax.axis_index("q_seq")
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        wg_idx = lax.axis_index("wg")
        qo_smem2, k_smem, v_smem = smem_buffers
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed_barriers, v_consumed_barriers = consumed_barriers
        history_lower_bound = plgpu.load(bound_ref, (batch, 0))
        history_upper_bound = plgpu.load(bound_ref, (batch, 1))
        candidate_lower_bound = plgpu.load(bound_ref, (batch, 2))
        candidate_upper_bound = plgpu.load(bound_ref, (batch, 3))

        def perform_schedule_barrier():
            plgpu.barrier_arrive(schedule_barrier)
            plgpu.barrier_wait(schedule_barrier)

        q_tile_base = q_seq * (2 * block_q)
        q_tile_end = q_tile_base + (2 * block_q)

        def tile_has_tokens(lower, upper):
            return (q_tile_base < upper) & (q_tile_end > lower)

        valid_program = tile_has_tokens(history_lower_bound, history_upper_bound) | tile_has_tokens(
            candidate_lower_bound, candidate_upper_bound
        )

        hist_k_start = lax.div(history_lower_bound, block_kv)
        hist_k_end = pl.cdiv(history_upper_bound, block_kv)
        hist_steps = jnp.maximum(hist_k_end - hist_k_start, 0)

        cand_start = jnp.maximum(candidate_lower_bound, q_tile_base)
        cand_end = jnp.minimum(candidate_upper_bound, q_tile_end)
        cand_has = cand_start < cand_end
        cand_k_start = lax.div(cand_start, block_kv)
        cand_k_end = pl.cdiv(cand_end, block_kv)
        cand_steps = jnp.where(cand_has, cand_k_end - cand_k_start, 0)

        block_max_kv_steps = hist_steps + cand_steps

        @pl.when((wg_idx < 2) & (~valid_program))
        def _zero_out_wg():
            qo_smem = qo_smem2.at[wg_idx]
            q_seq_base = q_seq * (2 * block_q) + wg_idx * block_q
            zero_acc = plgpu.layout_cast(
                jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            qo_smem[...] = zero_acc.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                qo_smem,
                out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
            )
            plgpu.wait_smem_to_gmem(0)

        @pl.when((wg_idx < 2) & valid_program)
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")
            qo_smem = qo_smem2.at[wg_idx]
            q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q

            kv_steps = block_max_kv_steps

            plgpu.copy_gmem_to_smem(
                q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
                qo_smem,
                q_barriers.at[wg_idx],
            )
            plgpu.barrier_wait(q_barriers.at[wg_idx])

            m_i = plgpu.layout_cast(
                jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            l_i = plgpu.layout_cast(
                jnp.full((block_q,), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA_ROW,
            )
            acc = plgpu.layout_cast(
                jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )

            @pl.when(kv_steps > 0)
            def _():
                plgpu.barrier_wait(k_barriers.at[0])

            pl.when(wg_idx == 1)(perform_schedule_barrier)

            def _step_body(kv_step, acc, m_i, l_i, apply_mask_fn):
                slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                def compute_qk(acc_ref):
                    plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    perform_schedule_barrier()
                    return acc_ref[...]

                qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
                plgpu.barrier_arrive(k_consumed_barriers.at[slot])

                if sm_scale != 1.0:
                    qk *= sm_scale
                if cap > 0.0 and cap_method != "none":
                    if cap_method == "tanh":
                        qk = cap * tanh(qk / cap)
                    elif cap_method == "soft_sign":
                        qk = qk / (1.0 + _abs(qk) / cap)

                qk = apply_mask_fn(qk, kv_step)

                log2e = math.log2(math.e)
                m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
                alpha = jnp.exp2(m_i - m_ij)
                m_i = m_ij
                p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i *= alpha
                p16 = p.astype(dtype)

                def end_softmax_barriers():
                    plgpu.barrier_arrive(schedule_barrier)
                    plgpu.barrier_wait(v_barriers.at[slot])
                    plgpu.barrier_wait(schedule_barrier)

                if head_dim <= 128:
                    l_i += p.sum(axis=1)
                    acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                    end_softmax_barriers()
                else:
                    end_softmax_barriers()
                    l_i += p.sum(axis=1)

                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                    wait_step = kv_step + 1
                    wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                    @pl.when(wait_step < kv_steps)
                    def _wait():
                        plgpu.barrier_wait(k_barriers.at[wait_slot])

                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_consumed_barriers.at[slot])
                return acc, m_i, l_i

            def no_mask_fn(qk, kv_step):
                return qk

            def hist_kv_loop(kv_step, carry):
                acc, m_i, l_i = carry
                return _step_body(kv_step, acc, m_i, l_i, no_mask_fn)

            acc, m_i, l_i = lax.fori_loop(0, hist_steps, hist_kv_loop, (acc, m_i, l_i))

            def diagonal_mask_fn(qk, kv_step):
                kv_block_idx = ranker_attention_utils.kv_block_for_step(
                    kv_step,
                    hist_k_start,
                    hist_steps,
                    cand_k_start,
                )
                kv_seq_base = kv_block_idx * block_kv
                q_ids = (
                    plgpu.broadcasted_iota(
                        jnp.int32,
                        (block_q, block_kv),
                        0,
                        layout=plgpu.Layout.WGMMA,
                    )
                    + q_seq_base
                )
                kv_ids = (
                    plgpu.broadcasted_iota(
                        jnp.int32,
                        (block_q, block_kv),
                        1,
                        layout=plgpu.Layout.WGMMA,
                    )
                    + kv_seq_base
                )
                diag_mask = (
                    (q_ids == kv_ids)
                    & (q_ids >= candidate_lower_bound)
                    & (q_ids < candidate_upper_bound)
                    & (kv_ids >= candidate_lower_bound)
                    & (kv_ids < candidate_upper_bound)
                )
                return jnp.where(diag_mask, qk, -jnp.inf)

            def cand_kv_loop(cand_step, carry):
                acc, m_i, l_i = carry
                kv_step = hist_steps + cand_step
                return _step_body(kv_step, acc, m_i, l_i, diagonal_mask_fn)

            acc, m_i, l_i = lax.fori_loop(0, cand_steps, cand_kv_loop, (acc, m_i, l_i))

            pl.when(wg_idx == 0)(perform_schedule_barrier)

            acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
            qo_smem[...] = acc.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                qo_smem,
                out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
            )
            plgpu.wait_smem_to_gmem(0)

        @pl.when((wg_idx == 2) & valid_program)
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")
            kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
            for i in range(max_concurrent_steps):

                @pl.when(i < block_max_kv_steps)
                def _prefetch(i=i):
                    kv_block_idx = ranker_attention_utils.kv_block_for_step(
                        jnp.array(i, jnp.int32),
                        hist_k_start,
                        hist_steps,
                        cand_k_start,
                    )
                    s = (batch, pl.ds(kv_block_idx * block_kv, block_kv), kv_head)
                    plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
                    plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])

            @pl.loop(0, jnp.maximum(block_max_kv_steps - max_concurrent_steps, 0))
            def _kv_loop(kv_step):
                tma_step = kv_step + max_concurrent_steps
                tma_slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                kv_block_idx = ranker_attention_utils.kv_block_for_step(
                    tma_step,
                    hist_k_start,
                    hist_steps,
                    cand_k_start,
                )
                s = (batch, pl.ds(kv_block_idx * block_kv, block_kv), kv_head)
                plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
                plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])

    def entry(q_ref, k_ref, v_ref, bound_ref, out_ref):
        compute_wgs = 2
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)
        qo_scratch = plgpu.SMEM(
            (compute_wgs, block_q, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )
        k_scratch = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )
        v_scratch = plgpu.SMEM(
            (max_concurrent_steps, block_kv, head_dim),
            dtype,
            transforms=(tiling, swizzle),
        )

        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, bound_ref, out_ref, args),
            [qo_scratch, k_scratch, v_scratch],
            (
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=compute_wgs),
            ),
            (plgpu.Barrier(num_arrivals=compute_wgs, num_barriers=max_concurrent_steps),) * 2,
            plgpu.Barrier(num_arrivals=compute_wgs),
            collective_axes="wg",
        )

    num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
    if rem:
        raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")

    out = plgpu.kernel(
        entry,
        out_shape=q,
        grid=(num_q_heads, num_q_tiles, batch_size),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, bound)

    return out
