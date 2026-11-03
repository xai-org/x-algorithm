# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
import typing
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.models.model_utils import Parameter
from xrex.optimizers.recsys.protocol import AsyncEmbOptimizer, RecsysEmbeddingOptimizer, _lookup

if TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )

_LN2 = math.log(2)
_MAX_FUSED_SHARD_WIDTH = 32


def _hash_u32(x: jax.Array) -> jax.Array:
    x = (x ^ (x >> 16)) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _stochastic_round_bf16(value: jax.Array, step: jax.Array, rows: jax.Array) -> jax.Array:
    base = _hash_u32(_hash_u32(step.astype(jnp.uint32)) ^ _hash_u32(rows.astype(jnp.uint32)))
    cols = jax.lax.broadcasted_iota(jnp.uint32, value.shape, 1)
    rand = (cols * jnp.uint32(0x9E3779B1) + base[:, None]) >> 16
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32) + rand
    return jax.lax.bitcast_convert_type((bits >> 16).astype(jnp.uint16), jnp.bfloat16)


@configclass
class RecsysRowwiseAdagradConfig(Config):
    enabled: bool = False
    learning_rate: float = 0.1
    eps: float = 1e-10
    initial_accumulator_value: float = 0.0
    half_life_steps: int | None = 5000
    lazy_decay: bool = False
    weight_decay: float = 0.0
    weight_decay_exempt_window: int = 0
    stochastic_rounding: bool = False


class RecsysRowwiseAdagradState(NamedTuple):
    row_sum_sq: dict[str, jnp.ndarray]
    step: jnp.ndarray | None = None
    last_step: dict[str, jnp.ndarray] | None = None


class RecsysRowwiseAdagradOptimizer(RecsysEmbeddingOptimizer, AsyncEmbOptimizer):
    def __init__(
        self,
        learning_rate: float = 0.1,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0,
        half_life_steps: int | None = None,
        lazy_decay: bool = False,
        weight_decay: float = 0.0,
        weight_decay_exempt_window: int = 0,
        stochastic_rounding: bool = False,
    ):
        if weight_decay_exempt_window < 0:
            raise ValueError(f"{weight_decay_exempt_window=} must be >= 0")
        if (weight_decay_exempt_window > 0 or stochastic_rounding) and not lazy_decay:
            raise ValueError("weight_decay_exempt_window and stochastic_rounding need lazy_decay")
        self._learning_rate = learning_rate
        self._eps = eps
        self._initial_accumulator_value = initial_accumulator_value
        self._lazy_decay = lazy_decay
        self._weight_decay = weight_decay
        self._weight_decay_exempt_window = weight_decay_exempt_window
        self._stochastic_rounding = stochastic_rounding
        self._decay_rate = _LN2 / half_life_steps if half_life_steps is not None else None

    def init(self, params: dict[str, Parameter]) -> RecsysRowwiseAdagradState:
        row_sum_sq = {}
        last_step = {}
        for key, p in params.items():
            num_rows = p.x.shape[0]
            row_sum_sq[key] = jnp.full(
                (num_rows,), self._initial_accumulator_value, dtype=jnp.float32
            )
            last_step[key] = jnp.zeros((num_rows,), dtype=jnp.int32)
        lazy_features = (
            self._decay_rate is not None or self._weight_decay > 0.0 or self._stochastic_rounding
        )
        need_timestamps = lazy_features and self._lazy_decay
        return RecsysRowwiseAdagradState(
            row_sum_sq=row_sum_sq,
            step=jnp.zeros((), dtype=jnp.int32) if need_timestamps else None,
            last_step=last_step if need_timestamps else None,
        )

    def transform_embeddings(
        self,
        embeddings: Any,
        token_ids: jax.Array,
        state: Any,
    ) -> tuple[Any, Any]:
        return embeddings, None

    @property
    def decay_factor(self) -> float:
        return math.exp(-self._decay_rate) if self._decay_rate is not None else 1.0

    def gradient_update_start(
        self,
        context: async_emb.AsyncEmbContextHandle,
        table: jax.Array,
        state: RecsysRowwiseAdagradState,
        gate: jax.Array,
    ) -> tuple[jax.Array, jax.Array, RecsysRowwiseAdagradState, dict[str, jax.Array]]:
        if self._lazy_decay and (state.step is None or state.last_step is None):
            raise ValueError("fused lazy decay needs timestamped state (step/last_step)")

        from xrex.utils import recsys_async_emb

        async_emb = recsys_async_emb.kernel_bindings(context)
        row_sharded = recsys_async_emb.is_row_sharded(context)
        if not row_sharded and _MAX_FUSED_SHARD_WIDTH % context.shard_width != 0:
            raise ValueError(
                "the fused rowwise Adagrad update needs shard_width to divide "
                f"{_MAX_FUSED_SHARD_WIDTH}, got shard_width={context.shard_width} "
                f"(emb_width={context.emb_width})"
            )

        metrics: dict[str, jax.Array] = {}
        lazy_options: dict[str, Any] = {}
        if self._weight_decay_exempt_window > 0 or self._stochastic_rounding:
            lazy_options = {
                "weight_decay_exempt_window": self._weight_decay_exempt_window,
                "stochastic_rounding": self._stochastic_rounding,
            }

        table_spec = recsys_async_emb.table_spec(context)
        state_spec = P(context.table_axis) if row_sharded else P()
        if self._lazy_decay:

            @shard_map(
                mesh=context.mesh,
                in_specs=(table_spec, state_spec, state_spec, P(), P(context.data_axis, None)),
                out_specs=(table_spec, state_spec, state_spec, P()),
                check_vma=False,
            )
            def start_lazy(
                table: jax.Array,
                accum: jax.Array,
                last_step: jax.Array,
                step: jax.Array,
                gate: jax.Array,
            ) -> tuple[jax.Array, ...]:
                return async_emb.rowwise_adagrad_lazy_update_start(
                    table,
                    accum,
                    last_step,
                    step,
                    gate,
                    context,
                    learning_rate=self._learning_rate,
                    eps=self._eps,
                    accum_decay_rate=self._decay_rate or 0.0,
                    weight_decay_rate=self._weight_decay,
                    **lazy_options,
                )

            assert state.step is not None and state.last_step is not None
            table_out, accum_out, last_step_out, pin = start_lazy(
                table, state.row_sum_sq["table"], state.last_step["table"], state.step, gate
            )
            return (
                pin,
                table_out,
                state._replace(
                    row_sum_sq={**state.row_sum_sq, "table": accum_out},
                    last_step={**state.last_step, "table": last_step_out},
                ),
                metrics,
            )

        @shard_map(
            mesh=context.mesh,
            in_specs=(table_spec, state_spec, P(context.data_axis, None)),
            out_specs=(table_spec, state_spec, P()),
            check_vma=False,
        )
        def start(table: jax.Array, accum: jax.Array, gate: jax.Array) -> tuple[jax.Array, ...]:
            return async_emb.rowwise_adagrad_update_start(
                table,
                accum,
                gate,
                context,
                learning_rate=self._learning_rate,
                eps=self._eps,
                decay_factor=self.decay_factor,
                weight_decay_factor=math.exp(-self._weight_decay),
            )

        table_out, accum_out, pin = start(table, state.row_sum_sq["table"], gate)
        return (
            pin,
            table_out,
            state._replace(row_sum_sq={**state.row_sum_sq, "table": accum_out}),
            metrics,
        )

    def gradient_update_done(
        self,
        context: async_emb.AsyncEmbContextHandle,
        state: RecsysRowwiseAdagradState,
        gate: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, RecsysRowwiseAdagradState, jax.Array]:
        from xrex.utils import recsys_async_emb

        async_emb = recsys_async_emb.kernel_bindings(context)

        @shard_map(
            mesh=context.mesh, in_specs=(P(),), out_specs=(P(), P(), P(), P()), check_vma=False
        )
        def done(gate: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            norm, valid, pending, done_pin = async_emb.rowwise_adagrad_update_done(gate, context)
            return norm, valid != 0, pending != 0, done_pin

        norm, valid, pending, done_pin = done(gate)
        if self._lazy_decay:
            assert state.step is not None
            state = state._replace(step=state.step + valid.astype(jnp.int32))
        return norm, valid, pending, state, done_pin

    def sparse_update(
        self,
        grads: Parameter,
        full_state: Any,
        full_emb_table: Parameter,
        unique_tokens: jax.Array,
        num_unique: int,
        lr: float,
        valid_step: jax.Array,
        carry: Any = None,
    ) -> tuple[Parameter, RecsysRowwiseAdagradState, dict[str, jax.Array]]:
        del lr
        full_state = typing.cast(RecsysRowwiseAdagradState, full_state)
        eps = self._eps

        big_accum = full_state.row_sum_sq["table"]
        small_accum = big_accum[unique_tokens]
        small_emb_table = _lookup(full_emb_table, unique_tokens)

        _has_timestamps = full_state.step is not None and full_state.last_step is not None
        delta = None
        if _has_timestamps:
            current_step = full_state.step
            small_last_step = full_state.last_step["table"][unique_tokens]
            delta = jnp.maximum(0, current_step - small_last_step)
        if self._decay_rate is not None:
            if self._lazy_decay:
                assert delta is not None
                decay_factor = jnp.exp(-delta * self._decay_rate)
            else:
                decay_factor = self.decay_factor
            small_accum = small_accum * decay_factor

        g = grads.x.astype(jnp.float32)
        row_mean_sq = jnp.mean(g**2, axis=-1)
        new_small_accum = small_accum + row_mean_sq
        scale = 1.0 / (jnp.sqrt(new_small_accum) + eps)
        update = -self._learning_rate * scale[:, None] * g

        wd_factor = None
        if self._weight_decay > 0.0:
            if self._lazy_decay:
                assert delta is not None
                wd_factor = jnp.exp(-delta.astype(jnp.float32) * self._weight_decay)
                if self._weight_decay_exempt_window > 0:
                    wd_factor = jnp.where(delta > self._weight_decay_exempt_window, wd_factor, 1.0)
            else:
                wd_factor = jnp.full_like(row_mean_sq, math.exp(-self._weight_decay))
            touched = row_mean_sq > 0.0
            wd_factor = jnp.where(touched, wd_factor, 1.0)[:, None]
        if self._stochastic_rounding:
            candidate = small_emb_table.x.astype(jnp.float32)
            if wd_factor is not None:
                candidate = candidate * wd_factor
            rows = jnp.where(
                unique_tokens < 0, unique_tokens + full_emb_table.x.shape[0], unique_tokens
            )
            assert full_state.step is not None
            new_x = _stochastic_round_bf16(candidate + update, full_state.step, rows)
        else:
            base_x = small_emb_table.x
            if wd_factor is not None:
                base_x = (base_x.astype(jnp.float32) * wd_factor).astype(small_emb_table.x.dtype)
            new_x = base_x + update.astype(small_emb_table.x.dtype)
        new_params = replace(small_emb_table, x=new_x)

        new_emb_table = replace(
            full_emb_table,
            x=full_emb_table.x.at[unique_tokens[:num_unique]].set(
                jnp.where(valid_step, new_params.x, small_emb_table.x)
            ),
        )

        metrics: dict[str, jax.Array] = {}

        new_big_accum = jnp.where(
            valid_step,
            big_accum.at[unique_tokens].set(new_small_accum),
            big_accum,
        )

        if _has_timestamps:
            big_last_step = full_state.last_step["table"]
            new_big_last_step = jnp.where(
                valid_step,
                big_last_step.at[unique_tokens].set(
                    jnp.broadcast_to(current_step, unique_tokens.shape)
                ),
                big_last_step,
            )
            new_step = jnp.where(valid_step, current_step + 1, current_step)
            return (
                new_emb_table,
                RecsysRowwiseAdagradState(
                    row_sum_sq={"table": new_big_accum},
                    step=new_step,
                    last_step={"table": new_big_last_step},
                ),
                metrics,
            )

        return (
            new_emb_table,
            RecsysRowwiseAdagradState(row_sum_sq={"table": new_big_accum}),
            metrics,
        )
