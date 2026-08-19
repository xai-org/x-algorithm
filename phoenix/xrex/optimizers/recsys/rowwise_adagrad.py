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
from xrex.optimizers.recsys.async_emb_gradient_update import (
    AsyncEmbGradientUpdate,
    AsyncEmbOptimizer,
)
from xrex.optimizers.recsys.protocol import RecsysEmbeddingOptimizer, _lookup

if TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )

_LN2 = math.log(2)


@configclass
class RecsysRowwiseAdagradConfig(Config):
    enabled: bool = False
    learning_rate: float = 0.1
    eps: float = 1e-10
    initial_accumulator_value: float = 0.0
    half_life_steps: int | None = 5000
    lazy_decay: bool = False
    weight_decay: float = 0.0


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
    ):
        self._learning_rate = learning_rate
        self._eps = eps
        self._initial_accumulator_value = initial_accumulator_value
        self._lazy_decay = lazy_decay
        self._weight_decay = weight_decay
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
        lazy_features = self._decay_rate is not None or self._weight_decay > 0.0
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
        update: AsyncEmbGradientUpdate,
        table: jax.Array,
        state: RecsysRowwiseAdagradState,
        gate: jax.Array,
    ) -> tuple[tuple[jax.Array, ...], jax.Array, RecsysRowwiseAdagradState]:
        if self._lazy_decay:
            raise ValueError("the fused rowwise Adagrad update does not support lazy decay")
        if self._weight_decay > 0.0:
            raise ValueError("the fused rowwise Adagrad update does not support weight decay")

        if 32 % context.shard_width != 0:
            raise ValueError(
                f"the fused rowwise Adagrad update needs a row shard that divides a warp, "
                f"got shard_width={context.shard_width} (emb_width={context.emb_width})"
            )

        from xrex.cuda.async_emb import async_emb

        @shard_map(
            mesh=context.mesh,
            in_specs=(
                P(context.data_axis, None),
                P(context.data_axis),
                P(),
                P(None, context.table_axis),
                P(),
                P(),
                P(context.data_axis, None),
            ),
            out_specs=(
                P(context.data_axis, None),
                P(context.data_axis),
                P(),
                P(None, context.table_axis),
                P(),
            ),
            check_vma=False,
        )
        def start(
            grads: jax.Array,
            segment_ids: jax.Array,
            unique_tokens: jax.Array,
            table: jax.Array,
            accum: jax.Array,
            pending: jax.Array,
            gate: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            grads_pin, segments_pin, tokens_pin, table_out, accum_out = (
                async_emb.rowwise_adagrad_update_start(
                    grads,
                    segment_ids,
                    unique_tokens,
                    table,
                    accum,
                    pending,
                    gate,
                    context,
                    learning_rate=self._learning_rate,
                    eps=self._eps,
                    decay_factor=self.decay_factor,
                )
            )
            return grads_pin, segments_pin, tokens_pin, table_out, accum_out

        grads_pin, segments_pin, tokens_pin, table_out, accum_out = start(
            update.grads,
            update.segment_ids,
            update.unique_tokens,
            table,
            state.row_sum_sq["table"],
            update.pending,
            gate,
        )
        return (
            (grads_pin, segments_pin, tokens_pin),
            table_out,
            state._replace(row_sum_sq={**state.row_sum_sq, "table": accum_out}),
        )

    def gradient_update_done(
        self, context: async_emb.AsyncEmbContextHandle, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        from xrex.cuda.async_emb import async_emb

        @shard_map(mesh=context.mesh, in_specs=(P(),), out_specs=(P(), P(), P()), check_vma=False)
        def done(gate: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            norm, valid, done_pin = async_emb.rowwise_adagrad_update_done(gate, context)
            return norm, valid != 0, done_pin

        return done(gate)

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

        base_x = small_emb_table.x
        if self._weight_decay > 0.0:
            if self._lazy_decay:
                assert delta is not None
                wd_factor = jnp.exp(-delta.astype(jnp.float32) * self._weight_decay)
            else:
                wd_factor = jnp.full_like(row_mean_sq, math.exp(-self._weight_decay))
            touched = row_mean_sq > 0.0
            wd_factor = jnp.where(touched, wd_factor, 1.0)[:, None]
            base_x = (base_x.astype(jnp.float32) * wd_factor).astype(small_emb_table.x.dtype)
        new_params = replace(
            small_emb_table,
            x=(base_x + update.astype(small_emb_table.x.dtype)),
        )

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
