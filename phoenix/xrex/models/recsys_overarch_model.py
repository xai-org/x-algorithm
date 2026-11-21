# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from xai_configlib import configclass
from xai_proto import recsys_pb2
from xrex.data.recsys import teacher_distill
from xrex.models.layers import get_parameter
from xrex.models.loss_recsys import multihot_loss_weights
from xrex.models.normalization import rms_norm_fn
from xrex.models.recsys_distillation_loss import DISTILL_LOSS_TYPES, distillation_loss_compute
from xrex.models.recsys_model import (
    CandidateInputs,
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
    RecsysFeaturesBatch,
    cast_jax,
)
from xrex.models.recsys_overarch import OverArch
from xrex.models.sharding_context import ShardingContext
from xrex.utils.sharding import with_sharding_constraint_unless_manual

DISTILL_HEAD_LOGITS = "distill_head"
DISTILL_HEAD_OFFSET = "distill_head_offset"


def parse_action_loss_weights(spec: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        if ":" not in item:
            raise ValueError(f"action_loss_weights: {item!r} must be NAME:weight")
        name, value = (x.strip() for x in item.rsplit(":", 1))
        try:
            index = recsys_pb2.ActionName.Value(name)
        except ValueError as e:
            raise ValueError(f"action_loss_weights: unknown action {name!r}") from e
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"action_loss_weights: {item!r} needs a finite weight > 0")
        if index in out:
            raise ValueError(f"action_loss_weights: {name!r} given twice")
        out[index] = weight
    return out


@configclass
class RecsysOverArchModelConfig(RecsysAggregatedModelConfig):
    overarch_enabled: bool = True
    overarch_hidden_dims: tuple[int, ...] = (4096, 4096)
    overarch_output_dim: int | None = 4096
    action_loss_weights: str = ""
    action_loss_weights_label_only: bool = False
    distill_teacher: str = ""
    distill_actions: str = teacher_distill.DEFAULT_ACTIONS
    distill_weight: float = 1.0
    distill_loss: str = "bce"
    distill_bad_spans_ms: str = ""
    distill_separate_head: bool = False
    distill_head_bias_init: str = "0"

    @classmethod
    def from_base(
        cls, base: RecsysAggregatedModelConfig, **changes: Any
    ) -> RecsysOverArchModelConfig:
        values = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
        values.update(changes)
        return cls(**values)

    @property
    def readout_width(self) -> int:
        if self.overarch_enabled and self.overarch_output_dim:
            return self.overarch_output_dim
        return self.model_config.emb_size

    @property
    def distill_action_indices(self) -> tuple[int, ...]:
        return teacher_distill.parse_actions(self.distill_actions)

    @property
    def distill_head_bias_inits(self) -> tuple[float, ...]:
        values = tuple(float(v) for v in self.distill_head_bias_init.split(",") if v.strip())
        count = len(self.distill_action_indices)
        if len(values) == 1:
            return values * count
        if len(values) != count:
            raise ValueError(
                f"distill_head_bias_init {self.distill_head_bias_init!r} needs one value or one "
                f"per distill_actions entry ({count})"
            )
        return values

    @property
    def needs_subclass(self) -> bool:
        return self.overarch_enabled or bool(self.action_loss_weights or self.distill_teacher)

    def validate(self) -> None:
        super().validate()
        if self.overarch_enabled:
            if self.overarch_output_dim is not None and self.overarch_output_dim < 1:
                raise ValueError("overarch_output_dim must be >= 1 (or None for emb_size)")
            if any(d < 1 for d in self.overarch_hidden_dims):
                raise ValueError(
                    f"overarch_hidden_dims must be >= 1, got {self.overarch_hidden_dims}"
                )
        weights = parse_action_loss_weights(self.action_loss_weights)
        vocab = self.model_config.output_vocab_size
        if weights and vocab is not None and max(weights) >= vocab:
            raise ValueError(
                f"action_loss_weights {self.action_loss_weights!r} exceed output_vocab_size"
            )
        teacher_distill.parse_spans_ms(self.distill_bad_spans_ms)
        if not self.distill_teacher:
            return
        if self.distill_weight <= 0:
            raise ValueError("distill_teacher needs distill_weight > 0")
        if self.distill_loss not in DISTILL_LOSS_TYPES:
            raise ValueError(
                f"distill_loss must be one of {DISTILL_LOSS_TYPES}, got {self.distill_loss!r}"
            )
        if vocab is not None and max(self.distill_action_indices) >= vocab:
            raise ValueError(f"distill_actions {self.distill_actions} exceed output_vocab_size")
        if not all(math.isfinite(v) for v in self.distill_head_bias_inits):
            raise ValueError(
                f"distill_head_bias_init must be finite, got {self.distill_head_bias_init!r}"
            )

    def make(self, sharding_context: ShardingContext) -> RecsysAggregatedModel:
        if not self.needs_subclass:
            return super().make(sharding_context)
        self.validate()
        return RecsysOverArchModel(
            model=self.model_config.make(sharding_context=sharding_context),
            config=self,
            sharding_context=sharding_context,
        )


@dataclass
class RecsysOverArchModel(RecsysAggregatedModel):
    config: RecsysOverArchModelConfig
    name: str | None = "recsys_aggregated_model"

    @property
    def readout_width(self) -> int:
        return self.config.readout_width

    @hk.transparent
    def readout_inputs(self, out_embeddings: jax.Array) -> jax.Array:
        config = self.config
        if not config.overarch_enabled:
            return out_embeddings
        scale_config = config.model_config.scale_config
        out = OverArch(
            widths=(*config.overarch_hidden_dims, config.readout_width), scale_config=scale_config
        )(out_embeddings)
        return rms_norm_fn(
            out, name="overarch_output_norm", weight_decay_mask=scale_config.ln_weight_decay_mask
        )

    @hk.transparent
    def decode_aux_value_heads(self, inputs: jax.Array) -> dict[str, jax.Array]:
        heads = super().decode_aux_value_heads(inputs)
        if self.config.distill_teacher and self.config.distill_separate_head:
            heads = {**heads, **self.decode_distill_head(inputs)}
        return heads

    @hk.transparent
    def decode_distill_head(self, inputs: jax.Array) -> dict[str, jax.Array]:
        config = self.config
        scale_config = config.model_config.scale_config
        width = inputs.shape[-1]
        count = len(config.distill_action_indices)
        own = jax.lax.stop_gradient(
            self._get_unembedding()[:, jnp.asarray(config.distill_action_indices)]
        )
        learned = get_parameter(
            "distill_unembeddings",
            [width, count],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=scale_config.head_lr_multiplier(width),
            rms_clip_axes=(-1, -2),
        )
        learned = with_sharding_constraint_unless_manual(learned, P(None, None))
        bias = get_parameter(
            "distill_head_bias",
            [count],
            dtype=jnp.float32,
            init=hk.initializers.Constant(jnp.asarray(config.distill_head_bias_inits)),
            pspec=P(None),
            lr_multiplier=1.0,
            rms_clip_axes=(),
            dtype_unwrap=jnp.float32,
        )
        x = inputs.astype(own.dtype)
        scale = scale_config.output_scale(width)
        base = jnp.dot(x, own).astype(jnp.float32) * scale
        logits = jnp.dot(x, learned.astype(own.dtype)).astype(jnp.float32) * scale + bias
        return {DISTILL_HEAD_LOGITS: logits, DISTILL_HEAD_OFFSET: logits - base}

    @hk.transparent
    def loss_terms_hook(
        self,
        batch: RecsysFeaturesBatch,
        c: CandidateInputs,
        candidate_logits: jax.Array,
        aux_logits: dict[str, jax.Array],
        loss_mask: jax.Array,
        normalizer: Callable[[str], jax.Array | None],
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        config = self.config
        distill_mask = loss_mask
        if config.action_loss_weights:
            head_w = jnp.ones(loss_mask.shape[-1])
            for index, weight in parse_action_loss_weights(config.action_loss_weights).items():
                head_w = head_w.at[index].set(weight)
            loss_mask = loss_mask * head_w
            if not config.action_loss_weights_label_only:
                distill_mask = loss_mask
        if not config.distill_teacher:
            return loss_mask, jnp.array(0.0), {}
        teacher_probs = batch["candidate_seq"].get("teacher_probs")
        assert teacher_probs is not None, "distill_teacher needs the dataset's teacher_probs"
        teacher_probs = cast_jax(teacher_probs)
        pad_len = c.loss_padding_mask.shape[1] - teacher_probs.shape[1]
        teacher_probs = jnp.pad(
            teacher_probs, ((0, 0), (0, pad_len), (0, 0)), constant_values=jnp.nan
        )
        heads = jnp.asarray(config.distill_action_indices)
        logits = (
            aux_logits[DISTILL_HEAD_LOGITS]
            if config.distill_separate_head
            else candidate_logits[:, :, heads]
        )
        distill_loss, coverage = distillation_loss_compute(
            logits=logits,
            teacher_probs=teacher_probs,
            padding_mask=c.loss_padding_mask,
            loss_mask=distill_mask[:, :, heads],
            raw_weights=c.loss_weights,
            normalizer=normalizer("multihot"),
            loss_type=config.distill_loss,
        )
        stats = {"distill-coverage": coverage, "distill-loss": distill_loss}
        if config.distill_separate_head:
            weights = multihot_loss_weights(c.loss_padding_mask, c.loss_weights)
            weights = weights * (~jnp.isnan(teacher_probs)).any(-1)
            offset = aux_logits[DISTILL_HEAD_OFFSET].mean(-1)
            stats["distill-head-offset-impressed"] = jnp.sum(offset * weights) / (
                jnp.sum(weights) + 1e-10
            )
        return loss_mask, config.distill_weight * distill_loss, stats
