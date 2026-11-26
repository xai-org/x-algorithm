# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.models.model_utils import get_parameter
from xrex.models.page_critic import masked_mean, page_critic_td_target
from xrex.models.page_decoder import (
    PageDecodeOutputs,
    overlap_features,
    overlap_prefix_features,
    pairwise_same_author,
    pairwise_sid_common_prefix,
)

PAGE_CRITIC_ONLY_PARAM_REGEX = "page_critic_only"


@configclass
class PageCriticOnlyConfig(Config):
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 1024
    head_hidden_dim: int = 512
    max_steps: int = 48
    use_overlap_features: bool = True
    huber_delta: float = 1.0
    exploration_return_weight: float = 9.0
    return_scale: float = 10.0
    td_lambda: float = 0.9


class PageCriticOnlyOutputs(NamedTuple):
    total: jax.Array
    stats: dict[str, jax.Array]


def _diag_overlap_features(
    auth_sorted: jax.Array,
    sids_sorted: jax.Array,
    slate_sorted: jax.Array,
) -> jax.Array:
    S = auth_sorted.shape[1]
    before = jnp.tril(jnp.ones((S, S), dtype=bool), k=-1)[None]
    same_author = pairwise_same_author(auth_sorted.astype(jnp.int32), slate_sorted) & before
    cpl = jnp.where(before, pairwise_sid_common_prefix(sids_sorted, slate_sorted), 0)
    return overlap_features(same_author.any(axis=-1), cpl.max(axis=-1), sids_sorted.shape[-1])


class PageCriticTransformer(hk.Module):
    def __init__(self, config: PageCriticOnlyConfig, name: str = "page_critic_only"):
        super().__init__(name=name)
        self._config: PageCriticOnlyConfig = config

    def _param(self, name: str, shape: list[int], init: hk.initializers.Initializer) -> jax.Array:
        return get_parameter(
            name, shape, dtype=jnp.float32, init=init, pspec=P(*([None] * len(shape)))
        )

    def _matrix(self, name: str, shape: list[int]) -> jax.Array:
        return self._param(name, shape, hk.initializers.VarianceScaling(1.0, mode="fan_in"))

    def _zeros(self, name: str, shape: list[int]) -> jax.Array:
        return self._param(name, shape, hk.initializers.Constant(0.0))

    def _rms_norm(self, name: str, x: jax.Array) -> jax.Array:
        scale = self._param(name, [x.shape[-1]], hk.initializers.Constant(1.0))
        x = x.astype(jnp.float32)
        return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5) * scale

    def _layer(self, i: int, z: jax.Array) -> jax.Array:
        cfg = self._config
        B, S, d = z.shape
        h = cfg.num_heads
        zn = self._rms_norm(f"layer{i}_attn_ln", z)
        q, k, v = (
            jnp.einsum("bsd,de->bse", zn, self._matrix(f"layer{i}_attn_{n}_w", [d, d])).reshape(
                B, S, h, d // h
            )
            for n in "qkv"
        )
        logits = jnp.einsum("bshe,bthe->bhst", q, k) / math.sqrt(d // h)
        causal = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None]
        attn = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
        ctx = jnp.einsum("bhst,bthe->bshe", attn, v).reshape(B, S, d)
        z = z + jnp.einsum("bsd,de->bse", ctx, self._matrix(f"layer{i}_attn_o_w", [d, d]))
        zn = self._rms_norm(f"layer{i}_ffn_ln", z)
        hidden = jax.nn.gelu(
            jnp.einsum("bsd,df->bsf", zn, self._matrix(f"layer{i}_ffn_in_w", [d, cfg.ffn_dim]))
            + self._zeros(f"layer{i}_ffn_in_b", [cfg.ffn_dim])
        )
        return (
            z
            + jnp.einsum(
                "bsf,fd->bsd", hidden, self._matrix(f"layer{i}_ffn_out_w", [cfg.ffn_dim, d])
            )
            + self._zeros(f"layer{i}_ffn_out_b", [d])
        )

    def trunk(self, emb_sorted: jax.Array, user_emb: jax.Array) -> jax.Array:
        cfg = self._config
        S = emb_sorted.shape[1]
        assert cfg.d_model % cfg.num_heads == 0
        assert S <= cfg.max_steps, f"S={S} exceeds max_steps={cfg.max_steps}"
        d = cfg.d_model
        items = jnp.einsum(
            "bsd,de->bse",
            emb_sorted.astype(jnp.float32),
            self._matrix("item_proj_w", [emb_sorted.shape[-1], d]),
        )
        z0 = jnp.einsum(
            "bd,de->be",
            user_emb.astype(jnp.float32),
            self._matrix("user_proj_w", [user_emb.shape[-1], d]),
        )
        pos = self._param("pos_emb", [cfg.max_steps, d], hk.initializers.RandomNormal(0.02))
        z = jnp.concatenate([z0[:, None, :], items[:, : S - 1, :]], axis=1) + pos[None, :S]
        for i in range(cfg.num_layers):
            z = self._layer(i, z)
        return self._rms_norm("final_ln", z)

    def candidate_keys(self, emb: jax.Array) -> jax.Array:
        w = self._matrix("cand_proj_w", [emb.shape[-1], self._config.d_model])
        return jnp.einsum("bcd,de->bce", emb.astype(jnp.float32), w)

    def q_head(self, state: jax.Array, keys: jax.Array, feats: jax.Array | None) -> jax.Array:
        cfg = self._config
        parts = [state, keys, state * keys]
        if cfg.use_overlap_features:
            assert feats is not None
            parts.append(feats.astype(jnp.float32))
        x = jnp.concatenate(parts, axis=-1)
        hidden = jax.nn.gelu(
            jnp.einsum(
                "...i,ih->...h", x, self._matrix("head_in_w", [x.shape[-1], cfg.head_hidden_dim])
            )
            + self._zeros("head_in_b", [cfg.head_hidden_dim])
        )
        return jnp.einsum(
            "...h,h->...", hidden, self._zeros("head_out_w", [cfg.head_hidden_dim])
        ) + (self._zeros("head_out_b", []))

    def __call__(
        self,
        emb_sorted: jax.Array,
        slate_sorted: jax.Array,
        user_emb: jax.Array,
        auth_sorted: jax.Array,
        sids_sorted: jax.Array,
    ) -> jax.Array:
        state = self.trunk(emb_sorted, user_emb)
        feats = (
            _diag_overlap_features(auth_sorted, sids_sorted, slate_sorted)
            if self._config.use_overlap_features
            else None
        )
        return self.q_head(state, self.candidate_keys(emb_sorted), feats)

    @hk.name_like("__call__")
    def value_grid(
        self,
        emb_sorted: jax.Array,
        slate_sorted: jax.Array,
        user_emb: jax.Array,
        auth_sorted: jax.Array,
        sids_sorted: jax.Array,
        steps: int,
    ) -> jax.Array:
        state = self.trunk(emb_sorted, user_emb)[:, :steps, None, :]
        keys = self.candidate_keys(emb_sorted)[:, None, :, :]
        shape = (*state.shape[:2], keys.shape[2], state.shape[-1])
        feats = None
        if self._config.use_overlap_features:
            same_author = pairwise_same_author(auth_sorted.astype(jnp.int32), slate_sorted)
            cpl = pairwise_sid_common_prefix(sids_sorted, slate_sorted)
            feats = overlap_prefix_features(same_author, cpl, sids_sorted.shape[-1])[:, :steps]
        return self.q_head(jnp.broadcast_to(state, shape), jnp.broadcast_to(keys, shape), feats)

    @hk.name_like("__call__")
    def decode(
        self,
        pool_emb: jax.Array,
        pool_mask: jax.Array,
        behavior: jax.Array,
        user_emb: jax.Array,
        steps: jax.Array,
        temperature: jax.Array,
        rng: jax.Array,
        pool_auth: jax.Array,
        pool_sids: jax.Array,
        *,
        max_steps: int,
        eta: jax.Array | float,
        nu_author: jax.Array | float,
        nu_sid: jax.Array | float,
        prior_tau: jax.Array | float,
    ) -> PageDecodeOutputs:
        B, P, _ = pool_emb.shape
        T = max_steps
        assert T <= self._config.max_steps, f"max_steps={T} exceeds {self._config.max_steps}"

        def per_row(x: jax.Array | float) -> jax.Array:
            return jnp.broadcast_to(jnp.asarray(x, jnp.float32), (B,))[:, None]

        eta, nu_author, nu_sid, prior_tau = map(per_row, (eta, nu_author, nu_sid, prior_tau))
        L = pool_sids.shape[-1]
        keys = self.candidate_keys(pool_emb)
        same_author = pairwise_same_author(pool_auth.astype(jnp.int32), pool_mask)
        cpl = pairwise_sid_common_prefix(pool_sids, pool_mask)
        prior = jnp.where(pool_mask, behavior.astype(jnp.float32) / prior_tau, 0.0)
        n_steps = jnp.minimum(
            jnp.minimum(steps.astype(jnp.int32), pool_mask.sum(axis=1).astype(jnp.int32)), T
        )
        tau = jnp.maximum(temperature.astype(jnp.float32), 0.0)

        seq = jnp.zeros((B, T, pool_emb.shape[-1]), jnp.float32)
        picked_mask = jnp.zeros((B, P), bool)
        run_author = jnp.zeros((B, P), bool)
        run_cpl = jnp.zeros((B, P), jnp.int32)
        picked = jnp.full((B, T), -1, jnp.int32)
        picked_prob = jnp.zeros((B, T), jnp.float32)
        for t in range(T):
            state = self.trunk(seq, user_emb)[:, t, None, :]
            feats = (
                overlap_features(run_author, run_cpl, L)
                if self._config.use_overlap_features
                else None
            )
            q = self.q_head(
                jnp.broadcast_to(state, (*keys.shape[:2], state.shape[-1])), keys, feats
            )
            utility = prior + q / eta - nu_author * run_author - nu_sid * run_cpl
            logits = jnp.where(picked_mask | ~pool_mask, -jnp.inf, utility)
            gumbel = jax.random.gumbel(jax.random.fold_in(rng, t), logits.shape, jnp.float32)
            pick = jnp.argmax(logits + tau[:, None] * gumbel, axis=-1)
            sampled_p = jax.nn.softmax(logits / jnp.maximum(tau[:, None], 1e-6), axis=-1)
            prob = jnp.where(
                tau > 0.0, jnp.take_along_axis(sampled_p, pick[:, None], axis=1)[:, 0], 1.0
            )

            live = t < n_steps
            picked = picked.at[:, t].set(jnp.where(live, pick.astype(jnp.int32), -1))
            picked_prob = picked_prob.at[:, t].set(jnp.where(live, prob, 0.0))
            picked_mask = picked_mask | (jax.nn.one_hot(pick, P, dtype=jnp.bool_) & live[:, None])
            col_author = jnp.take_along_axis(same_author, pick[:, None, None], axis=2)[:, :, 0]
            col_cpl = jnp.take_along_axis(cpl, pick[:, None, None], axis=2)[:, :, 0]
            run_author = run_author | (col_author & live[:, None])
            run_cpl = jnp.maximum(run_cpl, jnp.where(live[:, None], col_cpl, 0))
            pick_emb = jnp.take_along_axis(pool_emb, pick[:, None, None], axis=1)[:, 0, :]
            seq = seq.at[:, t].set(jnp.where(live[:, None], pick_emb.astype(jnp.float32), 0.0))
        return PageDecodeOutputs(picked=picked, picked_prob=picked_prob)


def _masked_corr(x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    m = mask.astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    dx = x - (x * m).sum() / n
    dy = y - (y * m).sum() / n
    cov = (dx * dy * m).sum() / n
    return cov / jnp.sqrt((dx * dx * m).sum() / n * ((dy * dy * m).sum() / n) + 1e-12)


def page_critic_only_loss(
    *,
    config: PageCriticOnlyConfig,
    emb_sorted: jax.Array,
    user_emb: jax.Array,
    slate_sorted: jax.Array,
    supervised_sorted: jax.Array,
    reward_sorted: jax.Array,
    return_sorted: jax.Array,
    explore_page: jax.Array,
    auth_sorted: jax.Array,
    sids_sorted: jax.Array,
    gamma: float,
) -> PageCriticOnlyOutputs:
    scale = config.return_scale
    q = PageCriticTransformer(config)(emb_sorted, slate_sorted, user_emb, auth_sorted, sids_sorted)
    target = (
        page_critic_td_target(
            td_lambda=config.td_lambda,
            gamma=gamma,
            q_served=q * scale,
            reward=reward_sorted,
            step_mask=slate_sorted,
            impressed=supervised_sorted,
        )
        / scale
    )
    weight = supervised_sorted.astype(jnp.float32) * jnp.where(
        explore_page[:, None], config.exploration_return_weight, 1.0
    )
    td_loss = masked_mean(optax.huber_loss(q, target, delta=config.huber_delta), weight)

    g_scaled = return_sorted / scale
    slot0 = supervised_sorted & (jnp.arange(q.shape[1]) == 0)[None, :]
    stats = {
        "page_rl_critic_only/td_loss": td_loss,
        "page_rl_critic_only/return_rmse_scaled": jnp.sqrt(
            masked_mean(jnp.square(q - g_scaled), supervised_sorted)
        ),
        "page_rl_critic_only/return_rmse_scaled_slot0": jnp.sqrt(
            masked_mean(jnp.square(q - g_scaled), slot0)
        ),
        "page_rl_critic_only/q_mean": masked_mean(q * scale, supervised_sorted),
        "page_rl_critic_only/return_mean": masked_mean(return_sorted, supervised_sorted),
        "page_rl_critic_only/q_return_corr": _masked_corr(q, g_scaled, supervised_sorted),
    }
    return PageCriticOnlyOutputs(total=td_loss, stats=stats)
