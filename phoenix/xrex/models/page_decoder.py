# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.models.model_utils import get_parameter
from xrex.models.normalization import RMSNorm
from xrex.models.page_critic import masked_mean


@configclass
class PageDecoderConfig(Config):
    dec_dim: int = 256
    num_heads: int = 8
    max_steps: int = 48
    serve_steps: int = 12
    beam_width: int = 4
    score_anchor_alpha: float = 1.0
    logit_cap: float = 30.0
    anchor_norm_top_k: int = 12

    def decode_steps(self, pool: int) -> int:
        return min(pool, self.serve_steps)


@configclass
class PagePgConfig(Config):
    kl_loss_weight: float = 0.3
    pi_b_tau: float = 0.2
    kl_beta: float = 1.0
    kl_target: float = 0.35
    kl_beta_min: float = 0.1
    kl_beta_max: float = 3.0
    q_adv_weight: float = 0.5
    q_steps: int = 16
    q_adv_support_gap: float = 0.6
    comp_penalty_weight: float = 0.5


class PageDecoderLogits(NamedTuple):
    step_logits: jax.Array
    step_mask: jax.Array


class PageDecodeOutputs(NamedTuple):
    picked: jax.Array
    picked_prob: jax.Array


class PagePgOutputs(NamedTuple):
    total: jax.Array
    stats: dict[str, jax.Array]


def anchor_norm_mask(behavior_score: jax.Array, pool_mask: jax.Array, top_k: int) -> jax.Array:
    P = behavior_score.shape[-1]
    if top_k >= P:
        return pool_mask
    masked = jnp.where(pool_mask, behavior_score, -jnp.inf)
    rank = (masked[..., None, :] > masked[..., :, None]).sum(axis=-1)
    return pool_mask & (rank < top_k)


def anchor_standardize(
    behavior_score: jax.Array, apply_mask: jax.Array, norm_mask: jax.Array
) -> jax.Array:
    m = norm_mask.astype(jnp.float32)
    n = jnp.maximum(m.sum(axis=-1, keepdims=True), 1.0)
    mu = (behavior_score * m).sum(axis=-1, keepdims=True) / n
    var = (((behavior_score - mu) ** 2) * m).sum(axis=-1, keepdims=True) / n
    return jnp.where(apply_mask, (behavior_score - mu) / jnp.sqrt(var + 1e-6), 0.0)


def pairwise_same_author(auth: jax.Array, valid: jax.Array) -> jax.Array:
    S = auth.shape[-1]
    known = (auth != 0) & valid
    same = (auth[:, :, None] == auth[:, None, :]) & known[:, :, None] & known[:, None, :]
    return same & ~jnp.eye(S, dtype=bool)[None]


def pairwise_sid_common_prefix(sids: jax.Array, valid: jax.Array) -> jax.Array:
    S = sids.shape[1]
    s = sids.astype(jnp.int32)
    eq = (s[:, :, None, :] == s[:, None, :, :]) & (s[:, :, None, :] != 0)
    cpl = jnp.cumprod(eq.astype(jnp.int32), axis=-1).sum(axis=-1)
    ok = valid[:, :, None] & valid[:, None, :] & ~jnp.eye(S, dtype=bool)[None]
    return jnp.where(ok, cpl, 0)


def _prefix_max(x: jax.Array) -> jax.Array:
    cum = jax.lax.cummax(x, axis=2)
    shifted = jnp.concatenate([jnp.zeros_like(cum[:, :, :1]), cum[:, :, :-1]], axis=2)
    return shifted.transpose(0, 2, 1)


def overlap_features(author: jax.Array, cpl: jax.Array, sid_depths: int) -> jax.Array:
    sid = cpl[..., None] >= jnp.arange(1, sid_depths + 1, dtype=cpl.dtype)
    return jnp.concatenate([author[..., None], sid], axis=-1).astype(jnp.float32)


def overlap_prefix_features(
    same_author: jax.Array,
    cpl: jax.Array,
    sid_depths: int,
) -> jax.Array:
    return overlap_features(
        _prefix_max(same_author.astype(jnp.int32)), _prefix_max(cpl), sid_depths
    )


class _DecodeState(NamedTuple):
    z_buf: jax.Array
    k_cache: jax.Array
    v_cache: jax.Array
    picked_mask: jax.Array
    picked: jax.Array
    picked_prob: jax.Array
    run_author: jax.Array
    run_cpl: jax.Array


class _Pool(NamedTuple):
    keys: jax.Array
    vals: jax.Array
    score_z: jax.Array
    mask: jax.Array
    same_author: jax.Array
    cpl: jax.Array
    n_steps: jax.Array


def _signed_weights(name: str, signs: tuple[float, ...]) -> jax.Array:
    theta = get_parameter(
        name,
        [len(signs)],
        dtype=jnp.float32,
        init=hk.initializers.Constant(math.log(math.expm1(1e-3))),
        pspec=P(None),
    )
    return jnp.asarray(signs, dtype=jnp.float32) * jax.nn.softplus(theta)


class PageDecoder(hk.Module):
    def __init__(self, config: PageDecoderConfig, name: str = "page_decoder"):
        super().__init__(name=name)
        self._config = config

    def _param(self, name: str, shape: list[int]) -> jax.Array:
        return get_parameter(
            name,
            shape,
            dtype=jnp.float32,
            init=hk.initializers.VarianceScaling(1.0, mode="fan_in")
            if len(shape) > 1
            else hk.initializers.RandomNormal(0.02),
            pspec=P(*([None] * len(shape))),
        )

    def _weights(self, cand_dim: int, user_dim: int, sid_depths: int) -> dict[str, jax.Array]:
        cfg = self._config
        d = cfg.dec_dim
        return {
            "key": self._param("decoder_key_w", [cand_dim, d]),
            "val": self._param("prefix_val_w", [cand_dim, d]),
            "user": self._param("user_proj_w", [user_dim, d]),
            "step_emb": self._param("step_emb", [cfg.max_steps, d]),
            "q": self._param("attn_q_w", [d, d]),
            "k": self._param("attn_k_w", [d, d]),
            "v": self._param("attn_v_w", [d, d]),
            "o": self._param("attn_o_w", [d, d]),
            "dec_q": self._param("decoder_query_w", [d, d]),
            "overlap": _signed_weights("overlap_theta", (-1.0,) * (1 + sid_depths)),
        }

    @staticmethod
    def _norms() -> tuple[RMSNorm, RMSNorm]:
        return (
            RMSNorm(axis=-1, create_scale=True, pspec=P(None), name="prefix_ln"),
            RMSNorm(axis=-1, create_scale=True, pspec=P(None), name="decoder_ln"),
        )

    def _encode(
        self,
        cand_emb: jax.Array,
        user_emb: jax.Array,
        auth: jax.Array,
        sids: jax.Array,
        mask: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        w = self._weights(cand_emb.shape[-1], user_emb.shape[-1], sids.shape[-1])
        x_in = cand_emb.astype(jnp.float32)
        keys = jnp.einsum("bsd,dk->bsk", x_in, w["key"])
        vals = jnp.einsum("bsd,dk->bsk", x_in, w["val"])
        z0 = jnp.einsum("bd,dk->bk", user_emb.astype(jnp.float32), w["user"])
        same_author = pairwise_same_author(auth.astype(jnp.int32), mask)
        return w, keys, vals, z0, same_author, pairwise_sid_common_prefix(sids, mask)

    def _qkv(
        self, z: jax.Array, w: dict[str, jax.Array], prefix_ln: RMSNorm
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        zn = prefix_ln(z)
        h = self._config.num_heads
        q, k, v = (
            jnp.einsum("...d,de->...e", zn, w[n]).reshape(*z.shape[:-1], h, -1) for n in "qkv"
        )
        return q, k, v

    def _attend(
        self,
        z: jax.Array,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array,
        w: dict[str, jax.Array],
    ) -> jax.Array:
        attn = jnp.einsum("b...he,bshe->bh...s", q, k) / (q.shape[-1] ** 0.5)
        attn = jax.nn.softmax(jnp.where(mask, attn, -jnp.inf), axis=-1)
        ctx = jnp.einsum("bh...s,bshe->b...he", attn, v).reshape(z.shape)
        return z + jnp.einsum("...d,de->...e", ctx, w["o"])

    def _candidate_logits(
        self,
        state: jax.Array,
        keys: jax.Array,
        score_z: jax.Array,
        feats: jax.Array,
        w: dict[str, jax.Array],
        dec_ln: RMSNorm,
    ) -> jax.Array:
        cfg = self._config
        dec_q = jnp.einsum("...d,de->...e", dec_ln(state), w["dec_q"])
        logits = jnp.einsum("b...e,bse->b...s", dec_q, keys) / (keys.shape[-1] ** 0.5)
        logits = logits + cfg.score_anchor_alpha * score_z
        logits = logits + jnp.einsum("...k,k->...", feats, w["overlap"])
        return cfg.logit_cap * jnp.tanh(logits / cfg.logit_cap)

    def _init_state(self, z0: jax.Array, step_emb: jax.Array, S: int, T: int) -> _DecodeState:
        cfg = self._config
        B, d = z0.shape
        h = cfg.num_heads
        z_buf = jnp.zeros((B, T, d), dtype=jnp.float32)
        z_buf = z_buf.at[:, 0, :].set(z0 + step_emb[None, 0, :])
        return _DecodeState(
            z_buf=z_buf,
            k_cache=jnp.zeros((B, T, h, d // h), dtype=jnp.float32),
            v_cache=jnp.zeros((B, T, h, d // h), dtype=jnp.float32),
            picked_mask=jnp.zeros((B, S), dtype=bool),
            picked=jnp.full((B, T), -1, dtype=jnp.int32),
            picked_prob=jnp.zeros((B, T), dtype=jnp.float32),
            run_author=jnp.zeros((B, S), dtype=bool),
            run_cpl=jnp.zeros((B, S), dtype=jnp.int32),
        )

    def _step_logits(
        self,
        st: _DecodeState,
        t: int | jax.Array,
        w: dict[str, jax.Array],
        norms: tuple[RMSNorm, RMSNorm],
        pool: _Pool,
    ) -> tuple[jax.Array, _DecodeState]:
        prefix_ln, dec_ln = norms
        z_t = st.z_buf[:, t, :]
        q_t, k_t, v_t = self._qkv(z_t, w, prefix_ln)
        st = st._replace(k_cache=st.k_cache.at[:, t].set(k_t), v_cache=st.v_cache.at[:, t].set(v_t))
        seen = (jnp.arange(st.z_buf.shape[1]) <= t)[None, None, :]
        state_t = self._attend(z_t, q_t, st.k_cache, st.v_cache, seen, w)
        feats_t = overlap_features(st.run_author, st.run_cpl, w["overlap"].shape[0] - 1)
        logits_t = self._candidate_logits(state_t, pool.keys, pool.score_z, feats_t, w, dec_ln)
        return jnp.where(st.picked_mask | ~pool.mask, -jnp.inf, logits_t), st

    def _advance(
        self,
        st: _DecodeState,
        t: int | jax.Array,
        pick_t: jax.Array,
        prob_t: jax.Array,
        pool: _Pool,
        step_emb: jax.Array,
    ) -> _DecodeState:
        valid_t = t < pool.n_steps
        picked = st.picked.at[:, t].set(jnp.where(valid_t, pick_t.astype(jnp.int32), -1))
        picked_prob = st.picked_prob.at[:, t].set(jnp.where(valid_t, prob_t, 0.0))
        pick_1h = jax.nn.one_hot(pick_t, pool.mask.shape[1], dtype=jnp.bool_) & valid_t[:, None]
        col_a = jnp.take_along_axis(pool.same_author, pick_t[:, None, None], axis=2)[:, :, 0]
        col_c = jnp.take_along_axis(pool.cpl, pick_t[:, None, None], axis=2)[:, :, 0]
        T = st.z_buf.shape[1]
        nxt = jnp.minimum(t + 1, T - 1)
        v_pick = jnp.take_along_axis(pool.vals, pick_t[:, None, None], axis=1)[:, 0, :]
        z_next = jnp.where(t + 1 < T, v_pick + step_emb[None, nxt, :], st.z_buf[:, nxt, :])
        return _DecodeState(
            z_buf=st.z_buf.at[:, nxt, :].set(z_next),
            k_cache=st.k_cache,
            v_cache=st.v_cache,
            picked_mask=st.picked_mask | pick_1h,
            picked=picked,
            picked_prob=picked_prob,
            run_author=st.run_author | (col_a & valid_t[:, None]),
            run_cpl=jnp.maximum(st.run_cpl, jnp.where(valid_t[:, None], col_c, 0)),
        )

    def _decode(
        self,
        w: dict[str, jax.Array],
        norms: tuple[RMSNorm, RMSNorm],
        pool: _Pool,
        z0: jax.Array,
        T: int,
        choose: Callable[..., tuple[_DecodeState, jax.Array, jax.Array, jax.Array | None]],
        carry: jax.Array | None = None,
    ) -> tuple[_DecodeState, jax.Array | None]:
        def step(t, val):
            st, carry = val
            logits_t, st = self._step_logits(st, t, w, norms, pool)
            st, pick_t, prob_t, carry = choose(t, logits_t, st, carry)
            return self._advance(st, t, pick_t, prob_t, pool, w["step_emb"]), carry

        st = self._init_state(z0, w["step_emb"], pool.mask.shape[1], T)
        return hk.fori_loop(0, T, step, (st, carry))

    def _autoregressive(
        self,
        w: dict[str, jax.Array],
        norms: tuple[RMSNorm, RMSNorm],
        pool: _Pool,
        z0: jax.Array,
        tau: jax.Array,
        rng: jax.Array,
        T: int,
    ) -> PageDecodeOutputs:
        def choose(t, logits_t, st, carry):
            gumbel = jax.random.gumbel(
                jax.random.fold_in(rng, t), logits_t.shape, dtype=jnp.float32
            )
            pick_t = jnp.argmax(logits_t + tau[:, None] * gumbel, axis=-1)
            shifted = (logits_t - jnp.max(logits_t, axis=-1, keepdims=True)) / jnp.maximum(
                tau[:, None], 1e-6
            )
            p = jnp.exp(shifted)
            p = p / jnp.maximum(p.sum(axis=-1, keepdims=True), 1e-9)
            return st, pick_t, jnp.take_along_axis(p, pick_t[:, None], axis=1)[:, 0], carry

        st, _ = self._decode(w, norms, pool, z0, T, choose)
        return PageDecodeOutputs(picked=st.picked, picked_prob=st.picked_prob)

    def _beam_search(
        self,
        w: dict[str, jax.Array],
        norms: tuple[RMSNorm, RMSNorm],
        pool: _Pool,
        z0: jax.Array,
        width: int,
        T: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        B, S = pool.mask.shape
        W = width
        cand_idx = jnp.arange(S, dtype=jnp.int32)
        pool_b = jax.tree.map(lambda x: jnp.repeat(x, W, axis=0), pool)

        def _from_parent(x: jax.Array, parent: jax.Array) -> jax.Array:
            xr = x.reshape(B, W, *x.shape[1:])
            return jax.vmap(lambda xb, pb: jnp.take(xb, pb, axis=0))(xr, parent).reshape(x.shape)

        def choose(t, logits_t, st, cum):
            finite = jnp.isfinite(logits_t)
            safe = jnp.where(finite, logits_t, -1e30)
            logp = jnp.where(
                finite, safe - jax.nn.logsumexp(safe, axis=-1, keepdims=True), -jnp.inf
            )
            scores = jnp.where(
                (t < pool_b.n_steps)[:, None],
                cum[:, None] + logp,
                jnp.where(cand_idx[None, :] == 0, cum[:, None], -jnp.inf),
            ).reshape(B, W * S)
            top_scores, top_flat = jax.lax.top_k(scores, W)
            st = jax.tree.map(functools.partial(_from_parent, parent=top_flat // S), st)
            prob_t = jnp.ones(B * W, jnp.float32)
            return st, (top_flat % S).reshape(-1), prob_t, top_scores.reshape(-1)

        cum = jnp.tile(jnp.where(jnp.arange(W) == 0, 0.0, -jnp.inf).astype(jnp.float32), (B,))
        st, cum = self._decode(w, norms, pool_b, jnp.repeat(z0, W, axis=0), T, choose, cum)
        return st.picked.reshape(B, W, T), st.picked_prob.reshape(B, W, T), cum.reshape(B, W)

    def __call__(
        self,
        cand_emb_sorted: jax.Array,
        slate_mask_sorted: jax.Array,
        behavior_score_sorted: jax.Array,
        user_emb: jax.Array,
        auth_sorted: jax.Array,
        sids_sorted: jax.Array,
    ) -> PageDecoderLogits:
        cfg = self._config
        S = cand_emb_sorted.shape[1]
        assert cfg.dec_dim % cfg.num_heads == 0
        assert S <= cfg.max_steps, f"S={S} exceeds max_steps={cfg.max_steps}"

        w, keys, vals, z0, same_author, cpl = self._encode(
            cand_emb_sorted, user_emb, auth_sorted, sids_sorted, slate_mask_sorted
        )
        prefix_ln, dec_ln = self._norms()

        z = jnp.concatenate([z0[:, None, :], vals[:, : S - 1, :]], axis=1)
        z = z + w["step_emb"][None, :S, :]

        causal = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None]
        state = self._attend(z, *self._qkv(z, w, prefix_ln), causal, w)
        score_z = anchor_standardize(behavior_score_sorted, slate_mask_sorted, slate_mask_sorted)
        feats = overlap_prefix_features(same_author, cpl, sids_sorted.shape[-1])
        logits = self._candidate_logits(state, keys, score_z[:, None, :], feats, w, dec_ln)

        steps = jnp.arange(S)
        target_ok = (steps[None, :, None] <= steps[None, None, :]) & slate_mask_sorted[:, None, :]
        logits = jnp.where(target_ok, logits, -jnp.inf)
        return PageDecoderLogits(step_logits=logits, step_mask=slate_mask_sorted)

    @hk.name_like("__call__")
    def decode(
        self,
        cand_emb: jax.Array,
        pool_mask: jax.Array,
        behavior_score: jax.Array,
        user_emb: jax.Array,
        steps: jax.Array,
        temperature: jax.Array,
        rng: jax.Array,
        auth_hashes: jax.Array,
        post_sids: jax.Array,
        slot_values: jax.Array | None = None,
    ) -> PageDecodeOutputs:
        cfg = self._config
        B, S, _ = cand_emb.shape
        assert cfg.dec_dim % cfg.num_heads == 0
        assert S <= cfg.max_steps, f"P={S} exceeds max_steps={cfg.max_steps}"
        assert cfg.beam_width <= 1 or slot_values is not None

        w, keys, vals, z0, same_author, cpl = self._encode(
            cand_emb, user_emb, auth_hashes, post_sids, pool_mask
        )
        norms = self._norms()
        norm_mask = anchor_norm_mask(behavior_score, pool_mask, cfg.anchor_norm_top_k)
        score_z = anchor_standardize(behavior_score, pool_mask, norm_mask)

        T = cfg.decode_steps(S)
        pool_count = pool_mask.sum(axis=1).astype(jnp.int32)
        n_steps = jnp.minimum(jnp.minimum(steps.astype(jnp.int32), pool_count), T)
        pool = _Pool(keys, vals, score_z, pool_mask, same_author, cpl, n_steps)
        tau = jnp.maximum(temperature.astype(jnp.float32), 0.0)
        if cfg.beam_width <= 1:
            return self._autoregressive(w, norms, pool, z0, tau, rng, T)
        assert slot_values is not None
        beams, beam_probs, cum_logp = self._beam_search(w, norms, pool, z0, cfg.beam_width, T)
        picked_values = slot_values[
            jnp.arange(B)[:, None, None], jnp.maximum(beams, 0), jnp.arange(T)[None, None, :]
        ]
        beam_value = jnp.where(beams >= 0, picked_values, 0.0).sum(axis=-1)
        values = jnp.where(jnp.isfinite(cum_logp), beam_value, -jnp.inf)
        best = jnp.argmax(values, axis=-1)
        rows = jnp.arange(B, dtype=jnp.int32)
        greedy = PageDecodeOutputs(picked=beams[rows, best], picked_prob=beam_probs[rows, best])
        use_beam = (tau <= 0.0)[:, None]

        def with_sampled_rows() -> PageDecodeOutputs:
            sampled = self._autoregressive(w, norms, pool, z0, tau, rng, T)
            return jax.tree.map(lambda g, s: jnp.where(use_beam, g, s), greedy, sampled)

        return hk.cond(jnp.all(use_beam), lambda: greedy, with_sampled_rows)


def _masked_log_softmax(logits: jax.Array) -> jax.Array:
    out = jax.nn.log_softmax(logits, axis=-1)
    return jnp.where(jnp.isfinite(out), out, -1e9)


def decoder_bc_loss(outputs: PageDecoderLogits) -> tuple[jax.Array, dict[str, jax.Array]]:
    diag = jnp.diagonal(_masked_log_softmax(outputs.step_logits), axis1=1, axis2=2)
    bc_loss = -masked_mean(diag, outputs.step_mask)
    top1 = jnp.argmax(outputs.step_logits, axis=-1) == jnp.arange(outputs.step_logits.shape[1])
    return bc_loss, {
        "page_rl/bc_loss": bc_loss,
        "page_rl/bc_top1_agreement": masked_mean(top1, outputs.step_mask),
    }


def page_pg_loss(
    *,
    config: PagePgConfig,
    decoder_out: PageDecoderLogits,
    behavior_score_sorted: jax.Array,
    final_score_sorted: jax.Array,
    impressed_sorted: jax.Array,
    logged_prob_sorted: jax.Array,
    decode_prob_sorted: jax.Array,
    q_grid_sorted: jax.Array,
    sids_sorted: jax.Array,
    auth_sorted: jax.Array,
) -> PagePgOutputs:
    logits = decoder_out.step_logits
    step_mask = decoder_out.step_mask
    S = logits.shape[1]
    m = step_mask.astype(jnp.float32)
    denom = jnp.maximum(m.sum(), 1.0)
    impressed = impressed_sorted & step_mask
    target_ok = jnp.isfinite(logits)

    logp = _masked_log_softmax(logits)
    p = jnp.exp(logp)

    def _b_logp(score: jax.Array) -> jax.Array:
        return _masked_log_softmax(
            jnp.where(target_ok, score[:, None, :].astype(jnp.float32) / config.pi_b_tau, -jnp.inf)
        )

    has_logged = (logged_prob_sorted > 0.0) & step_mask
    has_decode = (decode_prob_sorted > 0.0) & step_mask

    k_q = min(config.q_steps, S)
    within = jnp.arange(S) < k_q
    q_grid = jax.lax.stop_gradient(q_grid_sorted.astype(jnp.float32))
    q_ok = target_ok & step_mask[:, :, None] & within[None, :, None] & within[None, None, :]
    q_stepf = (q_ok.any(axis=-1) & impressed).astype(jnp.float32)
    ws = jnp.where(target_ok, behavior_score_sorted[:, None, :].astype(jnp.float32), -jnp.inf)
    near_best = (ws.max(axis=-1, keepdims=True) - ws) <= config.q_adv_support_gap
    a_okf = (q_ok & (near_best | jnp.eye(S, dtype=bool)[None])).astype(jnp.float32)

    def _over_support(logp_full: jax.Array) -> jax.Array:
        dist = jnp.exp(logp_full) * a_okf
        return dist / jnp.maximum(dist.sum(axis=-1, keepdims=True), 1e-9)

    v_b = (_over_support(_b_logp(behavior_score_sorted)) * q_grid).sum(axis=-1)
    a_q = (q_grid - v_b[:, :, None]) * a_okf
    q_pairf = a_okf * q_stepf[:, :, None]
    a_q_rms = jnp.sqrt(masked_mean(jnp.square(a_q), q_pairf) + 1e-4)
    exp_a_q = (_over_support(logp) * (a_q / a_q_rms)).sum(axis=-1)
    q_loss = -masked_mean(exp_a_q, q_stepf)

    kl_score = jnp.where(
        has_logged.any(axis=-1, keepdims=True), behavior_score_sorted, final_score_sorted
    )
    kl_terms = optax.losses.kl_divergence_with_log_targets(_b_logp(kl_score), logp, where=target_ok)
    kl_w = m * jnp.where(has_decode.any(axis=-1, keepdims=True), 0.0, 1.0)
    kl_loss = (kl_terms * kl_w).sum() / denom
    kl_mean = masked_mean(kl_terms, kl_w)

    lo, hi = config.kl_beta_min, config.kl_beta_max
    assert 0.0 < lo < config.kl_beta < hi, (lo, config.kl_beta, hi)
    init_frac = (config.kl_beta - lo) / (hi - lo)
    kl_beta_u = get_parameter(
        "kl_beta_u",
        [],
        dtype=jnp.float32,
        init=hk.initializers.Constant(math.log(init_frac / (1.0 - init_frac))),
        pspec=P(),
    )
    kl_beta = lo + (hi - lo) * jax.nn.sigmoid(kl_beta_u)
    kl_dual_loss = kl_beta * jax.lax.stop_gradient(config.kl_target - kl_mean)
    kl_beta_eff = jax.lax.stop_gradient(kl_beta)

    sim = jnp.maximum(
        pairwise_sid_common_prefix(sids_sorted, step_mask).astype(jnp.float32)
        / sids_sorted.shape[-1],
        pairwise_same_author(auth_sorted, step_mask).astype(jnp.float32),
    )
    exp_pen = (p * _prefix_max(sim)).sum(axis=-1)
    reach = (impressed.astype(jnp.float32) * m).sum(axis=0) / jnp.maximum(m.sum(axis=0), 1.0)
    reach_w = reach[None, :] * m
    reach_w = reach_w * (denom / jnp.maximum(reach_w.sum(), 1e-6))
    comp_penalty = (exp_pen * reach_w).sum() / denom

    total = (
        config.kl_loss_weight * (kl_beta_eff * kl_loss + kl_dual_loss)
        + config.q_adv_weight * q_loss
        + config.comp_penalty_weight * comp_penalty
    )
    stats = {
        "page_rl/q_loss": q_loss,
        "page_rl/kl_to_pi_b": kl_loss,
        "page_rl/kl_leash_mean": kl_mean,
        "page_rl/kl_beta_eff": kl_beta_eff,
        "page_rl/comp_penalty": comp_penalty,
    }
    return PagePgOutputs(total=total, stats=stats)
