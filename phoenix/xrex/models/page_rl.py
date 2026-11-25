# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xai_proto import recsys_pb2
from xrex.models.model_utils import get_parameter
from xrex.models.page_critic import (
    PageCriticConfig,
    discounted_suffix_sum,
    page_critic_loss,
    page_critic_value_matrix,
)
from xrex.models.page_critic_only import (
    PageCriticOnlyConfig,
    PageCriticTransformer,
    page_critic_only_loss,
)
from xrex.models.page_decoder import (
    PageDecoder,
    PageDecoderConfig,
    PagePgConfig,
    decoder_bc_loss,
    page_pg_loss,
)
from xrex.models.recsys_feature_prep import IMMERSIVE_NOT_DWELLED_SEC

PAGE_PARAM_REGEX = "page_critic|page_decoder|kl_beta_u"
USER_DEBIAS_PARAM_REGEX = "page_user_debias"

USER_DEBIAS_ACTIONS: tuple[int, ...] = (
    recsys_pb2.ActionName.SERVER_TWEET_FAV,
    recsys_pb2.ActionName.SERVER_TWEET_REPLY,
    recsys_pb2.ActionName.SERVER_TWEET_RETWEET,
    recsys_pb2.ActionName.SERVER_TWEET_QUOTE,
    recsys_pb2.ActionName.CLIENT_TWEET_CLICK,
    recsys_pb2.ActionName.CLIENT_TWEET_SHARE,
)
USER_DEBIAS_OUTPUTS = len(USER_DEBIAS_ACTIONS) + 1
USER_DEBIAS_DWELL_CAP_S = 300.0

REALIZED_POSITIVE_ACTIONS: tuple[str, ...] = (
    "ServerTweetFav",
    "ServerTweetReply",
    "ServerTweetRetweet",
    "ServerTweetQuote",
    "ClientTweetBookmark",
    "ClientTweetShare",
    "ClientTweetFollowAuthor",
    "ClientTweetClickProfile",
    "ClientTweetClick",
    "ClientTweetRecapDwelled",
)
REALIZED_NEGATIVE_ACTIONS: tuple[str, ...] = (
    "ClientTweetNotInterestedIn",
    "ClientTweetReport",
    "ServerTweetReport",
    "ClientTweetBlockAuthor",
    "ClientTweetMuteAuthor",
    "ClientTweetSeeFewer",
)

VIDEO_COMPLETED_ACTION = recsys_pb2.ActionName.CLIENT_NOTIFICATION_SENT


@configclass
class VideoCompletionLabelConfig(Config):
    floor: float = 0.35
    top: float = 1.4
    d0: float = 50.0
    k: float = 2.0
    cap_factor: float = 3.0
    cap_floor_s: float = 10.0


def video_completion_labels(
    dwell_s: jax.Array, duration_ms: jax.Array, cfg: VideoCompletionLabelConfig
) -> tuple[jax.Array, jax.Array]:
    dwell_s = dwell_s.astype(jnp.float32)
    duration_ms = duration_ms.astype(jnp.float32)
    d = duration_ms / 1000.0
    p = cfg.floor + (cfg.top - cfg.floor) / (1.0 + (d / cfg.d0) ** cfg.k)
    cap = cfg.cap_factor * jnp.maximum(d, cfg.cap_floor_s)
    trains = (duration_ms > 0.0) & (dwell_s >= IMMERSIVE_NOT_DWELLED_SEC)
    return trains, (dwell_s >= p * d) & (dwell_s <= cap)


@configclass
class PageRewardConfig(Config):
    min_dwell_s: float = 3.0
    dwell_cap_s: float = 300.0
    engagement_count_cap: float = 3.0
    engagement_offset: float = 1.0
    engagement_beta: float = 1.0
    completion_bonus: float = 1.0
    gamma: float = 0.8


@configclass
class PageServeScoreConfig(Config):
    favorite_weight: float = 0.5
    reply_weight: float = 2.0
    retweet_weight: float = 1.0
    click_weight: float = 0.4
    profile_click_weight: float = 0.5
    vqv_weight: float = 1.0
    share_weight: float = 4.0
    share_via_dm_weight: float = 0.0
    share_via_copy_link_weight: float = 0.0
    quote_weight: float = 5.0
    completion_weight: float = 0.0
    cont_dwell_time_weight: float = 0.1
    not_interested_weight: float = -500.0
    report_weight: float = -1000.0
    not_dwelled_weight: float = -1.2
    min_video_duration_ms: float = 10_000.0
    negative_offset: float = 0.001


RANKING_SCORE_WEIGHT_FIELDS: tuple[str, ...] = (
    "favorite_weight",
    "reply_weight",
    "retweet_weight",
    "click_weight",
    "profile_click_weight",
    "vqv_weight",
    "share_weight",
    "quote_weight",
    "cont_dwell_time_weight",
    "not_interested_weight",
    "report_weight",
    "not_dwelled_weight",
    "min_video_duration_ms",
    "share_via_dm_weight",
    "share_via_copy_link_weight",
    "completion_weight",
)

PAGE_SERVE_DECODE_MODES = ("actor", "critic")


@configclass
class PageServeDecodeConfig(Config):
    mode: str = "actor"
    eta: float = 1.0
    nu_author: float = 0.0
    nu_sid: float = 0.0
    prior_tau: float = 0.2
    max_pool: int = 0


@configclass
class PageUserDebiasConfig(Config):
    enabled: bool = False
    hidden_dim: int = 256
    clip_logit: float = 1.0
    serve_heads: bool = False
    serve_dwell: bool = False


@configclass
class PageRlConfig(Config):
    enabled: bool = False
    reward: PageRewardConfig = PageRewardConfig()
    critic: PageCriticConfig = PageCriticConfig()
    decoder: PageDecoderConfig = PageDecoderConfig()
    pg: PagePgConfig = PagePgConfig()
    serve_score: PageServeScoreConfig = PageServeScoreConfig()
    user_debias: PageUserDebiasConfig = PageUserDebiasConfig()
    bc_loss_weight: float = 0.25
    critic_only: PageCriticOnlyConfig | None = None
    serve_decode: PageServeDecodeConfig = PageServeDecodeConfig()


class PageRlFields(NamedTuple):
    slate_mask: jax.Array
    impressed_mask: jax.Array
    reward: jax.Array
    return_g: jax.Array
    adv_label: jax.Array
    slot_order: jax.Array


def compute_page_rl_fields(
    *,
    rl_served_slot: jax.Array,
    impressed: jax.Array,
    dwell_s: jax.Array,
    not_dwelled: jax.Array,
    vqv_observed: jax.Array,
    engagement_count: jax.Array,
    negative_any: jax.Array,
    dwell_weight: float,
    config: PageRewardConfig,
) -> PageRlFields:
    num_candidates = rl_served_slot.shape[1]
    slate_mask = rl_served_slot >= 0
    impressed_mask = slate_mask & impressed

    dwell = jnp.clip(dwell_s.astype(jnp.float32), 0.0, config.dwell_cap_s)
    engagement = jnp.minimum(engagement_count.astype(jnp.float32), config.engagement_count_cap)
    mult = jnp.where(
        negative_any, 0.0, config.engagement_offset + config.engagement_beta * engagement
    )
    rewarded = impressed_mask & (dwell >= config.min_dwell_s)
    reward = jnp.where(rewarded, dwell_weight * dwell * mult, 0.0).astype(jnp.float32)

    sort_key = jnp.where(slate_mask, rl_served_slot, num_candidates + 1)
    slot_order = jnp.argsort(sort_key, axis=1, stable=True).astype(jnp.int32)

    def to_slot(x: jax.Array) -> jax.Array:
        return jnp.take_along_axis(x, slot_order, axis=1)

    reward_s = to_slot(reward)
    slate_s = to_slot(slate_mask)
    impressed_s = to_slot(impressed_mask)
    vqv_s = to_slot(vqv_observed & slate_mask)

    next_is_slate = jnp.concatenate([slate_s[:, 1:], jnp.zeros_like(slate_s[:, :1])], axis=1)
    is_last_slate = slate_s & ~next_is_slate
    all_impressed = ~jnp.any(slate_s & ~impressed_s, axis=1)
    completed = jnp.any(slate_s, axis=1) & all_impressed & jnp.any(is_last_slate & vqv_s, axis=1)

    next_impressed = jnp.concatenate(
        [impressed_s[:, 1:], jnp.zeros_like(impressed_s[:, :1])], axis=1
    )
    advanced_s = jnp.where(is_last_slate, completed[:, None], next_impressed)
    adv_label_s = advanced_s & ~to_slot(not_dwelled)

    reward_s = reward_s + jnp.where(
        is_last_slate & completed[:, None], config.completion_bonus, 0.0
    )
    return_s = jnp.where(slate_s, discounted_suffix_sum(reward_s, config.gamma), 0.0)

    return PageRlFields(
        slate_mask=slate_s,
        impressed_mask=impressed_s,
        reward=reward_s,
        return_g=return_s,
        adv_label=adv_label_s,
        slot_order=slot_order,
    )


def page_rl_loss_and_stats(
    *,
    config: PageRlConfig,
    candidate_embeddings: jax.Array,
    user_embeddings: jax.Array,
    rl_served_slot: jax.Array,
    rl_weighted_score: jax.Array,
    rl_final_score: jax.Array,
    rl_rerank_prob: jax.Array,
    rl_decode_prob: jax.Array,
    rl_impressed: jax.Array,
    dwell_s: jax.Array,
    not_dwelled: jax.Array,
    vqv_observed: jax.Array,
    engagement_count: jax.Array,
    negative_any: jax.Array,
    valid_mask: jax.Array,
    post_sids: jax.Array,
    auth_hashes: jax.Array,
    user_debias: UserDebiasBatch | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    fields = compute_page_rl_fields(
        rl_served_slot=rl_served_slot,
        impressed=rl_impressed,
        dwell_s=dwell_s,
        not_dwelled=not_dwelled,
        vqv_observed=vqv_observed,
        engagement_count=engagement_count,
        negative_any=negative_any,
        dwell_weight=config.serve_score.cont_dwell_time_weight,
        config=config.reward,
    )

    s_max = min(config.decoder.max_steps, candidate_embeddings.shape[1])
    order = fields.slot_order[:, :s_max]

    def _sorted(x: jax.Array) -> jax.Array:
        idx = order if x.ndim == 2 else order[:, :, None]
        return jnp.take_along_axis(x, idx, axis=1)

    emb_sorted = _sorted(jax.lax.stop_gradient(candidate_embeddings))
    user_emb = jax.lax.stop_gradient(user_embeddings)
    valid_sorted = _sorted(valid_mask)
    slate_sorted = fields.slate_mask[:, :s_max] & valid_sorted
    supervised_sorted = fields.impressed_mask[:, :s_max] & valid_sorted
    score_sorted = _sorted(rl_weighted_score.astype(jnp.float32))
    logged_prob_sorted = _sorted(rl_rerank_prob.astype(jnp.float32))
    explore_page = (logged_prob_sorted > 0.0).any(axis=1)
    reward_sorted = fields.reward[:, :s_max]
    return_sorted = fields.return_g[:, :s_max]
    sids_sorted = _sorted(post_sids)
    auth_sorted = _sorted(auth_hashes)

    debias = config.user_debias
    debias_loss, debias_stats = jnp.zeros((), jnp.float32), {}
    if debias.enabled:
        assert user_debias is not None, "page_rl.user_debias needs its training inputs"
        heads = PageUserDebias(debias)
        debias_loss, debias_stats = page_user_debias_loss(
            heads(user_emb, user_debias.stats), heads.global_values(), user_debias, dwell_s
        )

    critic_out = page_critic_loss(
        config=config.critic,
        emb_sorted=emb_sorted,
        slate_sorted=slate_sorted,
        served_slot_sorted=_sorted(rl_served_slot),
        supervised_sorted=supervised_sorted,
        adv_label_sorted=fields.adv_label[:, :s_max],
        reward_sorted=reward_sorted,
        return_sorted=return_sorted,
        explore_page=explore_page,
        gamma=config.reward.gamma,
        grid_steps=min(config.pg.q_steps, s_max),
    )

    decoder_out = PageDecoder(config.decoder)(
        emb_sorted, slate_sorted, score_sorted, user_emb, auth_sorted, sids_sorted
    )
    bc_loss, bc_stats = decoder_bc_loss(decoder_out)
    pg_out = page_pg_loss(
        config=config.pg,
        decoder_out=decoder_out,
        behavior_score_sorted=score_sorted,
        final_score_sorted=_sorted(rl_final_score.astype(jnp.float32)),
        impressed_sorted=supervised_sorted,
        logged_prob_sorted=logged_prob_sorted,
        decode_prob_sorted=_sorted(rl_decode_prob.astype(jnp.float32)),
        q_grid_sorted=critic_out.q_grid_sorted,
        sids_sorted=sids_sorted,
        auth_sorted=auth_sorted,
    )
    total = critic_out.total + config.bc_loss_weight * bc_loss + pg_out.total + debias_loss
    stats = {**critic_out.stats, **bc_stats, **pg_out.stats, **debias_stats}
    if config.critic_only is not None:
        critic_only_out = page_critic_only_loss(
            config=config.critic_only,
            emb_sorted=emb_sorted,
            user_emb=user_emb,
            slate_sorted=slate_sorted,
            supervised_sorted=supervised_sorted,
            reward_sorted=reward_sorted,
            return_sorted=return_sorted,
            explore_page=explore_page,
            auth_sorted=auth_sorted,
            sids_sorted=sids_sorted,
            gamma=config.reward.gamma,
        )
        total = total + critic_only_out.total
        stats.update(critic_only_out.stats)
    return total, stats


class PageDecodeInputs(NamedTuple):
    video_duration_ms: jax.Array
    n_candidates: jax.Array
    pool_n: jax.Array
    steps: jax.Array
    temperature: jax.Array
    ranking_score_weights: jax.Array
    eta: jax.Array | None = None
    nu_author: jax.Array | None = None
    nu_sid: jax.Array | None = None
    prior_tau: jax.Array | None = None
    not_dwelled_power_exponent: jax.Array | None = None
    not_dwelled_power_offset: jax.Array | None = None
    engage_bias: jax.Array | None = None
    multiplicative: jax.Array | None = None
    user_debias_heads: jax.Array | None = None

    @classmethod
    def no_decode(cls, n_users: int, n_candidates: int) -> PageDecodeInputs:
        unset = jnp.full((n_users,), jnp.nan, dtype=jnp.float32)
        return cls(
            video_duration_ms=jnp.zeros((n_users, n_candidates), dtype=jnp.float32),
            n_candidates=jnp.zeros((n_users,), dtype=jnp.int32),
            pool_n=jnp.zeros((n_users,), dtype=jnp.int32),
            steps=jnp.zeros((n_users,), dtype=jnp.int32),
            temperature=jnp.zeros((n_users,), dtype=jnp.float32),
            ranking_score_weights=jnp.full(
                (n_users, len(RANKING_SCORE_WEIGHT_FIELDS)), jnp.nan, dtype=jnp.float32
            ),
            eta=unset,
            nu_author=unset,
            nu_sid=unset,
            prior_tau=unset,
            not_dwelled_power_exponent=unset,
            not_dwelled_power_offset=unset,
            engage_bias=unset,
            multiplicative=unset,
            user_debias_heads=jnp.full((n_users, USER_DEBIAS_OUTPUTS), jnp.nan, dtype=jnp.float32),
        )


_HEAD_FEATURE_ACTIONS = (
    recsys_pb2.ActionName.SERVER_TWEET_FAV,
    recsys_pb2.ActionName.SERVER_TWEET_REPLY,
    recsys_pb2.ActionName.SERVER_TWEET_RETWEET,
    recsys_pb2.ActionName.CLIENT_TWEET_CLICK,
    recsys_pb2.ActionName.CLIENT_TWEET_CLICK_PROFILE,
    recsys_pb2.ActionName.CLIENT_TWEET_VIDEO_QUALITY_VIEW,
    recsys_pb2.ActionName.CLIENT_TWEET_SHARE,
    recsys_pb2.ActionName.SERVER_TWEET_QUOTE,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_INTERESTED_IN,
    recsys_pb2.ActionName.CLIENT_TWEET_REPORT,
)


def page_rl_head_features(
    logits: jax.Array,
    cont_preds: jax.Array,
    dwell_norm_scale: float,
) -> jax.Array:
    heads = jax.nn.sigmoid(logits[..., list(_HEAD_FEATURE_ACTIONS)].astype(jnp.float32))
    dwell_s = cont_preds[..., recsys_pb2.ContinuousActionName.DWELL_TIME].astype(jnp.float32)
    dwell = jnp.clip(dwell_s / dwell_norm_scale, 0.0, 1.0)
    return jnp.concatenate([heads, dwell[..., None]], axis=-1)


def _row_or_default(values: jax.Array | None, default: float) -> jax.Array:
    if values is None:
        return jnp.asarray(default, dtype=jnp.float32)
    x = values.astype(jnp.float32)[:, None]
    return jnp.where(jnp.isnan(x), default, x)


def page_serve_weighted_score(
    cfg: PageServeScoreConfig,
    logits: jax.Array,
    cont_preds: jax.Array,
    video_duration_ms: jax.Array,
    request_weights: jax.Array | None = None,
    power_exponent: jax.Array | None = None,
    power_offset: jax.Array | None = None,
    engage_bias: jax.Array | None = None,
    multiplicative: jax.Array | None = None,
) -> jax.Array:
    an = recsys_pb2.ActionName

    def head(idx: int) -> jax.Array:
        return jax.nn.sigmoid(logits[..., idx].astype(jnp.float32))

    def weight(name: str) -> jax.Array | float:
        default = getattr(cfg, name)
        if request_weights is None:
            return default
        w = request_weights[:, RANKING_SCORE_WEIGHT_FIELDS.index(name), None].astype(jnp.float32)
        return jnp.where(jnp.isnan(w), default, w)

    favorite = weight("favorite_weight")
    reply = weight("reply_weight")
    retweet = weight("retweet_weight")
    click = weight("click_weight")
    profile_click = weight("profile_click_weight")
    vqv = weight("vqv_weight")
    share = weight("share_weight")
    share_via_dm = weight("share_via_dm_weight")
    share_via_copy_link = weight("share_via_copy_link_weight")
    quote = weight("quote_weight")
    completion = weight("completion_weight")
    cont_dwell_time = weight("cont_dwell_time_weight")
    not_interested = weight("not_interested_weight")
    report = weight("report_weight")
    not_dwelled = weight("not_dwelled_weight")

    vqv_w = jnp.where(
        video_duration_ms.astype(jnp.float32) > weight("min_video_duration_ms"), vqv, 0.0
    )
    vqv_term = vqv_w * head(an.CLIENT_TWEET_VIDEO_QUALITY_VIEW)
    engage = (
        favorite * head(an.SERVER_TWEET_FAV)
        + reply * head(an.SERVER_TWEET_REPLY)
        + retweet * head(an.SERVER_TWEET_RETWEET)
        + click * head(an.CLIENT_TWEET_CLICK)
        + profile_click * head(an.CLIENT_TWEET_CLICK_PROFILE)
        + share * head(an.CLIENT_TWEET_SHARE)
        + share_via_dm * head(an.CLIENT_TWEET_CLICK_SEND_VIA_DIRECT_MESSAGE)
        + share_via_copy_link * head(an.CLIENT_TWEET_SHARE_VIA_COPY_LINK)
        + quote * head(an.SERVER_TWEET_QUOTE)
        + completion * head(VIDEO_COMPLETED_ACTION)
        + not_interested * head(an.CLIENT_TWEET_NOT_INTERESTED_IN)
        + report * head(an.CLIENT_TWEET_REPORT)
    )
    dwell_term = cont_dwell_time * cont_preds[
        ..., recsys_pb2.ContinuousActionName.DWELL_TIME
    ].astype(jnp.float32)
    not_dwelled_p = head(an.CLIENT_TWEET_RECAP_NOT_DWELLED)
    raw_add = engage + vqv_term + dwell_term + not_dwelled * not_dwelled_p

    use_mult = _row_or_default(multiplicative, 0.0) > 0.5
    exp = _row_or_default(power_exponent, 0.0)
    off = jnp.maximum(_row_or_default(power_offset, 0.0), 1e-2)
    skip = jnp.where(
        exp == 0.0,
        1.0,
        jnp.power((not_dwelled_p + off).clip(min=1e-6), exp),
    )
    engage_b = engage + _row_or_default(engage_bias, 0.0)
    time = 1.0 + jnp.maximum(dwell_term, 0.0) + jnp.maximum(vqv_term, 0.0)
    raw_mul = jnp.where(engage_b >= 0.0, skip * engage_b * time, engage_b)
    raw = jnp.where(use_mult, raw_mul, raw_add)

    add_positive = (
        favorite
        + reply
        + retweet
        + click
        + profile_click
        + vqv
        + share
        + share_via_dm
        + share_via_copy_link
        + quote
        + completion
        + jnp.maximum(not_dwelled, 0.0)
    )
    add_negative = -(not_interested + report + jnp.minimum(not_dwelled, 0.0))
    mul_positive = (
        add_positive - jnp.maximum(not_dwelled, 0.0) - vqv + _row_or_default(engage_bias, 0.0)
    )
    mul_negative = -(not_interested + report)
    positive_sum = jnp.where(use_mult, mul_positive, add_positive)
    negative_sum = jnp.where(use_mult, mul_negative, add_negative)
    total_sum = positive_sum + negative_sum
    return jnp.where(
        total_sum == 0.0,
        jnp.maximum(raw, 0.0),
        jnp.where(
            raw < 0.0,
            (raw + negative_sum) / total_sum * cfg.negative_offset,
            raw + cfg.negative_offset,
        ),
    )


def run_page_decode(
    config: PageRlConfig,
    logits: jax.Array,
    cont_preds: jax.Array,
    cand_emb: jax.Array,
    user_emb: jax.Array,
    request: PageDecodeInputs,
    auth: jax.Array,
    sids: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_cand = request.video_duration_ms.shape[1]
    wscore = page_serve_weighted_score(
        config.serve_score,
        logits[:, :n_cand],
        cont_preds[:, :n_cand],
        request.video_duration_ms,
        request.ranking_score_weights,
        power_exponent=request.not_dwelled_power_exponent,
        power_offset=request.not_dwelled_power_offset,
        engage_bias=request.engage_bias,
        multiplicative=request.multiplicative,
    )
    valid = jnp.arange(n_cand)[None, :] < request.n_candidates[:, None]
    pool_k = page_decode_pool_k(config, n_cand)
    top_vals, top_idx = jax.lax.top_k(jnp.where(valid, wscore, -jnp.inf), pool_k)
    pool_mask = jnp.isfinite(top_vals) & (jnp.arange(pool_k)[None, :] < request.pool_n[:, None])
    behavior = jnp.where(pool_mask, top_vals, 0.0)
    pool_emb = jnp.take_along_axis(cand_emb[:, :n_cand], top_idx[:, :, None], axis=1)
    pool_auth = jnp.take_along_axis(auth[:, :n_cand].astype(jnp.int32), top_idx, axis=1)
    pool_sids = jnp.take_along_axis(sids[:, :n_cand], top_idx[:, :, None], axis=1)

    decoder = config.decoder
    serve = config.serve_decode
    if serve.mode not in PAGE_SERVE_DECODE_MODES:
        raise ValueError(f"serve_decode.mode={serve.mode!r} not in {PAGE_SERVE_DECODE_MODES}")
    if serve.mode != "critic" and serve.max_pool > decoder.max_steps:
        raise ValueError(
            f"serve_decode.max_pool={serve.max_pool} exceeds decoder.max_steps="
            f"{decoder.max_steps}: only serve_decode.mode='critic' decodes a wider pool"
        )
    if serve.mode == "critic":
        if config.critic_only is None:
            raise ValueError("serve_decode.mode='critic' needs page_rl.critic_only")
        if serve.eta <= 0.0 or serve.prior_tau <= 0.0:
            raise ValueError(
                f"serve_decode eta={serve.eta} and prior_tau={serve.prior_tau} must be > 0"
            )
        n_users = behavior.shape[0]

        def row_param(override: jax.Array | None, default: float, *, positive: bool) -> jax.Array:
            value = jnp.full((n_users,), default, jnp.float32)
            if override is None:
                return value
            x = override.astype(jnp.float32)
            ok = jnp.isfinite(x) & ((x > 0.0) if positive else (x >= 0.0))
            return jnp.where(ok, x, value)

        decoded = PageCriticTransformer(config.critic_only).decode(
            pool_emb,
            pool_mask,
            behavior,
            user_emb,
            request.steps,
            request.temperature,
            hk.next_rng_key(),
            pool_auth,
            pool_sids,
            max_steps=decoder.decode_steps(pool_k),
            eta=row_param(request.eta, serve.eta, positive=True),
            nu_author=row_param(request.nu_author, serve.nu_author, positive=False),
            nu_sid=row_param(request.nu_sid, serve.nu_sid, positive=False),
            prior_tau=row_param(request.prior_tau, serve.prior_tau, positive=True),
        )
    else:
        slot_values = None
        if decoder.beam_width > 1:
            slot_values = page_critic_value_matrix(
                config=config.critic,
                candidate_embeddings=pool_emb,
                top_k=decoder.decode_steps(pool_k),
            )
        decoded = PageDecoder(decoder).decode(
            pool_emb,
            pool_mask,
            behavior,
            user_emb,
            request.steps,
            request.temperature,
            hk.next_rng_key(),
            pool_auth,
            pool_sids,
            slot_values=slot_values,
        )
    picked = decoded.picked
    safe = jnp.maximum(picked, 0)
    picked_idx = jnp.where(picked >= 0, jnp.take_along_axis(top_idx, safe, axis=1), -1)
    picked_score = jnp.where(picked >= 0, jnp.take_along_axis(top_vals, safe, axis=1), 0.0)
    pad = ((0, 0), (0, pool_k - picked.shape[1]))
    return (
        jnp.pad(picked_idx, pad, constant_values=-1),
        jnp.pad(decoded.picked_prob, pad),
        jnp.pad(picked_score, pad),
    )


def page_decode_pool_k(config: PageRlConfig, n_cand: int) -> int:
    return min(config.serve_decode.max_pool or config.decoder.max_steps, n_cand)


def page_decode_noop_outputs(
    config: PageRlConfig, request: PageDecodeInputs
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_users, n_cand = request.video_duration_ms.shape
    pool_k = page_decode_pool_k(config, n_cand)
    return (
        jnp.full((n_users, pool_k), -1, dtype=jnp.int32),
        jnp.zeros((n_users, pool_k), dtype=jnp.float32),
        jnp.zeros((n_users, pool_k), dtype=jnp.float32),
    )


class UserDebiasBatch(NamedTuple):
    stats: jax.Array
    labels: jax.Array
    rows: jax.Array


_USER_DEBIAS_GLOBAL_INIT = (-4.0,) * len(USER_DEBIAS_ACTIONS) + (2.0,)


class PageUserDebias(hk.Module):
    def __init__(self, config: PageUserDebiasConfig, name: str = "page_user_debias"):
        super().__init__(name=name)
        self._config = config

    @staticmethod
    def _param(name: str, shape: list[int], init: hk.initializers.Initializer) -> jax.Array:
        return get_parameter(
            name,
            shape,
            dtype=jnp.float32,
            init=init,
            pspec=P(*([None] * len(shape))),
            dtype_unwrap=jnp.float32,
        )

    def global_values(self) -> jax.Array:
        init = hk.initializers.Constant(jnp.asarray(_USER_DEBIAS_GLOBAL_INIT))
        return self._param("global", [USER_DEBIAS_OUTPUTS], init)

    def __call__(
        self,
        user_emb: jax.Array,
        stats: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate([user_emb.astype(jnp.float32), stats.astype(jnp.float32)], axis=-1)
        hidden = self._config.hidden_dim
        dims = (x.shape[-1], hidden, hidden, USER_DEBIAS_OUTPUTS)
        for i in range(3):
            last = i == 2
            w_init = (
                hk.initializers.Constant(0.0)
                if last
                else hk.initializers.VarianceScaling(2.0, mode="fan_in")
            )
            b_init = hk.initializers.Constant(
                jnp.asarray(_USER_DEBIAS_GLOBAL_INIT) if last else 0.0
            )
            w = self._param(f"both_w{i}", [dims[i], dims[i + 1]], w_init)
            b = self._param(f"both_b{i}", [dims[i + 1]], b_init)
            x = jnp.einsum("ud,dk->uk", x, w) + b
            if not last:
                x = jax.nn.relu(x)
        return x


def page_user_debias_shift(
    config: PageUserDebiasConfig, user_emb: jax.Array, user_stats: jax.Array
) -> jax.Array:
    heads = PageUserDebias(config)
    shift = heads(user_emb, user_stats) - heads.global_values()
    return jnp.clip(shift, -config.clip_logit, config.clip_logit)


def page_user_debias_serve(
    config: PageUserDebiasConfig,
    logits: jax.Array,
    cont_preds: jax.Array,
    shift: jax.Array,
    heads_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits, cont_preds = jnp.asarray(logits), jnp.asarray(cont_preds)
    shift = shift * jnp.where(jnp.isnan(heads_mask), 1.0, heads_mask)
    if config.serve_heads:
        logits = logits.at[..., jnp.asarray(USER_DEBIAS_ACTIONS)].add(
            -shift[:, None, :-1].astype(logits.dtype)
        )
    if config.serve_dwell:
        cont_preds = cont_preds.at[..., recsys_pb2.ContinuousActionName.DWELL_TIME].multiply(
            jnp.exp(-shift[:, None, -1]).astype(cont_preds.dtype)
        )
    return logits, cont_preds


def page_user_debias_loss(
    user: jax.Array,
    global_values: jax.Array,
    batch: UserDebiasBatch,
    dwell_s: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    rows = batch.rows.astype(jnp.float32)
    n = jnp.maximum(rows.sum(), 1.0)
    labels = batch.labels.astype(jnp.float32)
    log_dwell = jnp.log1p(jnp.clip(dwell_s.astype(jnp.float32), 0.0, USER_DEBIAS_DWELL_CAP_S))

    def fit(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        v = jnp.broadcast_to(values, (rows.shape[0], USER_DEBIAS_OUTPUTS))[:, None, :]
        bce = optax.sigmoid_binary_cross_entropy(v[..., :-1], labels)
        bce = (bce * rows[..., None]).sum((0, 1)) / n
        mae = (jnp.abs(v[..., -1] - log_dwell) * rows).sum() / n
        return bce, mae

    global_bce, global_mae = fit(global_values)
    user_bce, user_mae = fit(user)
    total = global_bce.sum() + global_mae + user_bce.sum() + user_mae
    return total, {
        "page_rl/user_debias/global/dwell_mae": global_mae,
        "page_rl/user_debias/both/dwell_mae": user_mae,
        "page_rl/user_debias/loss": total,
    }
