# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from xai_configlib import Config, configclass
from xai_proto import recsys_pb2
from xrex.models.page_critic import (
    PageCriticConfig,
    discounted_suffix_sum,
    page_critic_loss,
    page_critic_value_matrix,
)
from xrex.models.page_decoder import (
    PageDecoder,
    PageDecoderConfig,
    PagePgConfig,
    decoder_bc_loss,
    page_pg_loss,
)

PAGE_PARAM_REGEX = "page_critic|page_decoder|kl_beta_u"

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
)


@configclass
class PageRlConfig(Config):
    enabled: bool = False
    reward: PageRewardConfig = PageRewardConfig()
    critic: PageCriticConfig = PageCriticConfig()
    decoder: PageDecoderConfig = PageDecoderConfig()
    pg: PagePgConfig = PagePgConfig()
    serve_score: PageServeScoreConfig = PageServeScoreConfig()
    bc_loss_weight: float = 0.25


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
    critic_out = page_critic_loss(
        config=config.critic,
        emb_sorted=emb_sorted,
        slate_sorted=slate_sorted,
        served_slot_sorted=_sorted(rl_served_slot),
        supervised_sorted=supervised_sorted,
        adv_label_sorted=fields.adv_label[:, :s_max],
        reward_sorted=fields.reward[:, :s_max],
        return_sorted=fields.return_g[:, :s_max],
        explore_page=(logged_prob_sorted > 0.0).any(axis=1),
        gamma=config.reward.gamma,
        grid_steps=min(config.pg.q_steps, s_max),
    )

    sids_sorted = _sorted(post_sids)
    auth_sorted = _sorted(auth_hashes)
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
    total = critic_out.total + config.bc_loss_weight * bc_loss + pg_out.total
    return total, {**critic_out.stats, **bc_stats, **pg_out.stats}


class PageDecodeInputs(NamedTuple):
    video_duration_ms: jax.Array
    n_candidates: jax.Array
    pool_n: jax.Array
    steps: jax.Array
    temperature: jax.Array
    ranking_score_weights: jax.Array

    @classmethod
    def no_decode(cls, n_users: int, n_candidates: int) -> PageDecodeInputs:
        return cls(
            video_duration_ms=jnp.zeros((n_users, n_candidates), dtype=jnp.float32),
            n_candidates=jnp.zeros((n_users,), dtype=jnp.int32),
            pool_n=jnp.zeros((n_users,), dtype=jnp.int32),
            steps=jnp.zeros((n_users,), dtype=jnp.int32),
            temperature=jnp.zeros((n_users,), dtype=jnp.float32),
            ranking_score_weights=jnp.full(
                (n_users, len(RANKING_SCORE_WEIGHT_FIELDS)), jnp.nan, dtype=jnp.float32
            ),
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


def page_serve_weighted_score(
    cfg: PageServeScoreConfig,
    logits: jax.Array,
    cont_preds: jax.Array,
    video_duration_ms: jax.Array,
    request_weights: jax.Array | None = None,
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
    cont_dwell_time = weight("cont_dwell_time_weight")
    not_interested = weight("not_interested_weight")
    report = weight("report_weight")
    not_dwelled = weight("not_dwelled_weight")

    vqv_w = jnp.where(
        video_duration_ms.astype(jnp.float32) > weight("min_video_duration_ms"), vqv, 0.0
    )
    raw = (
        favorite * head(an.SERVER_TWEET_FAV)
        + reply * head(an.SERVER_TWEET_REPLY)
        + retweet * head(an.SERVER_TWEET_RETWEET)
        + click * head(an.CLIENT_TWEET_CLICK)
        + profile_click * head(an.CLIENT_TWEET_CLICK_PROFILE)
        + vqv_w * head(an.CLIENT_TWEET_VIDEO_QUALITY_VIEW)
        + share * head(an.CLIENT_TWEET_SHARE)
        + share_via_dm * head(an.CLIENT_TWEET_CLICK_SEND_VIA_DIRECT_MESSAGE)
        + share_via_copy_link * head(an.CLIENT_TWEET_SHARE_VIA_COPY_LINK)
        + quote * head(an.SERVER_TWEET_QUOTE)
        + not_interested * head(an.CLIENT_TWEET_NOT_INTERESTED_IN)
        + report * head(an.CLIENT_TWEET_REPORT)
    )
    raw = (
        raw
        + cont_dwell_time
        * cont_preds[..., recsys_pb2.ContinuousActionName.DWELL_TIME].astype(jnp.float32)
        + not_dwelled * head(an.CLIENT_TWEET_RECAP_NOT_DWELLED)
    )
    positive_sum = (
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
        + jnp.maximum(not_dwelled, 0.0)
    )
    negative_sum = -(not_interested + report + jnp.minimum(not_dwelled, 0.0))
    total_sum = positive_sum + negative_sum
    return jnp.where(
        raw < 0.0,
        (raw + negative_sum) / total_sum * cfg.negative_offset,
        raw + cfg.negative_offset,
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
    )
    valid = jnp.arange(n_cand)[None, :] < request.n_candidates[:, None]
    pool_k = min(config.decoder.max_steps, n_cand)
    top_vals, top_idx = jax.lax.top_k(jnp.where(valid, wscore, -jnp.inf), pool_k)
    pool_mask = jnp.isfinite(top_vals) & (jnp.arange(pool_k)[None, :] < request.pool_n[:, None])
    behavior = jnp.where(pool_mask, top_vals, 0.0)
    pool_emb = jnp.take_along_axis(cand_emb[:, :n_cand], top_idx[:, :, None], axis=1)
    pool_auth = jnp.take_along_axis(auth[:, :n_cand].astype(jnp.int32), top_idx, axis=1)
    pool_sids = jnp.take_along_axis(sids[:, :n_cand], top_idx[:, :, None], axis=1)

    decoder = config.decoder
    slot_values = None
    if decoder.beam_width > 1:
        slot_values = page_critic_value_matrix(
            config=config.critic, candidate_embeddings=pool_emb, top_k=decoder.decode_steps(pool_k)
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


def page_decode_noop_outputs(
    config: PageRlConfig, request: PageDecodeInputs
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_users, n_cand = request.video_duration_ms.shape
    pool_k = min(config.decoder.max_steps, n_cand)
    return (
        jnp.full((n_users, pool_k), -1, dtype=jnp.int32),
        jnp.zeros((n_users, pool_k), dtype=jnp.float32),
        jnp.zeros((n_users, pool_k), dtype=jnp.float32),
    )
