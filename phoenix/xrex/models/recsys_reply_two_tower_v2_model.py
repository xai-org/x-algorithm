# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from xai_configlib import configclass
from xai_proto import recsys_pb2
from xrex.data.recsys.constants import engagement_to_ids
from xrex.data.recsys.feature_config import BoolFeature, UserCategoricalFeature
from xrex.data.recsys.reply_features import (
    REPLY_AUTHOR_FOLLOWER_COUNT_INT64_INDEX,
    REPLY_AUTHOR_VERIFIED_INT64_INDEX,
    REPLY_FAV_BY_ROOT_AUTHOR_INT64_INDEX,
    REPLY_FAV_COUNT_INT64_INDEX,
    REPLY_FOCAL_AUTHOR_FOLLOWS_INT64_INDEX,
    REPLY_FOCAL_AUTHOR_REPLY_INT64_INDEX,
    REPLY_GROX_SCORE_INT64_INDEX,
    REPLY_HAS_MEDIA_INT64_INDEX,
    REPLY_INJECTED_NEGATIVE_INT64_INDEX,
    REPLY_NEGFB_TOTAL_COUNT_INT64_INDEX,
    REPLY_POST_LANGUAGE_CATEGORICAL_INDEX,
    REPLY_QUOTE_COUNT_INT64_INDEX,
    REPLY_REPLY_COUNT_INT64_INDEX,
    REPLY_REPOST_COUNT_INT64_INDEX,
    REPLY_SECTION_INT64_INDEX,
    REPLY_SELF_REPLY_INT64_INDEX,
    REPLY_VIDEO_DURATION_MS_INT64_INDEX,
    REPLY_VIEW_COUNT_INT64_INDEX,
    ReplyUserInt64Feature,
)
from xrex.models.loss_recsys import multihot_loss_compute, tweedie_loss_compute
from xrex.models.model_utils import get_parameter
from xrex.models.normalization import RMSNorm
from xrex.models.page_rl import PageDecodeInputs
from xrex.models.recsys_embedding import (
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import (
    DTYPE_BY_NAME,
    RecsysAggregatedModelConfig,
    RecsysFeaturesBatch,
    get_probs_and_labels,
    metric_loss,
    metric_rce,
)
from xrex.models.recsys_sid import _load_sid_decoder
from xrex.models.sharding_context import ShardingContext

_LOG_SCALE = 4.0
_RATE_PRIOR = 100.0
_COS_BUCKETS = 34
_DWELL_BUCKETS = 24
_SURFACE_VOCAB = 16
_DWELL_INDEX = int(recsys_pb2.ContinuousActionName.DWELL_TIME)
_POSITION_EDGES = (1, 2, 3, 4, 5, 6, 8, 11, 16, 26, 41)
_GROX_SCORE_NUM_BUCKETS = 26
_SID_PREFIX_MAX_LEN = 4
_SID_PREFIX_BUCKETS = 131072

REPLY_NEGFB_HEAD_INDICES: tuple[int, ...] = tuple(engagement_to_ids("reply")["IsNegativeFeedback"])


def _log1p(x: jax.Array) -> jax.Array:
    return jnp.log1p(jnp.maximum(jnp.asarray(x).astype(jnp.float32), 0.0)) / _LOG_SCALE


def _log_rate(num: jax.Array, den: jax.Array) -> jax.Array:
    num = jnp.maximum(jnp.asarray(num).astype(jnp.float32), 0.0)
    den = jnp.maximum(jnp.asarray(den).astype(jnp.float32), 0.0)
    return (jnp.log(num + 1.0) - jnp.log(den + _RATE_PRIOR)) / _LOG_SCALE


@jax.custom_vjp
def _small_table_take(table: jax.Array, ids: jax.Array) -> jax.Array:
    return table[ids]


def _small_table_take_fwd(table: jax.Array, ids: jax.Array):
    return table[ids], (table, ids)


def _small_table_take_bwd(res, g: jax.Array):
    table, ids = res
    one_hot = jax.nn.one_hot(ids.reshape(-1), table.shape[0], dtype=g.dtype)
    grad = jnp.dot(one_hot.T, g.reshape(-1, table.shape[-1]), preferred_element_type=jnp.float32)
    return grad.astype(table.dtype), np.zeros(ids.shape, dtype=jax.dtypes.float0)


_small_table_take.defvjp(_small_table_take_fwd, _small_table_take_bwd)


def log_bucket(x: jax.Array, num_buckets: int, per_octave: int) -> jax.Array:
    v = jnp.maximum(jnp.asarray(x).astype(jnp.float32), 0.0)
    b = jnp.floor(per_octave * jnp.log2(v + 1.0)).astype(jnp.int32)
    return jnp.clip(b, 0, num_buckets - 1)


def known_log_bucket(
    x: jax.Array, known: jax.Array, num_buckets: int, per_octave: int
) -> jax.Array:
    return jnp.where(known, 1 + log_bucket(x, num_buckets - 1, per_octave), 0)


def sid_prefix_bucket_ids(codes: jax.Array, max_len: int, num_buckets: int) -> jax.Array:
    c = jnp.asarray(codes).astype(jnp.uint32)
    present = c[..., 0] > 0
    h = c[..., 0] * jnp.uint32(0x9E3779B1) + jnp.uint32(0x7F4A7C15)
    out = []
    for p in range(1, max_len):
        h = (h ^ c[..., p]) * jnp.uint32(0x01000193)
        h = h ^ (h >> jnp.uint32(15))
        present = present & (c[..., p] > 0)
        out.append(jnp.where(present, (h % jnp.uint32(num_buckets)).astype(jnp.int32), -1))
    return jnp.stack(out, axis=-1)


def decode_entity_sid(sids: jax.Array, decoder_path: str) -> jax.Array:
    stages_np, (w0_np, b0_np, w1_np, b1_np) = _load_sid_decoder(decoder_path)
    num_levels, codebook_size, _ = stages_np.shape
    assert sids.shape[-1] == num_levels, (
        f"post_sids have {sids.shape[-1]} levels but the decoder at {decoder_path!r} "
        f"expects {num_levels}; set sid_num_levels to match the artifact."
    )
    stages = jnp.asarray(stages_np)
    w0, b0 = jnp.asarray(w0_np), jnp.asarray(b0_np)
    w1, b1 = jnp.asarray(w1_np), jnp.asarray(b1_np)

    codes = jnp.clip(sids.astype(jnp.int32) - 1, 0, codebook_size - 1)
    missing = sids[..., 0] == 0
    quant = stages[jnp.arange(num_levels), codes].sum(-2)
    h = jax.nn.relu(quant @ w0 + b0)
    recon = h @ w1 + b1
    recon = recon * jax.lax.rsqrt((recon**2).sum(-1, keepdims=True) + 1e-12)
    recon = jnp.where(missing[..., None], 0.0, recon)
    return jax.lax.stop_gradient(recon)


def _f32(x: jax.Array) -> jax.Array:
    return jnp.asarray(x).astype(jnp.float32)


def _masked_mean(x: jax.Array, w: jax.Array) -> jax.Array:
    w = w.astype(jnp.float32)
    denom = jnp.maximum(w.sum(axis=1, keepdims=True), 1.0)
    return ((x.astype(jnp.float32) * w[..., None]).sum(axis=1) / denom).astype(x.dtype)


def _action_group_ids(num_actions: int) -> tuple[list[int], list[int]]:
    ids = engagement_to_ids("reply")
    pos = [ids[k][0] for k in ("IsFavorited", "IsReplied", "IsRetweeted") if k in ids]
    neg = list(ids.get("IsNegativeFeedback", []))
    return [i for i in pos if i < num_actions], [i for i in neg if i < num_actions]


def grox_score_bucket(slot_value: jax.Array) -> jax.Array:
    bucket = jnp.clip(jnp.floor_divide(slot_value - 1001, 25), -4, 20) + 5
    return jnp.where(slot_value > 0, bucket, 0)


def placement_feature_buckets(int64_features: jax.Array) -> list[tuple[str, int, jax.Array]]:
    grox = int64_features[..., REPLY_GROX_SCORE_INT64_INDEX].astype(jnp.int32)
    out = [("cand_grox_score", _GROX_SCORE_NUM_BUCKETS, grox_score_bucket(grox))]
    for name, slot, vocab in (
        ("cand_reply_section", REPLY_SECTION_INT64_INDEX, 10),
        ("cand_author_verified", REPLY_AUTHOR_VERIFIED_INT64_INDEX, 5),
        ("cand_fav_by_root_author", REPLY_FAV_BY_ROOT_AUTHOR_INT64_INDEX, 3),
        ("cand_focal_author_reply", REPLY_FOCAL_AUTHOR_REPLY_INT64_INDEX, 3),
        ("cand_self_reply", REPLY_SELF_REPLY_INT64_INDEX, 3),
    ):
        value = int64_features[..., slot].astype(jnp.int32)
        out.append((name, vocab, jnp.where((value > 0) & (value < vocab), value, 0)))
    return out


def within_record_auc(
    scores: jax.Array, labels: jax.Array, candidate_padding_mask: jax.Array
) -> jax.Array:
    s = scores.astype(jnp.float32)
    valid = candidate_padding_mask.astype(jnp.bool_)
    pos = (labels > 0.5) & valid
    neg = (labels <= 0.5) & valid
    gt = (s[:, :, None] > s[:, None, :]).astype(jnp.float32)
    tie = (s[:, :, None] == s[:, None, :]).astype(jnp.float32)
    pair = pos[:, :, None] & neg[:, None, :]
    n_pair = pair.sum(axis=(1, 2)).astype(jnp.float32)
    correct = ((gt + 0.5 * tie) * pair).sum(axis=(1, 2))
    has = n_pair > 0.0
    auc = jnp.where(has, correct / jnp.maximum(n_pair, 1.0), 0.0)
    return auc.sum() / jnp.maximum(has.astype(jnp.float32).sum(), 1.0)


def compute_reply_ranker_loss(
    logits: jax.Array,
    actions: jax.Array,
    candidate_padding_mask: jax.Array,
    *,
    metric_group: str,
    loss_mask: jax.Array,
    metric_mask: jax.Array,
    raw_weights: jax.Array | None,
    negfb_loss_weight: float,
    position_logit_bias: jax.Array | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = logits.astype(jnp.float32)
    actions = actions.astype(jnp.float32)
    padding_mask = candidate_padding_mask.astype(jnp.int32)

    head_weight = (
        jnp.ones((logits.shape[-1],), dtype=jnp.float32)
        .at[jnp.array(REPLY_NEGFB_HEAD_INDICES)]
        .set(negfb_loss_weight)
    )
    loss_mask = loss_mask * head_weight

    raw_logits = logits
    if position_logit_bias is not None:
        logits = logits + position_logit_bias

    cross_entropy_loss, _ = multihot_loss_compute(
        logits,
        actions,
        padding_mask,
        loss_mask,
        raw_weights,
    )
    stats: dict[str, jax.Array] = {"ce_loss": cross_entropy_loss}

    valid = metric_mask.astype(jnp.float32)
    prob_labels = get_probs_and_labels(logits, actions, metric_group)
    n_valid = jnp.maximum(valid.sum(), 1.0)
    for eng_name, (p, y) in prob_labels.items():
        stats[f"{eng_name}_RCE"] = metric_rce(p, y, valid)
        stats[f"{eng_name}_loss"] = metric_loss(p, y, valid)
        n_pos = (valid * y).sum()
        ratio_pos = n_pos / n_valid
        mean_pred = (valid * p).sum() / n_valid
        stats[f"{eng_name}_num_pos"] = n_pos
        stats[f"{eng_name}_ratio_pos"] = ratio_pos
        stats[f"{eng_name}_mean_pred"] = mean_pred
        stats[f"{eng_name}_calib"] = mean_pred / jnp.maximum(ratio_pos, 1e-9)
        stats[f"{eng_name}_within_record_auc"] = within_record_auc(p, y, metric_mask)
        stats[f"{eng_name}_n_valid"] = valid.sum()
    if position_logit_bias is not None:
        raw_prob_labels = get_probs_and_labels(raw_logits, actions, metric_group)
        for eng_name, (p, y) in raw_prob_labels.items():
            mean_pred_raw = (valid * p).sum() / n_valid
            stats[f"{eng_name}_mean_pred_raw"] = mean_pred_raw
            stats[f"{eng_name}_calib_raw"] = mean_pred_raw / jnp.maximum(
                (valid * y).sum() / n_valid, 1e-9
            )

    return cross_entropy_loss, stats


@configclass
class RecsysReplyTwoTowerV2Config(RecsysAggregatedModelConfig):
    tower_hidden_dim: int = 1024
    head_hidden_dim: int = 1024
    pair_dim: int = 128
    num_interests: int = 4
    num_attn_heads: int = 4
    num_history_layers: int = 1
    history_encoder_width: int = 2048
    history_encoder_layers: int = 0
    history_encoder_heads: int = 16
    head_tower_rank: int = 32
    bilinear_rank: int = 16
    head_bias_init: float = -3.0
    head_bias_lr_multiplier: float = 10.0

    count_num_buckets: int = 48
    count_buckets_per_octave: int = 2
    language_vocab_size: int = 256
    user_categorical_vocab_size: int = 2048

    negfb_loss_weight: float = 1.0

    def make(self, sharding_context: ShardingContext) -> "RecsysReplyTwoTowerV2Model":
        for lc in self.continuous_action_losses:
            if lc.loss_type != "tweedie" or lc.action_index >= self.num_continuous_actions:
                raise ValueError(f"only in-range tweedie continuous losses are supported: {lc}")
        cfg = self
        if self.sid_decoder_path:
            stages, _ = _load_sid_decoder(self.sid_decoder_path)
            cfg = dataclasses.replace(cfg, sid_codebook_size=int(stages.shape[1]))
        return RecsysReplyTwoTowerV2Model(cfg, sharding_context)


class RecsysReplyTwoTowerV2Model(hk.Module):
    config: RecsysReplyTwoTowerV2Config
    sharding_context: ShardingContext

    def __init__(
        self,
        config: RecsysReplyTwoTowerV2Config,
        sharding_context: ShardingContext,
    ):
        super().__init__(name="reply_two_tower_v2")
        self.config = config
        self.sharding_context = sharding_context
        self._norms: dict[str, RMSNorm] = {}
        self._content_dim = 0
        if config.use_post_sid and config.sid_decoder_path:
            _, (_, _, w1, _) = _load_sid_decoder(config.sid_decoder_path)
            self._content_dim = int(w1.shape[-1])

    @property
    def _dtype(self) -> jnp.dtype:
        return DTYPE_BY_NAME[self.config.fprop_dtype]

    def _weight(self, name: str, shape: list[int], zeros: bool = False) -> jax.Array:
        cfg = self.config
        fan_in = shape[0]
        init = hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(fan_in))
        return get_parameter(
            name,
            shape,
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0) if zeros else init,
            pspec=P(*([None] * len(shape))),
            lr_multiplier=cfg.model_config.scale_config.hidden_lr_multiplier(fan_in),
        )

    def _vector(
        self, name: str, dim: int, init_value: float = 0.0, lr_multiplier: float = 1.0
    ) -> jax.Array:
        return get_parameter(
            name,
            [dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(init_value),
            pspec=P(None),
            lr_multiplier=lr_multiplier,
            dtype_unwrap=jnp.float32,
        )

    def _table(
        self, name: str, vocab: int, dim: int | None = None, zeros: bool = False
    ) -> jax.Array:
        cfg = self.config
        d = dim or cfg.emb_table_width
        return get_parameter(
            name,
            [vocab, d],
            dtype=jnp.float32,
            init=(
                hk.initializers.Constant(0.0)
                if zeros
                else hk.initializers.TruncatedNormal(stddev=cfg.embed_init_scale / math.sqrt(d))
            ),
            pspec=P(None, None),
            lr_multiplier=cfg.model_config.scale_config.emb_lr_multiplier(d),
        )

    def _dense(
        self, name: str, x: jax.Array, out_dim: int, *, bias: bool = False, zeros: bool = False
    ) -> jax.Array:
        w = self._weight(name, [x.shape[-1], out_dim], zeros=zeros)
        y = jnp.dot(x.astype(w.dtype), w).astype(self._dtype)
        if bias:
            y = y + self._vector(f"{name}_b", out_dim).astype(y.dtype)
        return y

    def _mlp(self, name: str, x: jax.Array, hidden: int, out_dim: int) -> jax.Array:
        h = jax.nn.silu(self._dense(f"{name}_in", x, hidden, bias=True))
        return self._dense(f"{name}_out", h, out_dim, bias=True)

    def _norm(self, name: str, x: jax.Array) -> jax.Array:
        if name not in self._norms:
            self._norms[name] = RMSNorm(axis=-1, name=name)
        return self._norms[name](x.astype(self._dtype))

    def _bag(
        self,
        lookups: list[tuple[str, int, jax.Array]],
        dim: int | None = None,
        zeros: bool = False,
    ) -> jax.Array:
        out = None
        for name, vocab, ids in lookups:
            e = _small_table_take(
                self._table(name, vocab, dim, zeros=zeros),
                jnp.clip(jnp.asarray(ids).astype(jnp.int32), 0, vocab - 1),
            )
            out = e if out is None else out + e
        assert out is not None
        return out.astype(self._dtype)

    @staticmethod
    def _user_int64(batch: RecsysFeaturesBatch, slot: int) -> jax.Array:
        return jnp.maximum(jnp.asarray(batch["user_int64_features"])[:, int(slot)], 0)

    @staticmethod
    def _user_categorical(batch: RecsysFeaturesBatch, slot: int) -> jax.Array:
        arr = jnp.asarray(batch["user_categorical_features"])
        return jnp.maximum(arr[:, int(slot)].astype(jnp.int32), 0)

    @staticmethod
    def _injected_mask(batch: RecsysFeaturesBatch) -> jax.Array:
        injected = jnp.asarray(batch["candidate_seq"]["int64_features"])[
            :, :, REPLY_INJECTED_NEGATIVE_INT64_INDEX
        ]
        return jnp.clip(injected, 0, 1).astype(jnp.float32)

    def _sids(self, batch: RecsysFeaturesBatch, side: str) -> jax.Array | None:
        if not self.config.use_post_sid:
            return None
        return jnp.asarray(batch[side]["post_sids"]).astype(jnp.int32)

    def _decode(self, sids: jax.Array | None) -> jax.Array | None:
        if sids is None or self._content_dim == 0:
            return None
        return decode_entity_sid(sids, self.config.sid_decoder_path).astype(jnp.float32)

    def _sid_embedding(self, sids: jax.Array) -> jax.Array:
        cfg = self.config
        L, cb = cfg.sid_num_levels, cfg.sid_codebook_size
        codes = jnp.clip(sids, 0, cb)
        table = self._table("sid_level_table", L * (cb + 1))
        flat = codes + (jnp.arange(L, dtype=jnp.int32) * (cb + 1))
        emb = table[flat] * (codes > 0)[..., None].astype(table.dtype)
        ptable = self._table("sid_prefix_table", _SID_PREFIX_BUCKETS)
        pid = sid_prefix_bucket_ids(sids, _SID_PREFIX_MAX_LEN, _SID_PREFIX_BUCKETS)
        pemb = ptable[jnp.maximum(pid, 0)] * (pid >= 0)[..., None].astype(ptable.dtype)
        return (emb.sum(axis=-2) + pemb.sum(axis=-2)).astype(self._dtype)

    def _item_encoder(
        self,
        post_emb: jax.Array,
        author_emb: jax.Array,
        sids: jax.Array | None,
        content: jax.Array | None,
    ) -> jax.Array:
        cfg = self.config
        D = cfg.emb_table_width
        blocks = [
            self._norm("item_post_norm", self._dense("item_post_proj", post_emb, D)),
            self._norm("item_author_norm", self._dense("item_author_proj", author_emb, D)),
        ]
        if sids is not None:
            blocks.append(self._norm("item_sid_norm", self._sid_embedding(sids)))
        if content is not None:
            blocks.append(
                self._norm("item_content_norm", self._dense("item_content_proj", content, D))
            )
        return self._dense("item_mix", jnp.concatenate(blocks, axis=-1), D, bias=True)

    def _history_context(
        self,
        batch: RecsysFeaturesBatch,
        emb: RecsysEmbeddings,
        hist_sids: jax.Array | None,
        hist_content: jax.Array | None,
    ) -> dict[str, jax.Array]:
        cfg = self.config
        post_hashes = jnp.asarray(batch["history_seq"]["post_hashes"])
        auth_hashes = jnp.asarray(batch["history_seq"]["auth_hashes"])
        B, S = post_hashes.shape[0], post_hashes.shape[1]
        items = self._item_encoder(
            emb.history_post_embeddings.reshape((B, S, -1)),
            emb.history_author_embeddings.reshape((B, S, -1)),
            hist_sids,
            hist_content,
        )
        actions = jnp.asarray(batch["history_seq"]["actions"])[:, 1:].astype(jnp.float32)
        valid = post_hashes[:, 1:, 0] != 0
        pos_ids, neg_ids = _action_group_ids(cfg.model_config.output_vocab_size)
        pos = (actions[..., jnp.array(pos_ids, jnp.int32)].sum(-1) > 0) & valid
        neg = (actions[..., jnp.array(neg_ids, jnp.int32)].sum(-1) > 0) & valid
        return {
            "items": items,
            "valid": valid,
            "actions": actions * valid[..., None],
            "pos": pos,
            "neg": neg,
            "post_hashes": post_hashes[:, 1:],
            "auth_hashes": auth_hashes[:, 1:],
            "focal_post_hashes": post_hashes[:, 0],
            "focal_auth_hashes": auth_hashes[:, 0],
        }

    @staticmethod
    def _match_counts(
        query: jax.Array, hist: dict[str, jax.Array], key: str
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = hist[key]
        extra = query.ndim - 2
        hq = h.reshape(h.shape[0], *([1] * extra), *h.shape[1:])
        eq = jnp.all(query[..., None, :] == hq, axis=-1) & (query[..., :1] != 0)
        valid, pos, neg = (
            hist[k].reshape(h.shape[0], *([1] * extra), h.shape[1]) for k in ("valid", "pos", "neg")
        )
        eqf = eq.astype(jnp.float32)
        return (
            (eqf * valid).sum(-1),
            (eqf * pos).sum(-1),
            (eqf * neg).sum(-1),
        )

    def _history_layer(
        self,
        name: str,
        x: jax.Array,
        valid: jax.Array,
        heads: int,
        mlp_mult: int = 2,
        zero_out: bool = False,
    ) -> jax.Array:
        B, S, D = x.shape
        dh = D // heads
        h = self._norm(f"{name}_attn_norm", x)
        qkv = self._dense(f"{name}_qkv", h, 3 * D).reshape(B, S, 3, heads, dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) / math.sqrt(dh)
        logits = jnp.where(valid[:, None, None, :], logits, -1e9)
        w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        a = jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(B, S, D)
        x = x + self._dense(f"{name}_attn_out", a, D, zeros=zero_out)
        h = self._norm(f"{name}_mlp_norm", x)
        h = jax.nn.silu(self._dense(f"{name}_mlp_in", h, mlp_mult * D, bias=True))
        return x + self._dense(f"{name}_mlp_out", h, D, bias=True, zeros=zero_out)

    def _history_encoder(self, x: jax.Array, valid: jax.Array) -> jax.Array:
        cfg = self.config
        h = self._dense("histx_in", x, cfg.history_encoder_width)
        for i in range(cfg.history_encoder_layers):
            h = self._history_layer(
                f"histx{i}", h, valid, cfg.history_encoder_heads, mlp_mult=4, zero_out=True
            )
        return self._dense("histx_out", self._norm("histx_norm", h), x.shape[-1], zeros=True)

    def _interest_pool(
        self,
        tokens: jax.Array,
        hist: dict[str, jax.Array],
        recency_bucket: jax.Array,
        ctx: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        B, S1, D = tokens.shape
        M, H = cfg.num_interests, cfg.num_attn_heads
        dh = D // H
        seeds = self._table("interest_seeds", M)
        q = seeds[None].astype(self._dtype) + self._dense("interest_ctx", ctx, M * D).reshape(
            B, M, D
        )
        q = self._norm("interest_q_norm", q)
        null = jnp.broadcast_to(
            self._table("interest_null", 1)[None].astype(self._dtype), (B, 1, D)
        )
        kv = self._norm("interest_kv_norm", jnp.concatenate([null, tokens], axis=1))
        qh = self._dense("interest_q", q, D).reshape(B, M, H, dh)
        kh = self._dense("interest_k", kv, D).reshape(B, S1 + 1, H, dh)
        vh = self._dense("interest_v", kv, D).reshape(B, S1 + 1, H, dh)
        logits = jnp.einsum("bmhd,bshd->bhms", qh, kh).astype(jnp.float32) / math.sqrt(dh)
        act_w = self._weight(
            "interest_action_bias", [cfg.model_config.output_vocab_size, M], zeros=True
        )
        rec_w = self._weight("interest_recency_bias", [cfg.count_num_buckets, M], zeros=True)
        bias = (
            jnp.dot(hist["actions"], act_w.astype(jnp.float32))
            + rec_w.astype(jnp.float32)[recency_bucket]
        )
        bias = jnp.concatenate([jnp.zeros((B, 1, M), jnp.float32), bias], axis=1)
        logits = logits + jnp.transpose(bias, (0, 2, 1))[:, None]
        mask = jnp.concatenate([jnp.ones((B, 1), jnp.bool_), hist["valid"]], axis=1)
        logits = jnp.where(mask[:, None, None, :], logits, -1e9)
        w = jax.nn.softmax(logits, axis=-1).astype(vh.dtype)
        out = jnp.einsum("bhms,bshd->bmhd", w, vh).reshape(B, M, D)
        return self._norm("interest_out_norm", q + self._dense("interest_out", out, D))

    def build_user_tower(
        self,
        batch: RecsysFeaturesBatch,
        emb: RecsysEmbeddings,
        hist: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        D, nb, po = cfg.emb_table_width, cfg.count_num_buckets, cfg.count_buckets_per_octave
        B = hist["items"].shape[0]
        dt = self._dtype

        user_vec = self._dense("user_id_proj", emb.user_embeddings.reshape((B, -1)), D)
        vocab = cfg.user_categorical_vocab_size
        user_cat = self._bag(
            [
                (f"user_{feat.name}_table", vocab, self._user_categorical(batch, feat))
                for feat in UserCategoricalFeature
            ]
        )

        focal_fav = self._user_int64(batch, ReplyUserInt64Feature.focalFavCount)
        focal_reply = self._user_int64(batch, ReplyUserInt64Feature.focalReplyCount)
        focal_view = self._user_int64(batch, ReplyUserInt64Feature.focalViewCount)
        focal_followers = self._user_int64(batch, ReplyUserInt64Feature.focalAuthorFollowerCount)
        focal_ts = self._user_int64(batch, ReplyUserInt64Feature.focalPostCreationTsSec)
        viewer_is_author = jnp.clip(
            self._user_int64(batch, ReplyUserInt64Feature.viewerIsFocalAuthor), 0, 2
        )
        in_history = jnp.clip(
            self._user_int64(batch, ReplyUserInt64Feature.focalInViewerHistory), 0, 2
        )
        page_ts = jnp.max(jnp.asarray(batch["candidate_seq"]["impr_ts"]), axis=1)
        focal_age_known = (focal_ts > 0) & (page_ts > 0)
        focal_age_min = jnp.maximum(page_ts - focal_ts, 0).astype(jnp.float32) / 60.0

        valid_f = hist["valid"].astype(jnp.float32)
        n_hist = valid_f.sum(-1)
        denom = jnp.maximum(n_hist, 1.0)
        fa_any, fa_pos, fa_neg = self._match_counts(hist["focal_auth_hashes"], hist, "auth_hashes")
        focal_numeric = jnp.stack(
            [
                _log1p(focal_fav),
                _log1p(focal_reply),
                _log1p(focal_view),
                _log1p(focal_followers),
                _f32(focal_fav > 0),
                _f32(focal_view > 0),
                _f32(focal_followers > 0),
                jnp.where(focal_age_known, _log1p(focal_age_min), 0.0),
                _f32(focal_age_known),
                _f32(viewer_is_author == 2),
                _f32(viewer_is_author > 0),
                _f32(in_history == 2),
                _f32(in_history > 0),
                _log1p(n_hist),
                hist["pos"].astype(jnp.float32).sum(-1) / denom,
                hist["neg"].astype(jnp.float32).sum(-1) / denom,
                _log1p(fa_any),
                _log1p(fa_pos),
                _log1p(fa_neg),
            ],
            axis=-1,
        )
        focal_lang = jnp.clip(
            self._user_int64(batch, ReplyUserInt64Feature.focalPostLanguageCode),
            0,
            cfg.language_vocab_size - 1,
        )
        focal_bag = self._bag(
            [
                ("focal_fav_bucket", nb, known_log_bucket(focal_fav, focal_fav > 0, nb, po)),
                ("focal_reply_bucket", nb, known_log_bucket(focal_reply, focal_reply > 0, nb, po)),
                ("focal_view_bucket", nb, known_log_bucket(focal_view, focal_view > 0, nb, po)),
                (
                    "focal_followers_bucket",
                    nb,
                    known_log_bucket(focal_followers, focal_followers > 0, nb, po),
                ),
                ("focal_age_bucket", nb, known_log_bucket(focal_age_min, focal_age_known, nb, po)),
                ("focal_language", cfg.language_vocab_size, focal_lang),
                ("focal_viewer_is_author", 3, viewer_is_author),
                ("focal_in_history", 3, in_history),
                ("user_history_size_bucket", nb, log_bucket(n_hist, nb, po)),
            ]
        )
        focal_item = hist["items"][:, 0]
        focal_in = jnp.concatenate(
            [
                focal_item,
                self._norm(
                    "focal_numeric_norm", self._dense("focal_numeric_proj", focal_numeric, D)
                ),
                self._norm("focal_bag_norm", focal_bag),
            ],
            axis=-1,
        )
        f = self._norm(
            "focal_out_norm", focal_item + self._mlp("focal_mlp", focal_in, cfg.tower_hidden_dim, D)
        )

        hs = batch["history_seq"]
        hist_ts = jnp.asarray(hs["impr_ts"])[:, 1:]
        rec_known = (hist_ts > 0) & (page_ts[:, None] > 0)
        rec_min = jnp.maximum(page_ts[:, None] - hist_ts, 0).astype(jnp.float32) / 60.0
        recency = known_log_bucket(rec_min, rec_known, nb, po)
        dwell = jnp.asarray(hs["continuous_actions"])[:, 1:, _DWELL_INDEX]
        surface = jnp.asarray(hs["product_surface"])[:, 1:]
        act_table = self._table("hist_action_table", cfg.model_config.output_vocab_size)
        tokens = (
            hist["items"][:, 1:]
            + jnp.dot(hist["actions"], act_table.astype(jnp.float32)).astype(dt)
            + self._bag(
                [
                    ("hist_recency_bucket", nb, recency),
                    (
                        "hist_dwell_bucket",
                        _DWELL_BUCKETS,
                        known_log_bucket(dwell, dwell > 0, _DWELL_BUCKETS, po),
                    ),
                    ("hist_surface", _SURFACE_VOCAB, surface),
                ]
            )
        )
        tokens = self._norm("hist_token_norm", tokens)
        for i in range(cfg.num_history_layers):
            tokens = self._history_layer(f"hist{i}", tokens, hist["valid"], cfg.num_attn_heads)
        if cfg.history_encoder_layers:
            tokens = tokens + self._history_encoder(tokens, hist["valid"])

        ctx = self._norm("interest_ctx_norm", f + self._norm("user_id_norm", user_vec))
        interests = self._interest_pool(tokens, hist, recency, ctx)
        pos_mean = self._norm("hist_pos_norm", _masked_mean(tokens, hist["pos"]))
        neg_mean = self._norm("hist_neg_norm", _masked_mean(tokens, hist["neg"]))
        u_in = jnp.concatenate(
            [
                self._norm("user_id_norm", user_vec),
                self._norm("user_cat_norm", user_cat),
                f,
                interests.reshape(B, -1),
                pos_mean,
                neg_mean,
            ],
            axis=-1,
        )
        u = self._norm("user_out_norm", self._mlp("user_mlp", u_in, cfg.tower_hidden_dim, D))
        return u, jnp.concatenate([f[:, None], interests], axis=1)

    def build_candidate_tower(
        self,
        batch: RecsysFeaturesBatch,
        emb: RecsysEmbeddings,
        sids: jax.Array | None,
        content: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        D, nb, po = cfg.emb_table_width, cfg.count_num_buckets, cfg.count_buckets_per_octave
        cs = batch["candidate_seq"]
        post_hashes = jnp.asarray(cs["post_hashes"])
        B, C = post_hashes.shape[0], post_hashes.shape[1]
        item = self._item_encoder(
            emb.candidate_post_embeddings.reshape((B, C, -1)),
            emb.candidate_author_embeddings.reshape((B, C, -1)),
            sids,
            content,
        )

        i64 = jnp.asarray(cs["int64_features"])
        fav = i64[..., REPLY_FAV_COUNT_INT64_INDEX]
        reply = i64[..., REPLY_REPLY_COUNT_INT64_INDEX]
        repost = i64[..., REPLY_REPOST_COUNT_INT64_INDEX]
        quote = i64[..., REPLY_QUOTE_COUNT_INT64_INDEX]
        view = i64[..., REPLY_VIEW_COUNT_INT64_INDEX]
        followers = i64[..., REPLY_AUTHOR_FOLLOWER_COUNT_INT64_INDEX]
        negfb_raw = i64[..., REPLY_NEGFB_TOTAL_COUNT_INT64_INDEX]
        negfb_known = negfb_raw > 0
        negfb = jnp.maximum(negfb_raw - 1, 0)
        video_s = i64[..., REPLY_VIDEO_DURATION_MS_INT64_INDEX].astype(jnp.float32) / 1000.0
        has_media = jnp.clip(i64[..., REPLY_HAS_MEDIA_INT64_INDEX], 0, 1)
        impr_ts = jnp.asarray(cs["impr_ts"])
        created = jnp.asarray(cs["post_creation_ts_sec"])
        age_known = (impr_ts > 0) & (created > 0)
        age_min = jnp.maximum(impr_ts - created, 0).astype(jnp.float32) / 60.0
        age_h = age_min / 60.0
        numeric = jnp.stack(
            [
                _log1p(fav),
                _log1p(reply),
                _log1p(repost),
                _log1p(quote),
                _log1p(view),
                _log1p(followers),
                jnp.where(negfb_known, _log1p(negfb), 0.0),
                _f32(negfb_known),
                _f32(view > 0),
                _log1p(video_s),
                _f32(has_media),
                jnp.where(age_known, _log1p(age_min), 0.0),
                _f32(age_known),
                _log_rate(fav, view),
                _log_rate(reply, view),
                _log_rate(repost, view),
                jnp.where(negfb_known, _log_rate(negfb, view), 0.0),
                jnp.where(age_known, _log1p(_f32(fav) / (age_h + 2.0)), 0.0),
                jnp.where(age_known, _log1p(_f32(reply) / (age_h + 2.0)), 0.0),
                _log_rate(fav, followers),
            ],
            axis=-1,
        )
        lang = jnp.clip(
            jnp.asarray(cs["categorical_features"])[..., REPLY_POST_LANGUAGE_CATEGORICAL_INDEX],
            0,
            cfg.language_vocab_size - 1,
        )
        bag = self._bag(
            [
                ("cand_fav_bucket", nb, log_bucket(fav, nb, po)),
                ("cand_reply_bucket", nb, log_bucket(reply, nb, po)),
                ("cand_repost_bucket", nb, log_bucket(repost, nb, po)),
                ("cand_view_bucket", nb, known_log_bucket(view, view > 0, nb, po)),
                ("cand_followers_bucket", nb, known_log_bucket(followers, followers > 0, nb, po)),
                ("cand_negfb_bucket", nb, known_log_bucket(negfb, negfb_known, nb, po)),
                ("cand_video_bucket", nb, known_log_bucket(video_s, video_s > 0, nb, po)),
                ("cand_age_bucket", nb, known_log_bucket(age_min, age_known, nb, po)),
                ("cand_language", cfg.language_vocab_size, lang),
                ("cand_has_media", 2, has_media),
            ]
        )
        for name, vocab, ids in placement_feature_buckets(i64):
            e = _small_table_take(self._table(name, vocab, None, zeros=True), ids)
            bag = bag + jnp.where((ids > 0)[..., None], e, 0).astype(self._dtype)
        x = jnp.concatenate(
            [
                item,
                self._norm("cand_numeric_norm", self._dense("cand_numeric_proj", numeric, D)),
                self._norm("cand_bag_norm", bag),
            ],
            axis=-1,
        )
        c = self._norm("cand_out_norm", item + self._mlp("cand_mlp", x, cfg.tower_hidden_dim, D))
        return c, post_hashes[:, :, 0] != 0

    def _pair_features(
        self,
        batch: RecsysFeaturesBatch,
        hist: dict[str, jax.Array],
        cand_content: jax.Array | None,
        focal_content: jax.Array | None,
    ) -> jax.Array:
        cfg = self.config
        Dp, nb, po = cfg.pair_dim, cfg.count_num_buckets, cfg.count_buckets_per_octave
        cs = batch["candidate_seq"]
        cand_auth = jnp.asarray(cs["auth_hashes"])
        cand_post = jnp.asarray(cs["post_hashes"])
        B, C = cand_post.shape[0], cand_post.shape[1]

        followed = jnp.asarray(cs["bool_features"])[
            ..., BoolFeature.isAuthorFollowedByViewerSeq
        ].astype(jnp.int32)
        focal_auth = hist["focal_auth_hashes"]
        is_parent = jnp.all(cand_auth == focal_auth[:, None, :], axis=-1) & (
            focal_auth[:, None, :1] != 0
        ).squeeze(-1)
        i64 = jnp.asarray(cs["int64_features"])
        focal_follows = jnp.clip(i64[..., REPLY_FOCAL_AUTHOR_FOLLOWS_INT64_INDEX], 0, 2)
        focal_ts = self._user_int64(batch, ReplyUserInt64Feature.focalPostCreationTsSec)
        created = jnp.asarray(cs["post_creation_ts_sec"])
        lag_known = (focal_ts[:, None] > 0) & (created > 0)
        lag_min = jnp.maximum(created - focal_ts[:, None], 0).astype(jnp.float32) / 60.0

        if cand_content is not None and focal_content is not None:
            cos = jnp.einsum("bcr,br->bc", cand_content, focal_content)
            cos_known = (jnp.sum(cand_content**2, -1) > 0.5) & (
                jnp.sum(focal_content**2, -1)[:, None] > 0.5
            )
        else:
            cos = jnp.zeros((B, C), jnp.float32)
            cos_known = jnp.zeros((B, C), jnp.bool_)
        cos_bucket = jnp.where(
            cos_known,
            1
            + jnp.clip(
                jnp.floor((cos + 1.0) / 2.0 * (_COS_BUCKETS - 1)).astype(jnp.int32),
                0,
                _COS_BUCKETS - 2,
            ),
            0,
        )

        cand_lang = jnp.asarray(cs["categorical_features"])[
            ..., REPLY_POST_LANGUAGE_CATEGORICAL_INDEX
        ].astype(jnp.int32)
        viewer_lang = self._user_categorical(batch, UserCategoricalFeature.userLanguageCode)
        focal_lang = self._user_int64(batch, ReplyUserInt64Feature.focalPostLanguageCode)

        def _match(a: jax.Array) -> jax.Array:
            known = (a[:, None] > 0) & (cand_lang > 0)
            return jnp.where(known, jnp.where(a[:, None] == cand_lang, 1, 2), 0)

        lang_viewer, lang_focal = _match(viewer_lang), _match(focal_lang)

        def _relative(cand_count: jax.Array, focal_slot: int) -> jax.Array:
            focal = self._user_int64(batch, focal_slot)[:, None]
            return jnp.where(focal > 0, _log1p(cand_count) - _log1p(focal), 0.0)

        a_any, a_pos, a_neg = self._match_counts(cand_auth, hist, "auth_hashes")
        p_any, p_pos, p_neg = self._match_counts(cand_post, hist, "post_hashes")
        cont = jnp.stack(
            [
                _f32(followed),
                _f32(is_parent),
                _f32(focal_follows == 2),
                _f32(focal_follows > 0),
                jnp.where(lag_known, _log1p(lag_min), 0.0),
                _f32(lag_known),
                jnp.where(cos_known, cos, 0.0),
                _f32(cos_known),
                _f32(lang_viewer == 1) - _f32(lang_viewer == 2),
                _f32(lang_focal == 1) - _f32(lang_focal == 2),
                _relative(
                    i64[..., REPLY_FAV_COUNT_INT64_INDEX], ReplyUserInt64Feature.focalFavCount
                ),
                _relative(
                    i64[..., REPLY_VIEW_COUNT_INT64_INDEX], ReplyUserInt64Feature.focalViewCount
                ),
                _relative(
                    i64[..., REPLY_REPLY_COUNT_INT64_INDEX], ReplyUserInt64Feature.focalReplyCount
                ),
                _log1p(a_any),
                _log1p(a_pos),
                _log1p(a_neg),
                _log1p(p_any),
                _log1p(p_pos),
                _log1p(p_neg),
            ],
            axis=-1,
        )
        bag = self._bag(
            [
                ("pair_followed", 2, followed),
                ("pair_is_parent_author", 2, is_parent),
                ("pair_focal_follows", 3, focal_follows),
                ("pair_lag_bucket", nb, known_log_bucket(lag_min, lag_known, nb, po)),
                ("pair_cos_bucket", _COS_BUCKETS, cos_bucket),
                ("pair_lang_viewer", 3, lang_viewer),
                ("pair_lang_focal", 3, lang_focal),
                ("pair_author_any_bucket", nb, log_bucket(a_any, nb, po)),
                ("pair_author_pos_bucket", nb, log_bucket(a_pos, nb, po)),
                ("pair_post_any_bucket", nb, log_bucket(p_any, nb, po)),
            ],
            dim=Dp,
        )
        x = jnp.concatenate(
            [
                self._norm("pair_numeric_norm", self._dense("pair_numeric_proj", cont, Dp)),
                self._norm("pair_bag_norm", bag),
            ],
            axis=-1,
        )
        return jax.nn.silu(self._dense("pair_mix", x, Dp, bias=True))

    def _live_ids(self) -> list[int]:
        V = self.config.model_config.output_vocab_size
        ids = {i for v in engagement_to_ids(self.config.metric_group).values() for i in v}
        return sorted(i for i in ids if 0 <= i < V)

    def _head(
        self, c: jax.Array, u: jax.Array, interests: jax.Array, pair: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        D, V = cfg.emb_table_width, cfg.model_config.output_vocab_size
        H1, H2 = cfg.head_hidden_dim, cfg.head_hidden_dim // 2
        dt = self._dtype

        sim = jnp.einsum(
            "bcd,bmd->bcm", c.astype(jnp.float32), interests.astype(jnp.float32)
        ) / math.sqrt(D)
        attn = jax.nn.softmax(sim, axis=-1)
        u_att = jnp.einsum("bcm,bmd->bcd", attn, interests.astype(jnp.float32)).astype(dt)
        ub = u[:, None, :].astype(dt)
        x = jnp.concatenate([c, c * ub, u_att, c * u_att, sim.astype(dt), pair.astype(dt)], axis=-1)
        h = self._dense("head_in", x, H1) + self._dense("head_in_user", u, H1, bias=True)[:, None]
        h = jax.nn.silu(self._norm("head_in_norm", h))
        h2 = jax.nn.silu(self._dense("head_hidden", h, H2, bias=True))

        head_bias = self._vector(
            "head_bias", V, cfg.head_bias_init, lr_multiplier=cfg.head_bias_lr_multiplier
        )
        logits = (
            self._dense("head_out", h2, V).astype(jnp.float32)
            + head_bias
            + self._dense("head_user_bias", u, V, zeros=True).astype(jnp.float32)[:, None]
            + self._dense("head_cand_bias", c, V, zeros=True).astype(jnp.float32)
        )
        live = self._live_ids()
        if live:
            nL, r, r2 = len(live), cfg.head_tower_rank, cfg.bilinear_rank
            w_t = self._weight("head_tower_w", [H2, nL, r])
            t = jnp.einsum("bch,hlr->bclr", h2, w_t.astype(dt))
            t = jax.nn.silu(t + self._vector("head_tower_b", nL * r).reshape(nL, r).astype(dt))
            v_t = self._weight("head_tower_v", [r, nL], zeros=True)
            tower = jnp.einsum("bclr,rl->bcl", t, v_t.astype(dt)).astype(jnp.float32)
            pc = self._dense("head_bilinear_c", c, nL * r2).reshape(*c.shape[:2], nL, r2)
            pu = self._dense("head_bilinear_u", u, nL * r2, zeros=True).reshape(-1, nL, r2)
            bil = jnp.einsum("bclr,blr->bcl", pc, pu).astype(jnp.float32) / math.sqrt(r2)
            logits = logits.at[..., jnp.array(live, jnp.int32)].add(tower + bil)

        preds = self._dense("continuous_out", h2, cfg.num_continuous_actions, bias=True)
        for lc in cfg.continuous_action_losses:
            col = jax.nn.softplus(preds[:, :, lc.action_index].astype(jnp.float32))
            preds = preds.at[:, :, lc.action_index].set(col.astype(preds.dtype))
        return logits, preds

    @hk.transparent
    def __call__(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cand_sids = self._sids(batch, "candidate_seq")
        hist_sids = self._sids(batch, "history_seq")
        cand_content = self._decode(cand_sids)
        hist_content = self._decode(hist_sids)

        hist = self._history_context(batch, recsys_embeddings, hist_sids, hist_content)
        u, interests = self.build_user_tower(batch, recsys_embeddings, hist)
        c, candidate_mask = self.build_candidate_tower(
            batch, recsys_embeddings, cand_sids, cand_content
        )
        focal_content = hist_content[:, 0] if hist_content is not None else None
        pair = self._pair_features(batch, hist, cand_content, focal_content)
        logits, continuous = self._head(c, u, interests, pair)
        return logits, continuous, candidate_mask

    @hk.transparent
    def forward(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        page_decode: PageDecodeInputs | None = None,
        run_decoder: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        del run_decoder
        if page_decode is not None:
            raise ValueError("the reply ranker has no page decoder; turn serve_page_decoder off")
        logits, continuous, _ = self(batch, recsys_embeddings)
        return logits, continuous

    def _continuous_action_loss(
        self,
        preds: jax.Array,
        batch: RecsysFeaturesBatch,
        candidate_mask: jax.Array,
        raw_weights: jax.Array | None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cfg = self.config
        targets = jnp.asarray(batch["candidate_seq"]["continuous_actions"]).astype(jnp.float32)
        valid = candidate_mask.astype(jnp.bool_)
        total = jnp.zeros((), jnp.float32)
        stats: dict[str, jax.Array] = {}
        for lc in cfg.continuous_action_losses:
            idx = lc.action_index
            action_loss = tweedie_loss_compute(
                targets[:, :, idx],
                preds[:, :, idx].astype(jnp.float32),
                valid,
                jnp.zeros_like(valid),
                p=lc.tweedie_power,
                norm_scale=lc.norm_config.norm_scale,
                mask_negatives=lc.mask_negatives,
                raw_weights=raw_weights,
            )[0]
            total = total + lc.loss_weight * action_loss
            stats[f"{lc.metric_name}_loss"] = action_loss
        return total, stats

    def _position_logit_bias(
        self, batch: RecsysFeaturesBatch, candidate_mask: jax.Array, num_actions: int
    ) -> jax.Array | None:
        cfg = self.config
        table = get_parameter(
            "position_debias_bias",
            [len(_POSITION_EDGES) + 1, num_actions],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=cfg.model_config.scale_config.emb_lr_multiplier(num_actions),
            dtype_unwrap=jnp.float32,
        )
        table = table.at[0].set(0.0)
        positions = batch.get("candidate_positions")
        if positions is None:
            return None
        bucket = jnp.searchsorted(
            jnp.array(_POSITION_EDGES),
            jnp.asarray(positions).astype(jnp.int32),
            side="right",
        )
        bias = table[bucket]
        known = ((candidate_mask > 0) & (bucket > 0)).astype(jnp.float32)[..., None]
        mean = (bias * known).sum(axis=(0, 1), keepdims=True) / jnp.maximum(
            known.sum(axis=(0, 1), keepdims=True), 1.0
        )
        return jnp.where(known > 0, bias - mean, 0.0)

    @hk.transparent
    def loss(
        self,
        inter,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = True,
        **_metrics_state_kwargs,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del inter, is_training, _metrics_state_kwargs
        cfg = self.config
        logits, continuous, candidate_mask = self(
            batch, get_recsys_embed_param_to_jax_array(recsys_embeddings)
        )
        V = logits.shape[-1]
        actions = jnp.asarray(batch["candidate_seq"]["actions"])
        cmask = candidate_mask.astype(jnp.float32)
        injected = self._injected_mask(batch)
        raw_weights = None
        sample_weights = batch.get("sample_weights")
        if sample_weights is not None:
            raw_weights = jnp.broadcast_to(
                jnp.asarray(sample_weights).astype(jnp.float32).reshape(-1, 1), cmask.shape
            )
        loss, stats = compute_reply_ranker_loss(
            logits,
            actions,
            candidate_mask,
            metric_group=cfg.metric_group,
            loss_mask=jnp.broadcast_to(1.0 - injected[..., None], (*injected.shape, V)),
            metric_mask=candidate_mask.astype(jnp.bool_) & (injected * cmask == 0.0),
            raw_weights=raw_weights,
            negfb_loss_weight=cfg.negfb_loss_weight,
            position_logit_bias=self._position_logit_bias(batch, candidate_mask, V),
        )
        cont_loss, cont_stats = self._continuous_action_loss(
            continuous, batch, candidate_mask, raw_weights
        )
        stats.update(cont_stats)
        return loss + cont_loss, stats
