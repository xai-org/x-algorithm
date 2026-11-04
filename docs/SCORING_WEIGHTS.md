# Scoring weights, in one place

This document collects what the For You ranking score is made of: every weighted head, its
default value in both places it is declared, the transform applied to the weighted sum, and the
multipliers applied afterwards.

It describes commit `b220448` (2026-11-03) and is descriptive, not normative. When the code and
this document disagree, the code is right.

Related: [`BIDIRECTIONAL_BOOST_CHANGE.md`](BIDIRECTIONAL_BOOST_CHANGE.md) walks through one real
weight change over time.

## 1. The formula

```
Final Score = offset( Σ_i  w_i · f_i )
```

`w_i` are the weights below. `f_i` is one of three things, depending on the head:

| head kind | `f_i` | heads |
| --- | --- | --- |
| probability | `P(action)` | favorite, reply, retweet, quote, share, share_via_dm, share_via_copy_link, photo_expand, video_open, click, open_link, profile_click, vqv, quoted_click, quoted_vqv, dwell, follow_author, post_unexplored, post_video, not_interested, block_author, mute_author, report, not_dwelled, click_dwell_time |
| predicted continuous value | seconds | dwell_time |
| product of the two | `P(open) × E[seconds \| open]` | video_continuation, user_video_continuation, profile_visit_secs |

- Term list and arithmetic: [`xai-value-model/scoring.rs`](../xai-value-model/scoring.rs) —
  `weighted_terms()` returns 29 terms, `compute_weighted_score()` sums them.
- Entry point on the request path: `VMRanker::local_scores()` in
  [`home-mixer/scorers/vm_ranker.rs`](../home-mixer/scorers/vm_ranker.rs), which falls back to
  `value_model::weighted_score()` in [`home-mixer/scorers/value_model.rs`](../home-mixer/scorers/value_model.rs)
  when a candidate has no cached `weighted_score`.

Weights scale predicted probabilities, never raw engagement counts. See the note in
[`home-mixer/params/param.rs`](../home-mixer/params/param.rs) above `FavoriteWeight`, and the
"468 likes" example in the [README](../README.md#scoring-and-ranking).

## 2. Weights

`home-mixer/params/param.rs` carries a sync header and is refreshed from the configuration
system; `vm-ranker/params.rs` declares the same feature-switch keys for the reranking service.
At runtime both services resolve the same key from the same configuration system, so the
hard-coded values are fallbacks — but they are also what a reader of this repository sees, so
where they disagree that is called out below.

| head | key (`rust_home_mixer_…`) | `home-mixer` | `vm-ranker` | note |
| --- | --- | --- | --- | --- |
| favorite | `favorite_weight` | **0.66** | **0.5** | defaults disagree |
| reply | `reply_weight` | 5.0 | 5.0 | +`bidirectional_follow_reply_weight_boost` when eligible |
| retweet | `retweet_weight` | 1.0 | 1.0 | |
| quote | `quote_weight` | 5.0 | 5.0 | |
| share | `share_weight` | 2.0 | 2.0 | |
| share_via_dm | `share_via_dm_weight` | 5.0 | 5.0 | |
| share_via_copy_link | `share_via_copy_link_weight` | 20.0 | 20.0 | |
| photo_expand | `photo_expand_weight` | 0.05 | 0.05 | |
| video_open | `video_open_weight` | **0.02** | **0.07** | defaults disagree |
| click | `click_weight` | 0.3 | 0.3 | |
| open_link | `open_link_weight` | 0.2 | 0.2 | |
| profile_click | `profile_click_weight` | 0.0 | 0.0 | weighted 0 by default |
| vqv | `vqv_weight` | 0.0 | 0.0 | weighted 0 by default; also gated, see §5 |
| quoted_click | `quoted_click_weight` | 0.05 | 0.05 | |
| quoted_vqv | `quoted_vqv_weight` | 0.0 | 0.0 | weighted 0 by default |
| dwell | `dwell_weight` | 0.05 | 0.05 | |
| dwell_time | `cont_dwell_time_weight` | **0.006** | **0.004** | per predicted second; defaults disagree |
| click_dwell_time | `cont_click_dwell_time_weight` | 0.4 | 0.4 | see §5 on what this head predicts |
| video_continuation | `video_continuation_weight` | — | 0.0 | vm-ranker only; 0.0 in `home-mixer` construction |
| user_video_continuation | `user_video_continuation_weight` | — | 0.0 | vm-ranker only |
| profile_visit_secs | `profile_visit_secs_weight` | — | 0.0 | vm-ranker only |
| follow_author | `follow_author_weight` | 4.0 | 4.0 | |
| post_unexplored | `post_unexplored_weight` | **0.015** | **0.02** | defaults disagree; in-network only by default |
| post_video | `post_video_weight` | 0.0 | 0.0 | weighted 0 by default |
| not_interested | `not_interested_weight` | **−52.272** | **−47.52** | defaults disagree |
| block_author | `block_author_weight` | −31.2 | −31.2 | |
| mute_author | `mute_author_weight` | −58.8 | −58.8 | |
| report | `report_weight` | −234.0 | −234.0 | |
| not_dwelled | `not_dwelled_weight` | −0.02 | −0.02 | see §5 |

Two further weights modify the reply and dwell terms rather than adding new ones, for candidates
that are not replies or reposts and whose author follows the viewer back
(`bidirectional_boost_eligible()` in [`xai-value-model/inputs.rs`](../xai-value-model/inputs.rs)):

| modifier | key | default |
| --- | --- | --- |
| reply boost | `bidirectional_follow_reply_weight_boost` | 15.0 |
| dwell boost | `bidirectional_follow_dwell_weight_boost` | 0.0 |

## 3. The transform applied to the sum

`offset_score()` in [`xai-value-model/scoring.rs`](../xai-value-model/scoring.rs):

```
if combined < 0:  (combined + negative_sum) / total_sum × 0.001
else:              combined + 0.001
```

with `NEGATIVE_SCORES_OFFSET = 0.001`. Using the `home-mixer` defaults above, `positive_sum` =
43.345, `negative_sum` = 376.292, `total_sum` = 419.637.

Two properties matter when reading a score:

- The map is monotone on each side of zero, so it preserves the ordering *within* the
  net-negative group and *within* the net-positive group.
- Every net-negative candidate lands in `(0, 0.001)` and every net-positive candidate in
  `[0.001, ∞)`, so a net-negative candidate is always ranked below every net-positive one.

`unoffset_score()` is the inverse, used when multipliers are applied before the offset
(`multiplier_pre_offset`).

## 4. Multipliers applied after the sum

These live in `vm-ranker` and are parameterised there:

| multiplier | key | default |
| --- | --- | --- |
| author diversity | `enable_author_diversity` | true |
| author diversity decay | `author_diversity_decay` | 0.5 |
| author diversity floor | `author_diversity_floor` | 0.25 |
| out-of-network factor | `oon_weight_factor` | 0.75 |
| topic-request out-of-network factor | `topic_oon_weight_factor` | 0.5 |
| new-viewer out-of-network factor | `new_user_oon_weight_factor` | 0.00001 |
| per-viewer weight perturbation σ | `weight_perturbation_sigma` | 0.0 |
| DPP reranking | `vm_ranker_dpp_enabled` | true (θ = 0.65, max selected rank = 150) |

Per-viewer perturbation, when enabled, multiplies each head weight by
`exp(σ · sign(md5("{salt}:{viewer_id}:{head}")))`.

## 5. Heads whose value is easy to misread

- **`not_dwelled`** is `ClientTweetRecapNotDwelled`, not "the viewer scrolled past". Its weight is
  −0.02, i.e. effectively inert against a positive term of order 0.1.
- **`click_dwell_time`** is a continuous action in the schema, but the training configuration
  declares it as `loss_type="binary"` with `binary_threshold=10.0`
  ([`phoenix/xrex/configs/xrecsys.py`](../phoenix/xrex/configs/xrecsys.py)), so it behaves like
  `P(clicked and dwelled ≥ 10 s)` rather than a duration.
- **`dwell_time`** is a duration in seconds, trained with a MAE head normalised at 30 s (and a
  Tweedie variant for the gallery surface).
- **`vqv`** applies only when the post has video longer than `min_video_duration_ms` (10 000) and
  the *viewer* has fewer than 10 000 followers (`vqv_eligible()` in
  [`home-mixer/util/candidates_util.rs`](../home-mixer/util/candidates_util.rs)). It is 0.0 by
  default anyway.
- **`post_unexplored`** is a positive term for originals whose views are below an exploration
  target derived from follower count, and by default it applies to in-network candidates only
  (`post_unexplored_include_out_of_network` = false).

## 6. A predicted head that carries no weight

`bookmark` is predicted and logged but has no term in `weighted_terms()`:

- the action exists in the schema (`ActionGroup.P_BOOKMARK`, `CLIENT_TWEET_BOOKMARK`), and the
  serving-side `PhoenixScores` exposes `bookmark_score`;
- it is logged as `phoenix.<cluster>.bookmark` and appears in the user-debias head list;
- [`xai-value-model/phoenix_scores.rs`](../xai-value-model/phoenix_scores.rs) has no `bookmark`
  field, and `shared_phoenix_scores()` in
  [`home-mixer/scorers/value_model.rs`](../home-mixer/scorers/value_model.rs) does not carry it.

So a bookmark cannot change the ranking score directly. It still reaches the model as part of the
viewer history, where `ClientTweetBookmark` is one of the positive engagements in
[`home-mixer/util/viewer_history.rs`](../home-mixer/util/viewer_history.rs).

## 7. What this document does not cover

Retrieval (`phoenix/xrex/models/recsys_two_tower_model.py`, `simclusters`, `thunder`), ad ranking
(`home-mixer/ads/`), and whether a post may be shown at all (`visibility-filtering/`). The
page-level RL rerank (`phoenix/xrex/models/page_rl.py`) has its own serve-score weights and is
disabled in the defaults this repository ships.
