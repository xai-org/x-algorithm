# From Phoenix probabilities to a final score: the ranking pipeline in detail

The [README's Scoring and Ranking section](../README.md#scoring-and-ranking) describes the four
stages a candidate goes through after Phoenix predicts its action probabilities: the base
weighted score, author cold-start, author diversity, and the out-of-network discount. This doc
fills in the formulas, default values, and code references behind each stage, with worked
examples, so you don't have to reconstruct them from
[`home-mixer/scorers/ranking_scorer.rs`](../home-mixer/scorers/ranking_scorer.rs) and
[`home-mixer/scorers/author_cold_start.rs`](../home-mixer/scorers/author_cold_start.rs) yourself.

All four stages run in that order, each one operating on the output of the last:

```
base weighted score  →  author cold-start boost  →  author diversity decay  →  OON discount
```

## Stage 1: the base weighted score

Phoenix outputs a probability (or, for a few heads like dwell time, a continuous value) per
action per candidate. `RankingScorer` turns those into one number per candidate in three steps.

**`apply`** ([`ranking_scorer.rs:447`](../home-mixer/scorers/ranking_scorer.rs#L447)) is the
atomic operation — it multiplies one action's probability by its configured weight:

```rust
fn apply(score: Option<f64>, weight: f64) -> f64 {
    score.unwrap_or(0.0) * weight
}
```

**`compute_weighted_parts`** ([`ranking_scorer.rs:460`](../home-mixer/scorers/ranking_scorer.rs#L460))
calls `apply` for each of the ~26 action terms (favorite, reply, retweet, report, block, mute,
dwell, etc.) using the weights in
[`home-mixer/params/param.rs`](../home-mixer/params/param.rs), then splits the results into two
running totals: `pos`, the sum of every non-negative term, and `neg`, the summed magnitude of
every negative term (not-interested, block, mute, report, not-dwelled all carry negative
weights).

**`offset_score`** ([`ranking_scorer.rs:554`](../home-mixer/scorers/ranking_scorer.rs#L554)) takes
`combined_score = pos - neg` and remaps it:

```rust
pub(crate) fn offset_score(combined_score: f64, w: &ScoringWeights) -> f64 {
    if w.total_sum == 0.0 {
        combined_score.max(0.0)
    } else if combined_score < 0.0 {
        (combined_score + w.negative_sum) / w.total_sum * NEGATIVE_SCORES_OFFSET
    } else {
        combined_score + NEGATIVE_SCORES_OFFSET
    }
}
```

`NEGATIVE_SCORES_OFFSET` is `0.001`
([`home-mixer/params/config.rs:40`](../home-mixer/params/config.rs#L40)). Net-negative candidates
get squeezed into `[0, 0.001)`; net-non-negative candidates get shifted up by a flat `0.001`. The
effect: every candidate whose predicted actions are net-good outranks every candidate whose
predicted actions are net-bad, while relative order is preserved within each group.

**Worked example.** Using the default weights (`FavoriteWeight = 0.5`,
`ReplyWeight = 5.0`, `ReportWeight = -234.0`) and a candidate with
`P(favorite) = 0.10`, `P(reply) = 0.02`, `P(report) = 0.001`:

```
favorite term = 0.10 × 0.5    =  0.050
reply term    = 0.02 × 5.0    =  0.100
report term   = 0.001 × -234  = -0.234

pos = 0.150, neg = 0.234
combined_score = 0.150 - 0.234 = -0.084   (net negative)
```

That -234 weight looks disproportionate until you remember it's multiplying a *probability*, not
a raw count — a 0.1% predicted report chance still outweighs a 10% predicted favorite chance,
which is the intended behavior: a small chance of a severe negative action should dominate a
larger chance of a mild positive one.

## Stage 2: author cold-start

Runs in [`author_cold_start.rs`](../home-mixer/scorers/author_cold_start.rs), gated by
`EnableViewerColdStart`. New/small accounts get almost no engagement signal, so Phoenix's
probability estimates for their posts are naturally weak — which keeps them from ever being
shown, which keeps them from ever collecting signal. This stage breaks that loop by forcing one
eligible candidate per request into a specific visible slot.

**Eligibility** (`cold_start_base_eligible`,
[`author_cold_start.rs:138`](../home-mixer/scorers/author_cold_start.rs#L138), plus filters in
`apply_cold_start`, [`author_cold_start.rs:257`](../home-mixer/scorers/author_cold_start.rs#L257)):
an original post (not a reply/retweet), from an author with at most `ColdStartFollowerCap`
(default `1000`) followers, with fewer than `ColdStartImpressionThreshold` (default `1000`) views
on this specific post, fresh enough (`ColdStartMaxPostAgeSecs`), and already ranked within the
top `LowImpressionsMaxPositionRatio` fraction of the slate.

**Picking one** (`pick_thompson` / `sample_reward`,
[`author_cold_start.rs:215`](../home-mixer/scorers/author_cold_start.rs#L215)): when
`EnableColdStartThompsonSampling` is on, each eligible candidate's true engagement rate is
modeled as `Beta(alpha0 + favs, beta0 + (views - favs))` and sampled from — a standard Bayesian
bandit that lets a post with zero engagement history still occasionally win, rather than always
being assumed bad because it has no data yet.

**The boost** (`cold_start_target` / `apply_cold_start`,
[`author_cold_start.rs:182`](../home-mixer/scorers/author_cold_start.rs#L182)): sorts all
candidate scores descending and samples `target` from the window
`ColdStartSlotMin..ColdStartSlotMax` (default `15..16`, i.e. rank 15 specifically). The chosen
candidate's score is force-raised to `max(current_score, target)` — so it lands in a real,
specific slot, not an arbitrary top position, giving it a shot at genuine impressions without
letting it dominate the feed.

## Stage 3: author diversity decay

Runs in `apply_author_diversity`
([`ranking_scorer.rs:697`](../home-mixer/scorers/ranking_scorer.rs#L697)), gated by
`EnableAuthorDiversity` (default `true`). Without it, an author with several high-scoring posts
in one candidate pool could occupy a disproportionate share of the feed.

**Ranking and counting** (`compute_slate_contexts`,
[`ranking_scorer.rs:647`](../home-mixer/scorers/ranking_scorer.rs#L647)): every candidate in the
pool — all authors together — is sorted once by its current score, descending. Walking that
sorted list, each candidate is assigned `k`: how many posts from the *same author* already
appeared earlier in the sorted order. An author's single strongest post always gets `k = 0`.

**The decay formula** (`diversity_multiplier`,
[`ranking_scorer.rs:643`](../home-mixer/scorers/ranking_scorer.rs#L643)):

```rust
fn diversity_multiplier(decay_factor: f64, floor: f64, exponent: f64) -> f64 {
    (1.0 - floor) * decay_factor.powf(exponent) + floor
}
```

With the defaults `AuthorDiversityDecay = 0.5` and `AuthorDiversityFloor = 0.25`
([`param.rs:229`](../home-mixer/params/param.rs#L229),
[`param.rs:235`](../home-mixer/params/param.rs#L235)), the formula is `0.75 × 0.5^k + 0.25`:

| k (rank among this author's posts) | multiplier |
|---|---|
| 0 (author's best post) | 1.000 — no penalty |
| 1 | 0.625 |
| 2 | 0.4375 |
| 3 | 0.34375 |
| → ∞ | → 0.25 (floor, never lower) |

**Applying it**: `pre_diversity_score × multiplier`. A prolific author's weakest post is never
docked more than 75% — the curve front-loads the penalty on the second and third posts, then
flattens, rather than punishing volume indefinitely.

## Stage 4: out-of-network discount

`effective_oon_weight` / `oon_applies`
([`ranking_scorer.rs:710`](../home-mixer/scorers/ranking_scorer.rs#L710) and
[`ranking_scorer.rs:776`](../home-mixer/scorers/ranking_scorer.rs#L776)) pick a flat multiplier
based on the viewer's relationship to the post:

| condition | weight | param |
|---|---|---|
| topic feed (`topic_ids` non-empty) | `0.5` | `TopicOonWeightFactor` |
| new account, < `NewUserAgeThresholdSecs` old, ≥ 5 follows | `0.00001` | `NEW_USER_OON_WEIGHT_FACTOR` (constant) |
| everyone else, post from an unfollowed author | `0.75` | `OonWeightFactor` |
| post from a followed author | `1.0` (no discount) | — |

One exception: if `EnableOonRescoreForInNetworkRepliesRetweets` is on (default `true`), a
reply/retweet from someone you *do* follow still gets the discount — a friend retweeting a
stranger is, functionally, stranger content reaching you.

`NewUserAgeThresholdSecs` defaults to `0`, so the near-zero new-account tier can't actually fire
under default configuration — it's a wired-up but currently dormant lever.

## Putting it together

A candidate's final score is the product of all four stages applied in sequence:

```
score = offset_score(pos - neg)          # Stage 1
      → cold-start boost (max, not ×)    # Stage 2 (only for the one chosen candidate, if any)
      × diversity_multiplier             # Stage 3
      × oon_weight                       # Stage 4
```

Example: two posts both reach a Stage 1 score of `0.045`. One is an out-of-network author's
second-best post in the pool (`k = 1`); one is an in-network author's best post (`k = 0`):

```
in-network,  k=0: 0.045 × 1.000 × 1.00  = 0.0450
out-of-network, k=1: 0.045 × 0.625 × 0.75 = 0.0211
```

The out-of-network post ends up scoring less than half as much, from two independent, stackable
discounts — not one large penalty.

## A caching note: `slate_context` can look stale on purpose

On a paginated request (`query.has_cached_posts == true`), the entire candidate pool is the same
batch of already-scored posts served from Redis by
[`CachedPostsSource`](../home-mixer/sources/cached_posts_source.rs) — not a fresh retrieval. It
would be reasonable to assume Stage 3's `k`/rank values get reused from that cache too. They
don't: `RankingScorer` always recomputes `compute_slate_contexts` fresh against the current
candidates and scores whenever `has_cached_posts` is true
([`ranking_scorer.rs:794`](../home-mixer/scorers/ranking_scorer.rs#L794)), so the diversity
multiplier — and therefore the score — is always correct for the current request.

What *does* stay stale on purpose: the `slate_context` field written back onto the returned
candidate. On a cached request it's populated from `stored_slate_contexts`
([`ranking_scorer.rs:680`](../home-mixer/scorers/ranking_scorer.rs#L680)) — whatever was cached
from the original, non-cached scoring pass — not from the freshly recomputed context used to
score it. This is intentional and covered by a dedicated test,
`recomputes_contexts_for_scoring_on_cached_posts`
([`ranking_scorer.rs:997`](../home-mixer/scorers/ranking_scorer.rs#L997)). `slate_context` feeds
into the request sent to `vm-ranker`
([`vm_ranker.rs:193`](../home-mixer/scorers/vm_ranker.rs#L193)), and the likely reason for keeping
it stable is to give that external reranker a consistent per-post identity across repeated page
loads, rather than a value that jitters every time the same cached post resurfaces. If you're
reading `slate_context` off a candidate for debugging, keep in mind it may not reflect the
multiplier that actually produced that candidate's `score`.
