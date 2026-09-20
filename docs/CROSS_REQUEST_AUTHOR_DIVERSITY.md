# Cross-request author diversity (#202)

A viewer can receive a different post by the same author on every For You request because the existing author multiplier only counts candidates in that request. This opt-in change seeds its exponent with decayed author exposures from the existing served-history path. For example, with history weight 1 and no elapsed time, an author's first post on the next request gets 0.625 instead of 1.0 after one prior exposure (decay 0.5, floor 0.25).

## Behavior and integration

`ServedHistoryQueryHydrator` already reads viewer/timeline/platform-scoped history before `ScoredPostsSource` clones the query into the scoring pipeline. The new `served_history_loaded` flag distinguishes a successful empty read from a skipped/failed read; it is internal and is not a protocol field. No new RPC, persistent store, schema field or client-supplied author list is required.

With treatment enabled, `RankingScorer` computes:

```
history[a] = weight * min(max_count, sum(2 ** (-age_ms / half_life_ms)))
exponent  = current_pool_count[a] + history[a]
multiplier = floor + (1 - floor) * decay ** exponent
```

The same computation is used in both `MultiplierPreOffset` branches. The existing negative-net-score rule in the pre-offset branch is preserved. Cached SlateContext remains logging/context data and cannot override the fresh history computation.

Both the candidate pool and historical exposures use original-content authors in treatment: nonzero `retweeted_user_id` with `author_id` fallback, matching the existing writer's `source_author_id`. Disabling treatment preserves the existing serving/retweeting-author pool key and all original multiplier values. This identity normalization is part of the experiment even on a successful empty history read.

## Exposure semantics

- Count served organic feed items, not retrieved candidates or confirmed client impressions.
- Only `EntityIdType::TWEET` contributes; advertisements and other modules do not.
- Require valid positive post/author IDs and a known, non-future timestamp within the configured window.
- Deduplicate by `(served_id, sort_index, source_author_id)`. The writer expands conversation ancestors into entries sharing position and source author; this prevents one feed item from counting several times.
- If a batch has no nonzero served ID, use its timestamp; without a valid position, use the original post ID (or post ID fallback). Missing metadata cannot provide perfect event identity; these are conservative fallbacks.
- Re-reading/replaying one batch is idempotent. Serving the same content in a distinct request is another exposure, since this feature measures delivery fatigue rather than unique lifetime posts.
- Prefer the newest events before applying the global item budget. Duplicate batches do not consume extra budget. Tie ordering is deterministic.

Author history is derived on the request and never accumulated from the unselected candidate pool. The existing writer continues to record the final feed selected by the blending pipeline.

## Configuration

All names below are feature-switch parameter identifiers; corresponding wire keys are in `home-mixer/params/param.rs` and the test override example.

| Parameter | Default | Meaning |
|---|---:|---|
| EnableCrossRequestAuthorDiversity | false | Controlled opt-in for For You only |
| AuthorDiversityHistoryWindowSeconds | 300 | Exclude events at or beyond the window |
| AuthorDiversityHistoryHalfLifeSeconds | 60 | Time for one exposure's contribution to halve |
| AuthorDiversityHistoryWeight | 0.25 | Historical versus current-pool strength |
| AuthorDiversityHistoryMaxCount | 5 | Per-author historical count cap, before weighting |
| AuthorDiversityHistoryMaxItems | 200 | Maximum newest distinct exposure items considered |

The experiment also requires `EnableAuthorDiversity` and the existing served-history read/write path (`EnableUrtMigrationComponents`) to be enabled. Defaults are starting points for evaluation, not tuned estimates of user fatigue. Non-For-You requests are unchanged.

History weight 0 disables treatment. Non-finite or invalid configuration, an unsuccessful read, or an oversized history input falls back to the exact legacy path. Window is limited to one hour, weight to (0,1], recent items to 2,000, and per-author cap to (0,2,000]. A hard 20,000 inspected-node budget bounds traversal/allocation of unexpectedly large history; it falls back rather than using an order-biased partial scan. Invalid individual entries are skipped. Sorting/storage cost is bounded; no cross-request process-local cache is introduced.

## Validation

`tools/author-diversity-tests/run.py` runs 37 offline tests, including the full new modules and both scorer branches, against explicit dependency adapters. Production regression tests are also in `ranking_scorer.rs` for internal CI. The optional mutation check restores the disconnected-history behavior in a temporary file and verifies the cross-request test fails.

Additional scheduling checks enumerate 90 legal three-request interleavings, run an eight-thread read barrier, cover 20 write-delay scenarios and execute 1,000 seeded history-property trials. They confirm the documented stale-snapshot limitation: 36 schedules have all reads before the first commit, and all eight synchronized readers obtain a full multiplier. These are reproduced counterexamples, not live incidence estimates or proof of atomic frequency control. See the test README and emitted `simulations.json` for assumptions and traces.

The deterministic demonstration has 12 requests, four fresh candidates per request, and two selected posts. A has input scores 100/99; each request has two other authors scoring 70/69. Results using the production fixed functions:

| History weight | A slots / 24 | Unique authors |
|---|---:|---:|
| 0 (legacy) | 12 | 13 |
| 0.25 | 7 | 18 |
| 1.0 | 3 | 22 |

This isolates the ranking mechanism. It is not a production A/B test or a measurement of engagement/retention. Selection is a controlled top-2 fixture without DPP, visibility filtering or model inference.

## Limits and production gate

This fixes the lack of history in author scoring once the existing history read sees prior deliveries. It does **not** make distributed serving linearizable: concurrent requests or a refresh racing the asynchronous history write can still see the same older snapshot. Enforcing an atomic cross-request quota would require a separate storage/serving consistency contract and is outside this soft-decay change.

Served does not mean viewed; prefetch can overstate fatigue. History remains scoped by the existing timeline/client-platform key, so it does not guarantee unified cross-device exposure accounting. The storage reader caps batches, and missing historical author/position metadata reduces accuracy.

The existing storage lkey is an inverted millisecond timestamp without a request ID. Two requests in the same partition and millisecond generate identical keys. The local last-write-wins model demonstrates a lost-history risk; actual store semantics and migration/cleanup behavior must be checked before changing this key format. This is an existing storage-path concern, separate from the new score integration.

The multiplier has a nonzero floor. A very high-scoring author may still dominate; the change provides no hard author cap. Such a cap would be a separate selector policy with candidate-shortage and strong-relationship tradeoffs. Do not label this a guarantee of a repetition-free feed.

Before enabling live traffic, internal CI must build the complete service against real schemas and run its existing tests; the public checkout lacks the home-mixer manifest/internal dependency graph. Validate history coverage, write-to-read delay, event identity, final-feed author distribution, rank displacement, p95/p99 latency, meaningful engagement and negative feedback. Use a viewer-level experiment and retain the default-off rollback switch.
