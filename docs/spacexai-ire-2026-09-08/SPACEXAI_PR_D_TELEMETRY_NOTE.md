# PR-D — Creator-visible follow telemetry (Tranche 1 companion)

**Goal:** one balance sheet for SpaceXAI + creators before heavier scorer PRs (B/C).  
**Pair with:** PR-A param A/B (+ optional joint RT/quote send). Measure-first.  
**Repo context:** x-algorithm @ `902a06f` — side effects already log follow_author; this is product/analytics surface, not a new Phoenix head.

---

## Expose (aggregates only — not per-viewer P)

1. **follow / impression** — 28d rolling, post + account  
2. **Serve contribution mix** — % of ranked serves where top positive `w·P` term was CopyLink vs FollowAuthor (and optionally ProfileClick when non-zero)  
3. **Verified HT / ALL** — only when OCR-eligible fields already exist in prod; **do not invent hydrators or fields**  
4. **RT vs original serve mix** (PR-F2a) — for amplified posts: % serves as original `tweet_id` vs RT-of-original  
5. **Quote attribution** (PR-F3) — follows on quoter vs profile clicks on nested author; QuotedClick contribution (weight default 0.05 today)

## Why with PR-A / F1

Bidirectional-style A/B needs a shared KPI. Today Studio “New follows” has no verified split, and API organic has no follows-per-post. Without PR-D, SpaceXAI and creators cannot debug the same conversion / slot sheet the param and dedup changes are meant to move.

## Non-goals

- New Phoenix training label  
- Publishing raw P(follow) to creators without product review  
- Ranking / dedup changes (PR-A / PR-F1)  
- Seen/served policy changes (PR-F2b — deferred)

## Success

Creators and SpaceXAI can answer, on the same numbers: “was this traffic share-dominated, follow-capable, or RT-slot-substituted?” before anyone ships OON gates or OCR terms.
