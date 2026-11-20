# RFC: Cross-Request Semantic Fatigue for For You Ranking

**Status:** Proposal for maintainer review

**Scope:** Documentation and experiment design only; no runtime behavior changes

**Repository audit:** Public repository state reviewed on August 30, 2026

## Summary

This RFC proposes an experiment for measuring and, only if justified by the data, modeling **viewer-specific semantic fatigue across For You requests**.

Exact post deduplication prevents the same object from being shown again. Current-slate diversity reduces repetition among candidates selected together. Neither mechanism directly represents the possibility that a viewer has encountered several different posts expressing the same narrow proposition over several requests.

The proposed experiment asks:

> After a viewer has recently been exposed to multiple posts from the same sufficiently narrow semantic family, does another equivalent recommendation have lower marginal utility?

The first milestone is measurement. It should derive historical exposure features from serving/impression data and test whether they predict trusted dissatisfaction or reduced utility after controlling for ordinary ranking signals. Only a validated relationship should justify a bounded online adjustment.

This RFC does not claim that an equivalent production system does not exist outside the public repository. It documents a testable gap in the behavior established by the published code and identifies the contracts that an implementation would need to confirm.

## Motivation

A feed can avoid the same post twice and still feel repetitive when different authors express the same narrow claim in slightly different words. For example:

- “This battery chemistry will make charging much faster.”
- “The next battery breakthrough will mainly be about charging speed.”
- “Faster charging is the important consequence of this new battery design.”

These are different post IDs and may come from different authors. The viewer may still experience them as repeated exposure to one proposition.

That failure mode is distinct from:

1. showing the same post twice;
2. showing too many posts from one author; or
3. placing several similar candidates next to one another in one slate.

The relevant state is temporal:

> What narrowly equivalent content has this viewer already been exposed to across recent feed requests?

This RFC calls that state **historical semantic exposure**.

## Scope and repository boundary

The proposal is based on the public `xai-org/x-algorithm` repository. The public snapshot may not include production schemas, experiments, logging, feature switches, or ranking components.

The claims in this document are therefore limited:

- The published code exposes several primitives that could support investigation of historical semantic exposure.
- The published code does not establish an end-to-end mechanism that joins semantic candidate information to a viewer's prior served history and applies the result to ranking across requests.
- The imported `xai_x_thrift::served_history::ServedHistory` schema is not defined in the visible Home Mixer source. This RFC does not assume that it contains semantic identifiers.

If an internal system already provides this behavior, the proposal can instead serve as a measurement and evaluation framework for that system.

## What the published code establishes

| Area | Published evidence | What it supports | Boundary of the claim |
| --- | --- | --- | --- |
| Request-level history | [`ScoredPostsQuery`](../home-mixer/models/query.rs) carries `seen_ids`, `served_ids`, and `served_history`. | The request path distinguishes exact historical exposure from the current candidate set. | These fields do not themselves represent semantic equivalence. |
| Exact exposure filters | [`PreviouslySeenPostsFilter`](../home-mixer/filters/previously_seen_posts_filter.rs) uses seen IDs and impression Bloom filters. [`PreviouslyServedPostsFilter`](../home-mixer/filters/previously_served_posts_filter.rs) uses served IDs. | Exact post-level repetition is already an explicit concern. | Exact identity is not a substitute for semantic-family history. |
| Served-history hydration | [`ForYouCandidatePipeline`](../home-mixer/candidate_pipeline/for_you_candidate_pipeline.rs) wires [`ServedHistoryQueryHydrator`](../home-mixer/query_hydrators/served_history_query_hydrator.rs) into the query path. The hydrator reads recent served history, derives served tweet IDs, and places the entries on the query. [`ServedHistoryClient`](../home-mixer/clients/served_history_client.rs) provides bounded reads and writes. | A historical state read is part of the For You pipeline before candidate processing when the relevant migration component is enabled. | The visible client reads `ServedHistory`; its external schema is not defined here. |
| Served-history writes | [`UpdateServedHistorySideEffect`](../home-mixer/side_effects/update_served_history_side_effect.rs) writes selected feed entries with a request ID and timestamp. | A post-selection event path exists for selected feed items. | The visible writer records tweet/source IDs and scores, but does not populate semantic IDs. |
| Candidate semantics | [`PostCandidate`](../home-mixer/models/candidate.rs) exposes optional `semantic_ids`. [`SlateContext`](../home-mixer/models/candidate.rs) carries `fatigue`, SID-level fields, and reconstruction-similarity fields. | Semantic/contextual metadata is represented in the candidate model. | The meaning and stability of each field must be confirmed with its owning system. |
| Existing ranking adjustments | [`RankingScorer`](../home-mixer/scorers/ranking_scorer.rs) combines predicted actions and supports soft author-diversity decay. | There is an existing pattern for bounded, configurable repetition-aware scoring. | The visible scorer does not establish that `SlateContext.fatigue` is historical semantic fatigue. |
| Current-slate diversity | [`vm-ranker/dpp.rs`](../vm-ranker/dpp.rs) and [`vm-ranker/scoring/dpp_model.rs`](../vm-ranker/scoring/dpp_model.rs) use candidate scores and embeddings to select a diverse subset of the current request. | Same-request candidate diversity is a separate, existing mechanism. | The DPP input does not represent the viewer's prior semantic exposure. |

These observations motivate an experiment. They do not justify changing ranking by themselves.

## Problem definition

The experiment needs a semantic representation that is precise enough to identify repeated propositions without turning a broad subject into an exhausted topic.

Let:

- `u` be a viewer;
- `p` be a candidate post;
- `H_u` be the viewer's recent exposure history;
- `c_l(p)` be a validated semantic-family key for `p` at granularity `l`;
- `t` be the current event time; and
- `t_j` be the exposure time for historical item `j`.

A simple decayed exposure feature is:

\[
E_l(p,u,t) = \sum_{j \in H_u} \mathbf{1}[c_l(p)=c_l(j)]e^{-(t-t_j)/\tau_l}
\]

where `τ_l` is selected from data rather than fixed by this RFC.

This is an exposure feature, not an engagement feature. A viewer does not need to click or like a post for repeated exposure to affect the next recommendation.

The event used to update `H_u` must be defined carefully. Candidates that were merely retrieved or scored should not count as exposure. The implementation should use the existing event contract that best represents content shown or confirmed as seen by the viewer, and should document that choice.

### Three related but different mechanisms

| Mechanism | Question it answers | Typical state |
| --- | --- | --- |
| Exact deduplication | Has this post or related post already been shown? | Post IDs, impression IDs, Bloom filters |
| Current-slate diversity | Are the candidates selected together too similar? | Author counts, semantic IDs, embeddings |
| Historical semantic fatigue | Has this viewer recently seen too many equivalent members of a narrow semantic family? | Viewer-relative semantic exposure with time decay |

The third mechanism should complement the first two. It should not silently replace them.

## Hypothesis

The primary hypothesis is:

> Conditional on ordinary relevance and context, increasing recent exposure to a sufficiently narrow semantic family is associated with diminishing utility and a higher probability of trusted dissatisfaction.

Potential outcomes include:

- explicit negative feedback;
- non-dwell or rapid abandonment;
- lower positive engagement;
- shorter or less frequent return sessions; and
- other satisfaction metrics that the ranking owners consider reliable.

The hypothesis is falsifiable. If the association is not robust, if it disappears after controlling for ranking quality, or if a randomized treatment lowers overall utility, the ranking adjustment should not ship.

## Non-goals

This RFC does not propose:

- a universal content-quality score;
- a judgment of author effort or intent;
- detection of whether a post was generated by AI;
- a global novelty bonus;
- a broad-topic mute or permanent topic suppression;
- a hard filter for semantically related posts;
- suppression of disagreement or opposing viewpoints;
- replacement of Phoenix relevance scoring;
- replacement of author diversity or VMRanker/DPP; or
- global propagation of one viewer's feedback to other viewers.

The proposal is specifically about **viewer-relative repeated exposure**.

## Proposed experiment sequence

### Phase 0: offline validation

No ranking changes should occur in this phase.

Construct a time-ordered analysis dataset from the strongest available exposure event. For each candidate impression, join:

1. the viewer and request identifiers;
2. the candidate and final serving position;
3. the candidate's semantic representation at each evaluated granularity;
4. prior exposure mass for the same family at several windows;
5. candidate quality and relevance signals available to the analysis; and
6. subsequent outcome labels.

The analysis should use a temporal holdout. It should not use future impressions when constructing a historical feature. Viewer-level clustering or another method appropriate for repeated observations should be used when estimating uncertainty.

At minimum, compare:

- no prior same-family exposure;
- low exposure;
- medium exposure; and
- high exposure.

The analysis should answer:

1. Does dissatisfaction rise with same-family exposure?
2. Is the relationship monotonic, or does it have a useful saturation point?
3. Does it remain after controlling for predicted utility, author repetition, candidate age, language, network status, request type, and session activity where available?
4. Does the relationship differ by semantic granularity?
5. How quickly does the relationship decay?
6. Does it differ between in-network and out-of-network content?
7. Are there classes of content where repeated exposure remains useful, such as developing events?
8. Is semantic-family coverage sufficient for a reliable online experiment?

Offline association is not a causal result. It is a gate for deciding whether a randomized online test is worth the operational cost.

### Semantic representation validation

The semantic key is the central quality dependency. A broad topic such as `artificial intelligence` should not become exhausted because a viewer saw several different AI posts. A narrow proposition such as `this battery design will materially reduce charging time` may be a useful unit of repeated exposure.

The representation should therefore be evaluated at multiple granularities. The SID fields visible in [`SlateContext`](../home-mixer/models/candidate.rs) suggest that hierarchical semantic context exists somewhere in the published interfaces, but this RFC does not assign meanings to those levels.

Before ranking use, evaluate a labelled sample of post pairs or another trusted benchmark for:

- same proposition versus same broad topic;
- paraphrase versus genuinely new information;
- disagreement versus redundancy;
- event update versus repeated event mention; and
- missing or ambiguous semantic identifiers.

Exact semantic-vector equality may be a conservative baseline, but it should not be assumed to be the correct production key. A representation owner should confirm the key, hierarchy, collision behavior, and versioning policy.

### Phase 1: online shadow hydration

Hydrate and log the proposed exposure features for a small randomized population without changing scores or order.

This phase should validate:

- online and offline feature agreement;
- state coverage and missing-key rates;
- state read/write latency;
- state-store error behavior;
- duplicate or retried request handling;
- the relationship between selected output and recorded exposure events; and
- the size and churn of bounded per-viewer state.

If the online path cannot fail open within the existing latency budget, it should not proceed to ranking treatment.

### Phase 2: bounded online adjustment

Only if Phase 0 and Phase 1 support the hypothesis should an online treatment be considered.

For an eligible candidate, define a configurable multiplier:

\[
M_{fatigue}(p,u) = f(E_1,E_2,\ldots)
\]

with:

\[
m_{min} \leq M_{fatigue}(p,u) \leq 1
\]

and apply it to the appropriate existing score path:

\[
Score'(p,u) = Score(p,u) \times M_{fatigue}(p,u)
\]

The treatment must be:

1. **bounded:** repeated exposure cannot become an implicit hard filter;
2. **monotonic:** more recent equivalent exposure cannot increase a candidate's score;
3. **time-decaying:** old exposure loses influence;
4. **viewer-specific:** one viewer's history does not globally suppress a family;
5. **configurable:** strength, floor, windows, and eligibility are experiment parameters;
6. **fail-open:** missing or unavailable semantic history preserves current behavior; and
7. **observable:** rank displacement and utility changes are measured directly.

This RFC intentionally proposes no numeric floor, threshold, or decay constant. Those values should come from replay and controlled experiments.

#### Conservative initial boundary

The first treatment could apply only to out-of-network recommendations. This is a proposed boundary for reducing risk, not a requirement.

Repeated recommendations from followed accounts can represent explicit subscription intent. Out-of-network recommendations are more directly controlled by the ranking system. An OON-only cell can therefore test the central hypothesis while reducing interference with explicitly followed relationships.

The experiment should still report whether an OON-only boundary creates unintended distribution or coverage effects.

#### Integration point

The adjustment must be placed deliberately relative to scoring, VMRanker, selection, post-selection filtering, and the final exposure event. A conceptual flow is:

```text
request history hydration
        |
        v
candidate retrieval and existing filters
        |
        v
Phoenix / existing ranking
        |
        v
historical semantic-exposure feature hydration
        |
        v
bounded, experiment-controlled adjustment
        |
        v
VMRanker / current-slate diversity
        |
        v
selection and final visibility filters
        |
        v
final exposure event
        |
        +--> update bounded viewer state
```

The exact location should be selected using score-cache semantics, latency, final-output observability, and the interaction with the existing reranker. The treatment must not update history from candidates that were not part of the final exposure event.

### Phase 3: explicit feedback propagation

Only after ordinary exposure fatigue is understood should explicit feedback be considered as an additional signal.

A conceptual model is:

\[
F_l = E_l + \beta R_l
\]

where `E_l` is ordinary exposure mass, `R_l` is decayed high-confidence rejection mass, and `β` is experimentally tuned.

This phase should remain separate because explicit feedback has different semantics from passive exposure. One negative action should not permanently suppress a broad subject. Repeated, narrow, viewer-specific feedback may justify a stronger local signal, but it should not become a global distribution penalty.

## Viewer state and data contract

A production implementation does not need to retain every historical post or raw text. A compact state could contain:

```text
semantic_family_key
semantic_level
decayed_exposure_mass
last_exposed_at_ms
optional_decayed_rejection_mass
```

The state should have the following properties:

- bounded active-family count;
- TTL or explicit expiration;
- lazy temporal decay;
- compact semantic identifiers instead of raw content text;
- a version or namespace for semantic-key changes;
- deletion behavior consistent with existing viewer-data controls; and
- fail-open behavior when the state store is unavailable.

### Event correctness

The state update is sensitive to event semantics and retries.

- Count only the event that the product defines as exposure. Do not count retrieval, scoring, or a candidate later removed by visibility filtering.
- Preserve a request or event identifier so retries do not double-count the same exposure.
- Keep request time and exposure time distinct if client-confirmed impressions can arrive later.
- Decide whether polling, bottom requests, Following, Topics, and other surfaces share state or use separate namespaces.
- Record enough metadata to analyze surface, network status, language, and experiment assignment without retaining raw post text for this feature.
- Define behavior when semantic identifiers are absent, stale, or produced by a newer representation version.

The visible served-history path is a possible integration point, but its imported thrift schema must be confirmed before any field or serialization change is proposed. The first phase should prefer existing logs or a shadow event rather than inventing a public field such as `served_history.semantic_ids`.

## Metrics

### Primary outcome

The primary outcome should be a trusted user-utility or dissatisfaction measure selected by the ranking owners. It should be evaluated by historical exposure bucket and by randomized treatment assignment once an online experiment begins.

### Repetition metrics

#### Semantic Reappearance Rate

For a sufficiently narrow semantic family `c`, measure the fraction of the next `K` exposure events that belong to `c` after it reaches a validated exposure threshold:

\[
SRR@K = \frac{\text{same-family exposures among the next }K\text{ events}}{K}
\]

This measures repeated exposure directly. It should not be treated as a success metric without utility guardrails.

#### Semantic exposure concentration

For a window of exposure events, let `p_c` be the share assigned to family `c`:

\[
SEC = \sum_c p_c^2
\]

Higher concentration indicates that exposure is dominated by fewer families. SEC can be reported at multiple granularities, but it should not be optimized in isolation. Maximum semantic variety is not necessarily maximum utility.

### User-utility metrics

Compare exposure buckets and experiment cells for:

- explicit negative feedback;
- non-dwell or rapid abandonment;
- positive engagement;
- dwell or active time using the product's preferred interpretation;
- session continuation and return behavior; and
- other trusted satisfaction measures.

### Operational and distribution guardrails

Monitor:

- semantic-key coverage and missing-key rate;
- precision of the semantic-family grouping;
- latency and timeout rate;
- state read/write failures;
- active-family count and storage size;
- score and rank displacement;
- in-network and out-of-network distribution;
- language and content-type coverage;
- developing-event coverage where a trusted signal exists; and
- deletion and retention compliance.

If repetition falls while overall utility, important-information coverage, or trusted safety/integrity measures worsen, the treatment should be stopped.

## Experiment design and decision rules

Use stable viewer-level randomization so the same viewer does not move between policies during a short evaluation window.

At minimum, compare:

1. control: current ranking and current diversity mechanisms;
2. shadow: exposure features logged with no ranking effect; and
3. treatment: current behavior plus the bounded historical-fatigue adjustment.

For the online treatment, keep the feature disabled by default outside the experiment and provide a configuration-level rollback. Compare the treatment against current author diversity and current-slate diversity. Do not attribute a change in repetition to historical fatigue without measuring the existing mechanisms separately.

Stop or revise the proposal when:

- no reliable exposure–dissatisfaction relationship is found;
- the relationship disappears after controlling for ordinary relevance;
- semantic-family precision is inadequate;
- coverage is too low to support a useful treatment;
- important or newly updated information is suppressed at an unacceptable rate;
- the treatment reduces trusted utility; or
- latency, storage, privacy, or deletion costs are disproportionate.

The result should be a decision about whether the feature belongs in ranking, not a requirement to ship a diversity metric.

## Risks and mitigations

### Broad-topic suppression

Several posts about a broad subject should not exhaust the subject.

**Mitigation:** evaluate multiple semantic granularities, use narrow-family validation, and keep penalties bounded.

### Disagreement treated as redundancy

Two posts can concern the same subject while making opposing claims. A topical similarity signal could incorrectly suppress a useful counterargument.

**Mitigation:** validate proposition-level grouping, include disagreement pairs in the evaluation sample, and keep the first treatment soft.

### Developing events

Repeated coverage of an event can contain new information: a revised measurement, a new warning, or a confirmed resolution.

**Mitigation:** measure event-update cases separately and use conservative eligibility until an event-freshness signal is available.

### Filter narrowing

Reducing redundant exposure should not become a general instruction to show fewer subjects or fewer perspectives.

**Mitigation:** report semantic coverage and viewpoint-related guardrails where trusted measurements exist. Do not reward novelty by itself.

### Feedback poisoning

Coordinated actions should not globally suppress a semantic family.

**Mitigation:** keep exposure and explicit feedback state viewer-specific, bounded, and subject to existing abuse controls.

### Missing or changing semantic identifiers

Coverage gaps or representation-version changes can make the feature inconsistent.

**Mitigation:** version the key namespace, log coverage, and apply no penalty when the candidate or history cannot be represented safely.

### State and privacy cost

Viewer-level history adds storage, retention, deletion, and consistency requirements.

**Mitigation:** cap state, expire old entries, avoid raw text, make updates idempotent, and complete privacy review before online treatment.

### Latency and failure behavior

An additional history read must not turn a recommendation dependency into a blocking failure.

**Mitigation:** establish a timeout budget in shadow mode, measure tail latency, and fail open to the current rank when the feature is unavailable.

### Adversarial evasion

Exact text matching is easy to evade, but an additional generative quality judge would introduce a larger and separate adversarial surface.

**Mitigation:** evaluate semantic representation quality without making the score externally predictable, and do not add a subjective quality classifier as part of this proposal.

## Relationship to current-slate diversity work

Open [PR #37](https://github.com/xai-org/x-algorithm/pull/37), “Add experimental slate-level diversity reranking,” addresses repetition among posts selected together in one slate. It proposes author caps and semantic-ID checks within the current selection.

This RFC addresses a different time horizon:

| Work | Question |
| --- | --- |
| Current-slate diversity | Are the posts selected together too repetitive? |
| Historical semantic fatigue | Has this viewer already encountered this narrow semantic content repeatedly across earlier requests? |

The mechanisms can coexist. A sequence can be diverse within every individual slate and still repeat the same narrow proposition across many requests. This RFC does not depend on PR #37 and does not replace it.

## Research basis

The following work motivates investigating repeated exposure. It does not establish that the proposed feature will improve X's ranking outcomes or that its models should be copied directly:

1. [Li et al., “Modeling User Fatigue for Sequential Recommendation” (SIGIR 2024)](https://arxiv.org/abs/2405.11764) models fatigue using relationships between a target item and historically exposed items, with temporal representations.
2. [Li et al., “FAN: Fatigue-Aware Network for Click-Through Rate Prediction in E-commerce Recommendation”](https://arxiv.org/abs/2304.04529) studies fatigue from repeated similar recommendations and implicit negative behavior.
3. [Cao et al., “Fatigue-aware Bandits for Dependent Click Models”](https://arxiv.org/abs/2008.09733) formalizes reduced attractiveness under repeated exposure to similar content in a sequential decision setting.

The relevant takeaway is that cross-request exposure is a recognized recommendation problem worth measuring. The required semantic representation, event definition, treatment strength, and utility trade-offs remain specific to this system.

## Open questions for maintainers

1. Does production already maintain an equivalent viewer-level semantic-exposure feature outside this repository?
2. What are the intended meanings and stability guarantees of `semantic_ids`, `sid_k_l1`, `sid_k_l2`, `sid_k_l3`, and `SlateContext.fatigue`?
3. Which event—selected response, client-confirmed impression, or another existing signal—should count as exposure?
4. Does the served-history schema already carry a usable semantic representation, or should this experiment use a separate event path?
5. Should exposure state be shared across For You requests only, or across additional surfaces and request types?
6. Where should the adjustment occur relative to score caching, VMRanker/DPP, selection, and final visibility filtering?
7. Which utility and dissatisfaction metrics are trusted enough to serve as primary outcomes?
8. What event/news freshness signal can protect legitimate repeated updates?
9. What state bound, retention policy, and deletion contract are acceptable?
10. Is a logging-only online implementation the safest first contribution if the required production schemas are not public?

## Recommendation

Proceed only with Phase 0 and, if its results support the hypothesis, a logging-only Phase 1. Do not add a ranking parameter or modify the imported served-history schema until the semantic key, exposure event, state contract, and primary utility metric have owners.

If those prerequisites are validated, run a small, OON-first, bounded randomized treatment with a kill switch and explicit rank-displacement and utility guardrails. If they are not validated, retain the work as an analysis proposal rather than adding a speculative ranking mechanism.

## References to the published code

- [`README.md`](../README.md): scoring, author diversity, filtering, candidate isolation, and VMRanker overview.
- [`home-mixer/models/query.rs`](../home-mixer/models/query.rs): request-level seen, served, and served-history fields.
- [`home-mixer/filters/previously_seen_posts_filter.rs`](../home-mixer/filters/previously_seen_posts_filter.rs): exact seen-post filtering.
- [`home-mixer/filters/previously_served_posts_filter.rs`](../home-mixer/filters/previously_served_posts_filter.rs): exact served-post filtering.
- [`home-mixer/query_hydrators/served_history_query_hydrator.rs`](../home-mixer/query_hydrators/served_history_query_hydrator.rs): served-history hydration and recent served-ID derivation.
- [`home-mixer/clients/served_history_client.rs`](../home-mixer/clients/served_history_client.rs): served-history read/write interface and bounded scan.
- [`home-mixer/side_effects/update_served_history_side_effect.rs`](../home-mixer/side_effects/update_served_history_side_effect.rs): selected-feed served-history writes.
- [`home-mixer/models/candidate.rs`](../home-mixer/models/candidate.rs): candidate semantic IDs and slate context.
- [`home-mixer/scorers/ranking_scorer.rs`](../home-mixer/scorers/ranking_scorer.rs): weighted ranking, negative-feedback predictions, and author-diversity decay.
- [`vm-ranker/dpp.rs`](../vm-ranker/dpp.rs): current-candidate DPP reranking inputs and selection.
- [`vm-ranker/scoring/dpp_model.rs`](../vm-ranker/scoring/dpp_model.rs): DPP candidate preparation and embedding lookup.
