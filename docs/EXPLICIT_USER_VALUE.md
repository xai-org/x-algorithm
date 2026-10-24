# Optional explicit-outcome utility

[Vible](https://www.holyvible.com/) describes an exit action that marks a video as
inspiring. [Empowerverse](https://www.empowerverse.org/) emphasizes useful time,
personal growth, and leaving motivated. A recommendation objective needs to
distinguish those explicit outcomes from an ordinary session ending.

`compute_value_scores_with_user_value` accepts optional calibrated probabilities of
four viewer-reported outcomes: worth my time, helped my chosen goal, inspired exit,
and regretted time. It does not infer goals, classify content as morally positive,
or reinterpret closes, crashes, skips, or inactivity as successful exits.

The adjustment, in raw weighted-score units, is:

```text
delta = clamp(w_worth * P(worth my time)
            + w_goal * P(helped my goal)
            + w_exit * P(explicit inspired exit)
            - w_regret * P(regretted time), -cap, cap)
```

Weights and cap must be finite and nonnegative. The aggregate weight sum must be
finite. Present probabilities must be finite and in `[0, 1]`; missing heads
contribute zero. The zero-cap default bypasses adjustment. A length mismatch is an
error rather than silently truncating or applying labels to the wrong candidate.

The adjustment is applied before the existing offset, exploration, cold-start,
author-diversity and out-of-network paths, by reusing `compute_value_scores`.
Existing entry points are unchanged. Existing negative-feedback contributions
remain in the base objective; the cap limits the new utility, but does not make
negative-feedback dominance a mathematical guarantee. Eligibility and visibility
filters remain separate.

## Integration boundary and feedback semantics

This is an optional scoring primitive, not a trained model or enabled feed feature.
The current Phoenix/protobuf schema does not contain these four heads. X would
need to add voluntary feedback collection, calibrated prediction heads, protocol
transport, consistent Home Mixer/VMRanker adapters, and an experiment before rollout.
Predictions must be matched by candidate ID before forming the aligned slice.

An inspired-exit label requires an explicit viewer action tied to an impression.
A chosen-goal label refers to the viewer's voluntarily specified goal, not a
platform-assigned goal. Treat absent feedback as unlabeled, not a negative example.
Use impression-level attribution and deduplication; avoid raw community vote totals
as per-viewer probabilities. Responses remain optional, editable, and revocable.

Cached inputs must contain *baseline* weighted scores under the same weights,
without prior outcome adjustment. Version the cache with model, preference, and
weight versions, or recompute the baseline. Do not repeatedly adjust cached output.
Callers must handle validation errors explicitly; retaining the baseline is a
possible fallback, with separate error telemetry.

Evaluate worth-my-time and chosen-goal feedback and successful explicit session
completion, together with relevance, unwanted content, and regret. Survey response
rates and calibration matter: a responding subset does not represent every viewer.
No improvement in well-being or causal outcome is established by the synthetic tests.
For policy evaluation with logged propensities, [Open Bandit Pipeline](https://github.com/st-tech/zr-obp)
provides existing estimators; [RecSim](https://github.com/google-research/recsim)
provides configurable sequential simulations. Neither supplies these labels or
proves that this objective is beneficial for X.

## Validation

```sh
cargo test --locked --manifest-path xai-value-model/tests/user-value/Cargo.toml
```

The harness compiles actual public modules and excludes the unpublished protobuf
adapter. It verifies no-op parity (including cached, cold-start, and both multiplier
paths), positive explicit exit, regret penalties, caps, cached/fresh equivalence,
existing diversity and out-of-network multipliers, alignment, invalid probabilities,
invalid weights, and arithmetic overflow. Full service and protocol integration
remain outside this harness.
