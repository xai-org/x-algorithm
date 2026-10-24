# Opt-in result budgets for intentional sessions

[Vible](https://www.holyvible.com/) and [Empowerverse](https://www.empowerverse.org/)
describe leaving a feed when inspired. A serving pipeline needs a way to honor an
explicit "done" state instead of fetching and scoring another batch.

`CandidatePipeline::request_result_limit(&query)` is an optional hook:

- `None`: retain the existing execution, static size, finalization, and side effects.
- `Some(n)`: cap final returned candidates to the smaller of `n` and `result_size()`.
- `Some(0)`: return an empty result after query hydration, before candidate sourcing,
  with the hydrated query intact. Skip candidate stages, finalization, and serving
  side effects; retain result-size statistics and pipeline summary emission.

The positive cap is enforced after finalization, so a finalizer cannot exceed the
request budget. Truncated candidates enter `non_selected_candidates`; side effects
receive exactly the final selected slice. Pre/post-selection eligibility filters
still run for positive budgets. Existing implementations do not need to change.

## Integration boundary

This PR adds the framework capability and tests. It does not create a user setting,
change a public request schema, enable a session policy, or collect exit feedback.
A serving pipeline can override the hook once authenticated query hydration makes
the user's explicit session state available. Apply the budget to the outer serving
pipeline, not indiscriminately to nested retrieval pools, which may need extra
candidates for filtering and diversity.

The unit is a pipeline candidate, not seconds or necessarily a visible post: thread
expansion, ads, carousels and other rendering can change the visible-item count.
An integration must define which items consume the chosen budget and enforce that
contract at the final response boundary. Suppress continuation cursors and client
prefetch for finished sessions; allow the user to intentionally start a new session.

Across requests, resolve an authoritative remaining budget with atomic reservations
and a session version so retries and concurrent requests cannot each spend the same
remaining budget. Account for filtered/unrendered items according to the chosen
contract. This per-request hook does not implement cross-request accounting or
cancel already-running requests. Authorization, feedback recording and session
accounting must be separate from serving side effects skipped by the zero-budget path.

Honor a generic explicit stop independently from its interpretation as positive
feedback. Only a separate intentional inspired-exit action supplies a successful
outcome label. Avoid treating inactivity, timeouts, crashes, or ordinary closure as
evidence that the session helped the viewer.

## Public-source validation

```sh
cargo test --locked --manifest-path candidate-pipeline/tests/request-budget/Cargo.toml
```

The harness imports the actual pipeline execution, stage traits, selector and
summary code. Public async/tracing crates are used normally. Only unpublished
feature-switch, decider and metrics adapters are test stand-ins, and the metrics
attribute passes the original item through unchanged. It validates the pipeline's
control flow, not production configuration, telemetry, or Home Mixer integration.

Tests cover legacy behavior, zero-budget early exit, hydrated budgets, ordered
positive truncation, accurate side-effect partitions, caps after finalization,
oversized limits, and preservation of legacy static-zero/finalizer behavior.
