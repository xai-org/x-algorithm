# Explicit control over attention rewards

Vible's [intentional feed and exit action](https://www.holyvible.com/) and
Empowerverse's [growth-oriented recommendations](https://www.empowerverse.org/)
motivate giving viewers a way to reduce how much consumption time influences
ranking. This API implements one small part of that idea: scale positive attention
rewards using an explicit viewer preference. It does not infer a person's goals or
well-being from how long they watch.

```rust,ignore
let viewer_weights = original_weights.with_attention_reward_scale(0.0)?;
let scores = compute_value_scores(&viewer_weights, &context, &candidates);
```

The scale is finite and in `[0, 1]`. One preserves the weights; zero removes their
positive contribution. Covered channels are dwell probability, quality video
views (including quoted views), continuous dwell/click dwell, video continuation,
user video continuation, profile-visit seconds, and the mutual-follow dwell boost.
Non-positive weights are retained. Likes, replies, shares, clicks, follows,
not-interested, blocks, mutes, reports, not-dwelled, and diversity settings retain
their weights. All existing entry points and defaults are unchanged.

This controls the raw weighted objective; it does not establish an empirical
benefit or make every resulting rank monotonic. Offset normalization and subsequent
diversity/reranking still apply. In particular, not-dwelled remains negative:
supporting an explicitly successful short visit requires a separate outcome signal.

## Integration boundary

This PR exposes a library method. No viewer setting, request field, experiment,
model, or service is enabled by it. A caller should apply the scale once to fresh
weights before scoring, propagate the same effective weights to remote reranking,
and invalidate cached weighted scores when the preference or weight version
changes. In the current Home Mixer, `weights_for` and the VMRanker request/local
fallback paths all need consistent treatment before enabling a viewer control.
Reapplying the method compounds the scale.

Evaluate voluntary opt-in with direct worth-my-time feedback and successful session
completion, alongside relevance and negative feedback. Reduced attention rewards
alone are not evidence of improved well-being. Existing visibility and eligibility
filters remain responsible for whether a candidate can be shown.

## Public-source tests

```sh
cargo test --manifest-path xai-value-model/tests/attention-rewards/Cargo.toml
```

The harness imports the original scoring modules directly and uses the public
`rustc-hash` and `md5` dependencies. It excludes the unpublished protobuf adapter. Tests cover
identity at one, every scaled channel, fractional scaling, preservation of negative
weights/feedback, invalid configuration, and a hand-built ranking reversal when
consumption time is the only reason one candidate outranks another. Synthetic
examples verify arithmetic, not user outcomes.
