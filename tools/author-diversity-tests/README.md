# Cross-request author diversity contract tests

Run from the repository root with Python 3.9+ and Rust 1.90+:

```sh
RUSTC=/path/to/rustc python3 tools/author-diversity-tests/run.py --mutation-check
```

If `rustc` is on PATH, omit `RUSTC`. No Cargo dependencies or network access are needed. Logs, the generated harness and source checksums go to a printed temporary directory; set `AUTHOR_DIVERSITY_TEST_OUT` to retain them at a chosen location.

The public repository has no home-mixer Cargo manifest and does not include its internal dependency graph. These are **isolated contract tests**, not a full service build. Internal CI still needs to compile and test home-mixer against the real generated schemas and feature-switch library.

The runner compiles:

- The complete production `util/author_diversity.rs` and `util/author_history.rs` modules, directly by path.
- Verbatim extracted production scorer methods, including the entire async `score` method and both offset branches.
- Verbatim served-history reader/update and writer helpers, old-post filter, and new production regression test bodies.
- Parameter defaults parsed from the actual `param!` declarations.

The fixture explicitly replaces protocol types, parameter storage, the history client, model score inputs, out-of-network weighting and cold-start adjustment. The real scorer's action-weight computation is replaced with controlled positive/negative scores. This does not test Phoenix inference, real RPC/storage, DPP or final visibility filtering. Viewer isolation tests verify the real reader passes the viewer ID to a keyed mock store; they do not validate production storage isolation.

Tests cover continuous requests, original authors through reposts, expanded conversations, duplicate batches, temporal decay/expiry, bounded state, advertisements, malformed metadata, unsuccessful history reads, cached contexts, disabled controls and both score branches. The regression copies in `ranking_scorer.rs` also remain available for internal tests using the actual service types.

The suite now has 37 tests. `scheduling.rs` adds a deterministic single-viewer/timeline/platform storage model with:

- All 90 legal read/write interleavings of three requests (each write follows its own read/rank).
- 20 request-interval/write-visibility-delay combinations and both operation orders at equal timestamps.
- Eight real OS threads synchronized by a barrier so all reads complete before writes.
- Write drops, read failures, recovery, out-of-order write completion and replayed history.
- 1,000 seeded property trials for history permutation, replay idempotence, finite values and count caps.
- A same-millisecond storage-key collision, using the production lkey expression verbatim.

The simulator assumes snapshots see committed writes, and models colliding keys with last-write-wins. These are explicit model choices, not a reproduction of Manhattan's distributed consistency protocol. The production lkey is only an inverted timestamp; two requests in the same partition/millisecond generate the same key. The test demonstrates the consequence *if* colliding puts overwrite. Actual overwrite, retry and cleanup semantics require internal validation.

Passing these tests includes successfully reproducing known race windows. It does not mean concurrency was fixed: 36 of the 90 schedules let all three reads precede the first write, and all eight barrier-synchronized readers see the old snapshot. Once history is visible, the real scoring functions use it correctly. Schedule counts are not estimates of live probability.

`simulations.json` contains 117 records, including per-schedule traces, the full delay matrix and summary assertions. The runner checks that all 90 schedules and 20 matrix cases emitted results. Compilation/test timeouts protect against a stalled fixture; concurrency tests use logical time/barriers rather than sleeps.

The optional mutation check sets the historical contribution to zero **only in a temporary generated file** and requires the cross-request regression to fail. Its expected failure is saved separately; the fixed implementation must pass all tests.

`treatment.json` is an example feature-switch override map for a controlled test. It does not change any local or production deployment configuration. See `docs/CROSS_REQUEST_AUTHOR_DIVERSITY.md` for rollout constraints.
