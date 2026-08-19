# Behavioral Detection Sequence Model (BDSM)

A sequence-of-actions transformer that detects inauthentic (bot / spam /
coordinated) accounts from their behavioral event streams, together with the
task-head training stack and a reference streaming scoring pipeline.

## What's here

```
bdsm/
├── runtime/     Backbone, task heads, scoring pipeline, results sink
├── training/    Task-head training on cached backbone activations
├── proto/       Protobuf definitions for scoring events
└── rust/        Rust components (Kafka accumulator, PyO3 feature extractor)
```

## The model

`runtime/model.py` — a bidirectional transformer encoder over a user's recent
action sequence, with:

- **Time-aware RoPE**: rotary position embeddings driven by normalized action
  timestamps rather than token index, so the model natively represents
  inter-action timing (burstiness, mechanical cadence).
- **Grouped-query attention**, RMSNorm, and SwiGLU.
- **Per-action features** (action type, product surface, dwell time,
  device/client signals, engagement-target hashes, and more), extracted from
  raw Arrow IPC sequence bytes by the Rust `abuse-v3-features` extractor and
  rewritten by `runtime/feature_norm.py` to the training-time input dialect.
- **Eight task heads** (`runtime/heads.py`): FollowBot, LikeBot,
  EngagementAmplifier, ReplySpamBot, TweetSpamBot, RTBot, MultiActionBot,
  and LegitimateUser. Trained with class-balanced BCE plus a symmetric
  reverse-CE noise-robust term, with focal weighting wired through to the
  loss (`runtime/loss.py`).

The backbone is frozen at serving time. Task heads (`runtime/task_heads.py`)
are an MLP over the backbone CLS embedding, trained separately on cached
activations (`training/train_head.py`) and loaded by sha256 alongside the
backbone export (`runtime/load_backbone.py`). Weights ship as npz + per-array
sha256 in `MANIFEST.json` — no pickle deserialization surface.

## Training

1. **Backbone**: self-supervised encoder over unlabeled action sequences
   (masked-attribute prediction every step; a two-view InfoNCE contrastive
   objective every 6th step). The public artifact is the exported
   `backbone.npz`.
2. **Task heads** (`training/train_head.py`): masked multi-label SCE over the
   eight heads, trained on cached CLS embeddings. Non-finite loss or
   gradient is a hard stop (never zero-and-continue). Focal gamma is a real
   parameter with an emitted stat, asserted present at step 0.

## Runtime pipeline

```
Kafka (user action events)
  → rust/accumulator          dedup + cooldown, enqueue user_ids to Redis
  → runtime/batch_prefetcher  fetch sequences (cache-first, store fallback),
                              Rust feature extraction,
                              publish GPU-ready batches to Kafka
  → runtime/gpu_scorer        feature_norm, JAX inference on GPU,
                              publish 8-wide score rows
  → runtime/score_results_sink_focal
                              thresholds per head, dedup/cooldown ledgers,
                              graduated actioning (challenge vs. suspend),
                              BigQuery + protobuf event output
```

`proto/abuse_inference.proto` defines the events exchanged between stages
(`ScoreRequest`, `ScoreResult`, `FiredHead`, etc.).

The scorer publishes an 8-wide row in `heads.HEAD_NAMES` order.

## Dependencies and caveats

- Python ≥ 3.10; see `pyproject.toml`. GPU inference/training additionally
  requires JAX with CUDA, `optax`, and the `haiku2` neural-network library.
- The Rust feature extractor (`rust/abuse-v3-features`) builds as a PyO3
  extension module (`abuse_v3_features`). Serving pins the extractor that
  produced the training-time features; rebuild from this tree or supply a
  matching `.so`.
- The pipeline reads action sequences from an internal key-value store
  via a gRPC sidecar, and account metadata over HTTP. Those clients are
  included for completeness but the backing services are not part of this
  release; adapt the fetch layer
  (`runtime/manhattan_scorer.py:ManhattanArrowReader`) to your data source.
- The prefetcher's sequence fetch is cache-first (`runtime/sequence_cache.py`):
  a Redis GET against a historical sequence cache (enabled by default,
  `--no-seq-cache` to disable) with fallback to the sequence store and a
  best-effort read-through fill, plus an optional realtime-action merge
  (`--realtime-cache-enabled`) that appends not-yet-flushed actions from a
  Redis sorted-set cache and re-sorts the sequence before feature extraction.
  Both caches are assumed to be reachable through local Envoy redis-proxy
  listeners; every cache failure degrades to the plain store path.
- Generated protobuf bindings for the internal sequence-store schema are not
  included; code paths that import `proto_gen.recsys_pb2` degrade gracefully
  or require regenerating bindings for your own schema.
- Hostnames, broker addresses, and project names in defaults are placeholders
  (`localhost:9092`, `your-gcp-project`) — override them via CLI flags or
  environment variables. Weight paths (`--backbone-dir`, `--head-checkpoint`)
  have no baked-in filesystem defaults.
- Authoritative dimensions for the shipped configuration: 256 action types,
  8 classification heads, sequence length 512, embedding width 1024.
- The results-sink enforcement **operating points** — the per-head decision
  thresholds in `runtime/sink_policy.yaml` (and the matching fallback defaults
  in `score_results_sink_focal.py`) — are **redacted** in this public release:
  they ship as an out-of-range `9.99` sentinel (the fields are probabilities in
  `[0, 1]`, so `9.99` never fires and is plainly a placeholder, not a real
  value). The min-actions enforcement gate (a count, not a probability) is
  redacted the same way with an impossible `999999` sentinel — far longer
  than any scoreable sequence. Publishing exact operating points would hand
  adversaries the detector's evasion boundary — including the account-size
  floor below which scoring never fires. The policy *structure*, head names,
  and gate logic are real and unredacted; only the tuned numbers are
  withheld. Supply your own via `--policy-file` / `BDSM_SINK_POLICY`.
- Per-head **appeal-note templates**: the production sink interpolates a
  short prose paragraph from the dominant bot head and selected histogram
  counts (`build_enforcement_note` in `runtime/score_results_sink_focal.py`).
  The public package keeps the **gates** (the min-actions gate — value
  redacted, wired from the policy — and the dominant-head pick) and the
  `enforcement_note` proto field. The template *strings* and the
  per-head `key_actions` interpolator are the sentinel `"<redacted>"` —
  same idea as the `9.99` operating points. When a note would have fired
  it carries that sentinel plus the model-head suffix, not the internal
  appeal paragraph or the action types each head keys off. ActionName
  proto enums are unchanged.
