# Phoenix: Recommendation System

This repository contains the JAX code for the Phoenix recommendation system,
which powers content ranking and retrieval. Phoenix uses transformer-based
architectures for both **retrieval** (finding relevant candidates from
millions of items) and **ranking** (ordering a smaller set of candidates by
predicted engagement).

> **Note:** Earlier releases shipped a sample transformer ported from the
> [Grok-1 open source release](https://github.com/xai-org/grok-1). This
> release ships the **production implementation itself**: the real model
> code, the real training step, and the real Rust serving engine, exported
> from the internal tree. What is *not* shipped is xAI-specific
> infrastructure (production data feeds, cluster orchestration, internal
> telemetry) — every such seam is replaced by a documented local equivalent,
> and synthetic data generators are included so the whole system runs end to
> end with nothing external. One training-recipe substitution is disclosed in
> [TRAINING.md](TRAINING.md): for configs on the legacy dense-optimizer
> slot, the export ships standard AdamW rather than production's tuned
> internal variant. The flagship ranking configs and the nano twin train
> the production Muon recipe, which ships in full.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Two-Stage Recommendation Pipeline](#two-stage-recommendation-pipeline)
  - [Retrieval: Two-Tower Model](#retrieval-two-tower-model)
  - [Ranking: Transformer with Candidate Isolation](#ranking-transformer-with-candidate-isolation)
- [Key Design Decisions](#key-design-decisions)
- [Running the Code](#running-the-code)
- [Model Architecture Configs](#model-architecture-configs)
- [License](#license)

---

## Overview

Phoenix is a recommendation system that predicts user engagement (likes,
reposts, replies, etc.) for content. It operates in two stages:

1. **Retrieval**: Efficiently narrow down millions of candidates to hundreds,
   scoring a user embedding against a precomputed candidate index
2. **Ranking**: Score and order the retrieved candidates using a more
   expressive transformer model

### About This Release

- **The production stack, not a sample**: the shipped tree (`xrex/` plus
  the vendored `crates/` engine workspace) is the code that trains and
  serves Phoenix in production — model definitions, trainer, checkpointing,
  and the Rust gRPC serving engine (built locally via
  `uv sync --extra engine`).
- **Nano configs for one GPU**: alongside the production configs, both models
  ship single-GPU `nano` presets (`home_direct_packed_nano` for ranking,
  `xrecsys_two_tower_nano` for retrieval) that keep the production losses
  and feature handling, shrunk in width/depth/table sizes so training runs
  in minutes. The ranking nano is geometry-identical to prod; the retrieval
  nano additionally trains unpacked with dense attention, a 1022-step
  history, and no user-features token (see the comparison table below).
- **Synthetic data generators replace artifact downloads**: there is no
  checkpoint or corpus bundle to fetch. `reference/world_snapshots.py` and
  `reference/dump_gen.py` generate a deterministic synthetic world — training
  dumps, semantic-ID snapshots, a multimodal-embedding snapshot, and the
  retrieval candidate corpus — and `reference/train_synth.py` trains either
  model on it. Trained this way, the checkpoints serve real gRPC traffic
  through the same engine production uses.

---

## Architecture

### Two-Stage Recommendation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────┐     ┌─────────────────────┐     ┌─────────────────────┐          │
│   │          │     │                     │     │                     │          │
│   │   User   │────▶│   STAGE 1:          │────▶│   STAGE 2:          │────▶ Feed│
│   │ Request  │     │   RETRIEVAL         │     │   RANKING           │          │
│   │          │     │   (Two-Tower)       │     │   (Transformer)     │          │
│   └──────────┘     │                     │     │                     │          │
│                    │   Millions → 1000s  │     │   1000s → Ranked    │          │
│                    └─────────────────────┘     └─────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Retrieval: Two-Tower Model

The retrieval stage uses a **two-tower architecture** that enables efficient
similarity search at scale.

#### How Retrieval Works

1. **User Tower**: Encodes the user's engagement history through a
   transformer to produce a normalized user embedding `[B, D]`. The sequence
   is the user's history plus a single user-features token (coarse profile
   features — country, language, and more on the combined config) —
   production retrieval carries **no learned per-user ID embedding**
   (`use_user_embedding=False`); beyond those profile features, the user is
   represented by what they interacted with. (The nano preset drops the
   user-features token and is literally history-only.)
2. **Candidate Tower**: Computes normalized embeddings for all items in the
   corpus `[N, D]`. Since the semantic-ID migration, candidates are
   represented by their **semantic IDs** — residual-quantized codes (6 levels
   × 256 codes) derived from each post's multimodal embedding — plus hashed
   author IDs, rather than by hashed post IDs alone. Same-topic posts share
   SID prefixes, which gives the tower compositional generalization to unseen
   posts.
3. **Index in the checkpoint**: at every checkpoint save, the trainer runs
   the candidate tower over the configured corpus and stores the resulting
   index (`post_embeddings`) **inside the checkpoint**. Serving loads it from
   there — nothing embeds a corpus at boot time.
4. **Similarity Search**: the serving engine retrieves top-K candidates by
   dot product between the user embedding and the index.

At serving time the retrieval server hydrates history SIDs through a
semantic-ID lookup service; the launchpad ships a parquet-backed
implementation of the same contract (`reference/sid_index_server.py`).

---

### Ranking: Transformer with Candidate Isolation

The ranking model uses a transformer architecture where **candidates cannot
attend to each other** during inference. This is a critical design choice
that ensures the score for a candidate doesn't depend on which other
candidates are in the batch.

#### Ranking Model Architecture

```
                              PHOENIX RANKING MODEL
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │                              OUTPUT LOGITS                                 │
    │              [B, num_candidates, num_actions] + dwell regression           │
    │                                    │                                       │
    │                                    │ Unembedding                           │
    │                                    │ Projection                            │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │    Extract Candidate Outputs  │                       │
    │                    │    (positions after history)  │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │         Transformer           │                       │
    │                    │     (with special masking)    │                       │
    │                    │                               │                       │
    │                    │   Candidates CANNOT attend    │                       │
    │                    │   to each other               │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │    ┌───────────────────────────────┼───────────────────────────────┐       │
    │    │                               │                               │       │
    │    ▼                               ▼                               ▼       │
    │ ┌──────────┐              ┌─────────────────┐              ┌────────────┐  │
    │ │   User   │              │     History     │              │ Candidates │  │
    │ │  Tokens  │              │   Embeddings    │              │ Embeddings │  │
    │ │  [B, 2]  │              │    [B, S, D]    │              │  [B, C, D] │  │
    │ │          │              │                 │              │            │  │
    │ │ Hashes + │              │ Posts + Authors │              │ Posts +    │  │
    │ │ Profile  │              │ + Actions +     │              │ Authors +  │  │
    │ │ Features │              │ SIDs + Context  │              │ SIDs +     │  │
    │ └──────────┘              └─────────────────┘              │ Context    │  │
    │                                                            └────────────┘  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```

Since the earlier sample release, the input side has grown a **feature-prep
stage**: besides hashed post/author IDs (and, on history positions, action
embeddings), history and candidate positions carry semantic-ID embeddings
and context features (timezone, local hour-of-day, product surface, post
age); history positions additionally carry dwell time, and a user-prefix
token carries profile features (country, language, location, gender, age
bracket, installed apps). Production training also packs multiple variable-length sessions per
row (**sequence packing**) and trains with a variable-length attention
kernel; both are training-throughput mechanisms — the serving-time contract
is unchanged.

#### Attention Mask: Candidate Isolation

A key detail is the **attention mask** that prevents candidates from
attending to each other while still allowing them to attend to the user and
history:

```
                    ATTENTION MASK VISUALIZATION

         Keys (what we attend TO)
         ─────────────────────────────────────────────▶

         │ User │    History (S positions)    │   Candidates (C positions)    │
    ┌────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │                               │
    │ U  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
    │    │      │                             │                               │
    ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
 Q  │    │      │                             │                               │
 u  │ H  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 e  │ i  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 r  │ s  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 i  │ t  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 e  │    │      │                             │                               │
 s  ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │  DIAGONAL ONLY (self-attend)  │
 │  │ C  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✓   ✗   ✗   ✗   ✗   ✗   ✗    │
 │  │ a  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✓   ✗   ✗   ✗   ✗   ✗    │
 │  │ n  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✓   ✗   ✗   ✗   ✗    │
 │  │ d  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✓   ✗   ✗   ✗    │
 │  │ i  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✓   ✗   ✗    │
 │  │ d  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✓   ✗    │
 ▼  │ s  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✓    │
    │    │      │                             │                               │
    └────┴──────┴─────────────────────────────┴───────────────────────────────┘

    ✓ = Can attend (1)          ✗ = Cannot attend (0)

    Legend:
    ├─ User + History: Full bidirectional attention among themselves
    ├─ Candidates → User/History: Candidates CAN attend to user and history  
    └─ Candidates → Candidates: Candidates CANNOT attend to each other (only self)
```

---

## Key Design Decisions

### 1. Hash-Based Embeddings, Plus Semantic IDs

Both models use multiple hash functions per entity for embedding lookup —
no dictionary service, deterministic, and collision-tolerant by combining
multiple independent hash lookups per entity. Since the semantic-ID
migration, posts additionally carry **semantic IDs**: residual-quantized
codes over the post's multimodal embedding, giving the models content-aware
generalization that pure ID hashing cannot.

### 2. Shared Architecture

The retrieval user tower uses the same transformer trunk and input
machinery as the ranking model (the combined retrieval config additionally
shares ranking's project-then-sum feature-prep stage; the flagship and nano
retrieval configs use the candidate tower's `enable_linear_proj` combine —
a small concat-then-MLP — instead); the two models differ in their heads,
not their trunk.

### 3. Multi-Action Prediction

The ranking model predicts many engagement types simultaneously — one logit
per action in a shared taxonomy, trained as multi-label targets — plus
regression heads for continuous signals (dwell time):

```
Output: [B, num_candidates, num_actions] (+ continuous-action heads)
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │ Like │ Repost │ Reply │ Click │ ... │
        └─────────────────────────────────────┘
```

Retrieval trains the two towers contrastively (in-batch and sampled global
negatives with log-Q correction) with favorites as the positive signal.

### 4. Batch-Invariant Serving

Candidate isolation (ranking) and the checkpoint-baked index (retrieval)
together make serving scores independent of batch composition: a
candidate's score depends only on the user and that candidate.

---

## Running the Code

Everything below runs from this directory (the launchpad root) with no
cluster, no Kafka, and no production data. The full validated walkthrough —
including expected output and timings — is
[`QUICKSTART.md`](QUICKSTART.md); [`TRAINING.md`](TRAINING.md) maps the
training internals component by component.

### Installation

> **Validated environment.** This walkthrough is exercised end to end on the
> public `nvidia/cuda:13.2.0-base-ubuntu22.04` image (NVIDIA GB300, aarch64,
> driver 580) with nothing preinstalled beyond what this section adds. The
> code detects A100 / H100 / H200 / GB200 / GB300 (override with
> `MACHINE_TYPE=<arch>` if detection misreads your box), and the attention
> kernels ship Hopper- and Blackwell-tuned configurations — on other GPU
> families expect to adjust the serving `attn_impl` override and possibly
> kernel block sizes.

Start with the system packages the engine build and runtime need — a C/C++
toolchain, `cmake`, `pkg-config`, RDMA verbs headers, bindgen's `libclang`, and
`libnuma` for NUMA-aware pinning (without it the runs log a harmless
`numa_num_possible_nodes` warning) — on Debian/Ubuntu:

```shell
apt update && apt install build-essential ca-certificates cmake curl pkg-config unzip \
    libibverbs-dev libnl-3-dev libnl-route-3-dev libclang-dev libnuma-dev
```

Then install [uv](https://docs.astral.sh/uv/getting-started/installation/), a
Rust toolchain (<https://rustup.rs>) — this workspace uses the 2024 edition,
which needs rustc **1.85 or newer**; your distribution's packaged `rustc`/
`cargo` is very likely too old (Ubuntu 24.04 ships 1.75) and will fail with a
confusing `feature edition2024 is required` error rather than a clear version
message, so install via rustup rather than `apt` — and `protoc` >= 3.15 (the
protos use proto3 `optional`, which older `protoc` rejects — Ubuntu 22.04's
`protobuf-compiler` is 3.12, too old). Install the official release binary
once to a system path:

```shell
# pick linux-x86_64 or linux-aarch_64 to match `uname -m`
curl -fsSL -o /tmp/protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protoc-28.3-linux-aarch_64.zip
unzip -o /tmp/protoc.zip -d /usr/local 'bin/*' 'include/*'
protoc --version   # libprotoc 28.3
```

On a fresh container image the `apt update` is load-bearing (a stale package
index resolves `build-essential` against libc versions the archive no longer
serves), and some CUDA base images additionally hold core libraries at the
image's versions — if the install still reports unresolvable `gcc-12-base` /
`libstdc++6` dependencies, unpin them first: `apt-mark showhold`, then
`apt-mark unhold <the listed packages>`.

> **GPU driver vs. bundled compat layer.** The engine and JAX use the host's
> NVIDIA driver. Some `nvidia/cuda` base images ship a *forward-compatibility*
> driver layer (`/usr/local/cuda-*/compat`, wired into `ldconfig`) built for a
> newer driver than the host runs; mixing the two segfaults inside the CUDA
> PTX JIT with no Python traceback. If `python -c "import jax;
> print(jax.devices())"` works but real model code dies with SIGSEGV in
> `libnvidia-ptxjitcompiler`, disable the compat layer (remove or rename the
> `/etc/ld.so.conf.d/*compat*.conf` entry and rerun `ldconfig`) so the host
> driver's own libraries resolve first.

Then:

```shell
uv sync --extra engine
export PYTHONPATH=$PWD
```

`--extra engine` builds the real Rust serving engine (~1 minute); training
and serving both import it. (The engine links `libibverbs`, the production
embedding transport; its `ibverbs-sys` build generates bindings against the
system verbs headers, which is what pulls in `libnl` and `libclang`.)

### Generate data, train, serve

There are no artifacts to download — the synthetic world replaces them:

```shell
# 1. Synthetic world: SID + multimodal + post-creation snapshots and the
#    retrieval candidate corpus, then a training dump of user sessions.
uv run python reference/world_snapshots.py --out ./synth_index --seed 20260721
export PHOENIX_INDEX_BASE=./synth_index
uv run python reference/dump_gen.py --out ./synth_dump --seed 20260721 \
  --num-rows 12288 --partitions 4 --rows-per-file 1024 \
  --sid ./synth_index/sid_snapshot/post_sid_v5_256x6.parquet --self-check

# 2. Train the nano RANKING model, then the nano RETRIEVAL model, on the
#    same dump (every retrieval checkpoint embeds the candidate corpus as
#    its serving index).
uv run python reference/train_synth.py --data ./synth_dump --steps 6 --out "$PWD/checkpoints"
uv run python reference/train_synth.py --config xrecsys_two_tower_nano_offline_kafka_dump \
  --data ./synth_dump --steps 6 --out "$PWD/checkpoints"

# 3. Serve and drive the integrated retrieve → rank loop over real gRPC.
#    retrieve_then_rank.py is only the CLIENT: the three servers (SID lookup,
#    retrieval, ranking) must be up first — QUICKSTART.md §5 has the exact
#    launch lines to start them, §4 the single-server variant. Run this after.
uv run python reference/retrieve_then_rank.py --data ./synth_dump \
  --sessions 3 --topk 16 --retrieval-port 9990 --ranking-port 9988
```

The loop sends each synthetic user's real history to the retrieval server,
takes the top-K posts from the checkpoint's index, and has the ranking
server score exactly those posts for the same user action sequence — the
same contract production composes, over the same two gRPC services.

---

## Model Architecture Configs

The production configs and their single-GPU nano twins, as shipped in
`xrex/configs/` (`xrecsys.py` for ranking, `xrecsys_two_tower.py` for
retrieval):

| Parameter | Ranking (prod) | Ranking (nano) | Retrieval (prod) | Retrieval (nano) |
|---|---|---|---|---|
| Embedding dimension | 2560 | 512 | 1024 | 512 |
| Transformer layers | 8 | 4 | 8 | 4 |
| Query / KV heads (GQA) | 20 / 4 | 4 / 2 | 16 / 4 | 4 / 2 |
| Attention key size | 128 | 128 | 128 | 128 |
| Embedding-table width | 1024 | 128 | 1024 | 512 |
| FFN widening factor | 2 | 2 | 2 | 2 |
| History sequence length | 1022 | 1022 | 1023 | 1022 |
| Candidate sequence length | 64 | 64 | 64 | 64 |
| Sequence packing | yes (varlen attention) | yes (varlen attention) | yes (varlen attention) | no (dense attention) |
| User / Item / Author vocab | 100M / 100M / 30M | 100k / 100k / 30k | — / 100M (hash) / 30M | — / 100k (hash) / 30k |
| IP-address vocab | 10M | 10k | — | — |
| Hashes per entity | 2 | 2 | 2 | 2 |
| Semantic IDs | 6 × 256 (input feature) | 6 × 256 (input feature) | 6 × 256 (candidate identity) | 6 × 256 (candidate identity) |
| Multimodal post embedding | off (`home_direct_packed` and `xrecsys_seqpack`) | — | — | — |
| SID cross-attention | no | no | yes | yes |
| Discrete action taxonomy | 64 | 64 | 64 (positives: favorite) | 64 (positives: favorite) |
| Continuous-action heads (dwell) | 8 slots | 8 slots | — (dwell input on combined only) | — |
| Candidate index (`max_posts`) | — | — | 10.24M (28.67M combined) | 65,536 |
| Global negatives / example | — | — | 64 | 64 |
| Per-device batch | 512 (GB300) / 256 (H100) | 64 | 480 (768 combined-GB300) | 64 |

The nano twins keep production's losses, checkpoint format and serving
contract, and `emb_size=512` is the μP base width — the transformer trunk's
width-dependent LR/scale multipliers are exactly 1 there. The ranking nano
exercises the same input code paths as its production parent
(`home_direct_packed`), feature prep included — the multimodal-embedding
input is off in both, and on every registered ranking config, as the table
shows;
the retrieval nano uses the flagship's `enable_linear_proj` candidate
combine (a small concat-then-MLP) and trains unpacked (dense attention), as
the table shows.

### Verifying an install

```shell
uv run python xrex/inference/oss_bench/bench.py --smoke --service_type ranking
```

boots the real serving stack with random weights and passes once the server
is up (with the built engine the gRPC port accepts right after warmup and
bench sends one synthetic request; on an install without the engine the
`Model warm up finished` log line is the fallback success signal) — no
checkpoint or data needed.

---

## License

This code is licensed under the Apache License 2.0 — see the `LICENSE` file
at the repository root. Third-party notices for vendored code are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
