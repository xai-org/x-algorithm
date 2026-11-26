# reference/ — synthetic data, reference providers, and the readable training path

Production Phoenix is fed by infrastructure that cannot ship: a Kafka stream of
training sessions, an MM-embedding service, a Semantic-ID (SID) service, and the
checkpoints they produce. This directory is the public replacement for all of
it — one seeded synthetic world, generators that emit every artifact the
shipped serving/training stack reads, reference implementations of the two
provider services, and the readable single-device training step. Everything
here is runnable end to end with zero production data.

⚠ **REFERENCE DATA, NOT PRODUCTION.** The synthetic artifacts share the exact
schemas, layouts, and wire contracts of their production counterparts, but the
*values* are deterministic stand-ins. The directory map below states, for
each file, whether it generates stand-in data or is an algorithm port / real
mirrored source — the map is the authority (published sources ship without
comments or docstrings, so no in-file banner survives to say it there).

## Directory map

**The synthetic world and its emitters**

| File | What it is |
| --- | --- |
| `world.py` | The seeded world every generator draws from: topics, authors, users, posts, and a planted engagement model. One seed → one world → mutually consistent artifacts. |
| `dump_gen.py` | Offline training-dump generator — the Kafka replacement. Writes user-session snapshots as parquet in the exact layout + schema the shipped `aggregated_kafka` reader consumes. |
| `world_snapshots.py` | Emits the three provider snapshots (MM, SID, post-creation) from the same world, all keyed on the same post ids. Its output directory is a ready `PHOENIX_INDEX_BASE`. |
| `oss_recsys_synth.py` | The deterministic value generators underneath the synthetic path: random-Fourier-feature MM embeddings and random-codebook RQ codes, in both server-wire and model-buffer conventions. |

**MM embeddings (post → 1024-d vector)**

| File | What it is |
| --- | --- |
| `mm_encoder.py` | The faithful v5 embedding algorithm: renderer + ChatML wrap + MRL truncate/L2-norm, ported verbatim. Ships code only — the reader supplies the stock public `Qwen/Qwen3-VL-Embedding-8B` weights (or an SGLang endpoint). |
| `mm_snapshot_gen.py` | Writes the MM-embedding snapshot parquet. Default: synthetic stand-in vectors. Opt-in: real v5 vectors via `embedder=mm_encoder.embed_post`. |
| `example_data/` | Tiny self-contained post fixtures (text / image / quote / video) for exercising the renderer without network or real media. |

**Semantic IDs (post → 6-level RQ code)**

| File | What it is |
| --- | --- |
| `sid_codebook.py` | The RQ-KMeans / RQ-VAE codebook **trainer** (JAX), mirrored from the internal pipeline source. CLI: `train` / `evaluate` / VAE variants. |
| `sid_assign.py` | The SID **assigner**, mirrored from the internal pipeline source: MLP encode → residual-quantize against a trained codebook. |
| `sid_io.py` | Shared IO for the two files above (codebook `.npz` format, MLP forward), mirrored from the internal pipeline source. |
| `sid_snapshot_gen.py` | Writes the post-SID snapshot parquet. Default: synthetic random-codebook codes. Opt-in: codes from a codebook you trained with `sid_codebook.py`. |
| `sid_index_server.py` | **Real** SID lookup server: serves codes from a snapshot parquet over the production `SidLookupService` gRPC contract. |
| `sid_mock_server.py` | Stand-alone mock SID server: same wire contract, codes generated on the fly (no snapshot needed). |
| `_sid_proto/` | Generated gRPC stubs for `sid_lookup.proto` (source: `crates/serving/xai-recsys-sid-proto/proto/`). |

**Training, checkpoints, and serving**

| File | What it is |
| --- | --- |
| `train_step.py` | The canonical single-device training step — the released composition (dense optimizer + sparse rowwise-AdaGrad; this reference composes the AdamW arm, while the ranking flagship/nano configs train the shipped Muon recipe) with the sharding infrastructure removed. See [`TRAINING.md`](../TRAINING.md). |
| `train_synth.py` | The public launcher: points the shipped trainer config at a `dump_gen.py` dump and trains the nano model end to end. |
| `repack_checkpoint.py` | Repacks a trained checkpoint into a publishable artifact: keeps load/infer tensors, drops optimizer state, scrubs internal metadata, regenerates checksums. |
| `retrieve_then_rank.py` | The QUICKSTART §5 driver: sends real dump sessions through the two live servers — `RetrieveTopKCandidates` on retrieval, then `PredictNextActions` on ranking — over the production gRPC contract. |

Provenance, in one line each: `sid_codebook.py`, `sid_assign.py`, and `sid_io.py`
are the real internal SID pipeline source, mirrored at export time with only
infrastructure stripped; `train_step.py` and `mm_encoder.py` are faithful ports
of the production algorithm; everything else is reference tooling authored for
this release.

## How the pieces relate

```
                       world.py  (one seeded world)
                          │
             ┌────────────┴──────────────┐
             ▼                           ▼
        dump_gen.py ◄─── --sid ──  world_snapshots.py
             │          (SID codes)     │
             ▼                           ▼
       offline dump              PHOENIX_INDEX_BASE/
        partition=*/...            mm_snapshot/post_mm_v5.parquet
        .valid_batches.json        sid_snapshot/post_sid_v5_256x6.parquet
        world/*.parquet            sid_snapshot/codebook_v5_256x6.npz
             │                     post_creation_snapshots/post_creation_1day.parquet
             ▼                           │
      train_synth.py                     ├──► serving stack (reads the snapshots)
      (train_step.py inside)             └──► sid_index_server.py --parquet ...
             │                                 ▲ gRPC (SidLookupService)
             ▼                                 │
        checkpoint ──► repack_checkpoint.py    │
             │                                 │
             ▼                                 │
        inference (loads checkpoint, hydrates SIDs from the live server)
```

`world_snapshots.py` composes the other generators — MM vectors via
`mm_snapshot_gen.write_mm_snapshot` and SID codes via a codebook trained by
`sid_codebook.py` + assigned by `sid_assign.py` — so its three artifacts
describe the same posts and the SID codes really quantize the MM vectors next
to them. `dump_gen.py --sid` then reads those same codes into the dump, which is
why the snapshots are generated first: the model trains on the semantic IDs the
SID server will later serve for the same posts.

## The three provider stories

**Training data.** Production training streams Arrow record batches of
user-session snapshots from Kafka. `dump_gen.py` writes those same batches as
parquet (`partition={p}/<bucket>/batch_{b}.parquet` plus a
`.valid_batches.json` manifest) so the shipped `aggregated_kafka` reader
consumes them unchanged. Sessions are sampled from the world's engagement
model, so the planted structure (topic affinities, author quality) is
learnable — a model trained on the dump produces meaningfully ranked
recommendations, not noise. The dump also carries `world/*.parquet` sidecar
tables (users, posts, authors) so you can decode what the model saw.

**MM embeddings.** Production runs a service that turns each post
(text / images / video) into a 1024-d unit vector ("v5"). Two paths ship:

- *Faithful:* `mm_encoder.py` is the real pipeline — render the post into one
  string with position-aware image-pad tokens, ChatML-wrap with the v5 system
  prompt, encode with the stock public `Qwen/Qwen3-VL-Embedding-8B`, truncate
  4096 → 1024, L2-normalize. Heavy deps are lazy; install via the `mm-encoder`
  extra (see the repo-root `pyproject.toml`).
- *Synthetic:* `oss_recsys_synth.synth_mm_embeddings` maps post id → unit
  vector with a seeded random-Fourier-feature sin map. Same schema, same
  unit-norm contract, zero heavy deps — this is what `world_snapshots.py` and
  CI use.

**Semantic IDs.** Production runs a service that maps each post to a 6-level
residual-quantization code (each level in `[0, 256)`) computed from its MM
embedding. What ships is the *entire* loop:

- *Train:* `sid_codebook.py train --training-data emb.npy --codebook-out cb.npz`
  (real mirrored trainer, RQ-KMeans or RQ-VAE).
- *Assign:* `sid_assign.py` encodes embeddings against the trained codebook.
- *Snapshot:* `sid_snapshot_gen.write_sid_snapshot` writes `post_id → post_sid`
  parquet — synthetic codes by default, trained-codebook codes via `codes=`.
- *Train on them:* `dump_gen.py --sid <snapshot>` copies those codes into the
  dump's `semanticIdSeq` column. Without it the column is absent, and because a
  SID-enabled config sizes the buffer from `sid_num_levels` alone, training
  reads every post as "no SID" — no error, and no gradient to the SID tables.
- *Serve:* `sid_index_server.py --parquet <snapshot>` answers the production
  `SidLookupService` gRPC contract from that parquet;
  `sid_mock_server.py` answers it with on-the-fly synthetic codes. The
  serving engine's `PySemanticIdClient` speaks to either, unchanged.

Wire convention note: servers speak 0-indexed codes (`[0, 256)`); the model's
input buffer is 1-indexed `uint16` (`0` = missing). `oss_recsys_synth.py`
implements both (`sid_codes_for_posts` vs `synth_post_sids`) and documents the
offset.

## End-to-end synthetic quickstart

From the repository root (after `uv sync --extra engine` — step 2 trains
through the shipped trainer, which imports the Rust engine's Python module;
see [`QUICKSTART.md`](../QUICKSTART.md)):

```bash
# 1. One seeded world -> provider snapshots, then the training dump.
#    Snapshots come first: the dump reuses their assigned SID codes, so both
#    describe each post the same way.
python reference/world_snapshots.py --out ./synth_index --seed 20260721 --self-check
export PHOENIX_INDEX_BASE=./synth_index
python reference/dump_gen.py  --out ./synth_dump  --seed 20260721 \
  --num-rows 12288 --partitions 4 --rows-per-file 1024 \
  --sid ./synth_index/sid_snapshot/post_sid_v5_256x6.parquet --self-check

# Peek at a few decoded sessions (human-readable, uses the world's post text).
python reference/dump_gen.py --out ./preview_dump --seed 20260721 --preview 3

# 2. Train the nano ranking model on the dump (writes ./checkpoints/...).
python reference/train_synth.py --data ./synth_dump --steps 500 --out ./checkpoints

# 3. Serve SIDs from the snapshot the world emitted (separate terminal).
python reference/sid_index_server.py \
  --parquet ./synth_index/sid_snapshot/post_sid_v5_256x6.parquet --port 50061
```

[`TRAINING.md`](../TRAINING.md) continues from step 2: loading the checkpoint
back and running real inference on it, plus the algorithm-by-component map.
There is no pre-trained checkpoint to download; training the nano on this
dump recipe takes minutes on one GPU.

Every module here is also directly runnable as its own self-check: tools parse
`--help` / `--self-check`, and libraries (`world.py`, `mm_encoder.py`,
`mm_snapshot_gen.py`, `sid_snapshot_gen.py`, `oss_recsys_synth.py`) run their
assertions when executed bare, e.g. `python reference/world.py`.

## Determinism

Same seed → same world → byte-identical artifacts, across runs and machines:
generators use explicitly seeded NumPy generators, fixed row order, and fixed
parquet writer settings. The one caveat: `world_snapshots.py` trains its SID
codebook on GPU with JAX, and k-means reductions can wobble in the last float
bit run-to-run — the `codebook_v5_256x6.npz` centroids may differ at ~1e-8
while every parquet artifact, including the assigned SID codes, stays
byte-identical. Default seed everywhere is `20260721`; pass `--seed` to make a
different world.
