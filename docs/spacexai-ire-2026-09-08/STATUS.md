# SpaceXAI pack — merge-ready STATUS

**Base commit:** `902a06fd616ed815f660e5546d16d492fa1ca825` (`x-algorithm`)  
**Date:** 2026-09-08  
**External send:** **HOLD** (per `READY_FOR_GO.md` — Falcon must say go)  
**Do not push / open PRs from this box workstream.**

## What landed (tree patches under `/workspace/x-algorithm`)

| Pack | Status | Notes |
|------|--------|-------|
| **PR-A** | Comment-only | FS keys already present (Follow=4, Profile=0, CopyLink=20). Documented T1/T2/T3 arms near params; **defaults unchanged**. Pattern cites `docs/BIDIRECTIONAL_BOOST_CHANGE.md`. |
| **PR-F1** | Code + params + tests | `EnablePreferOriginalRetweetDedup` default **false**. Flag off = first-wins; flag on = Following prefer-original. |
| **PR-F5** | Code + params + tests | New `ControlledSourceReexposureFilter`; PreviouslyServed defers when Enable && K>1. Defaults: Enable=false, K=1, gap=0, InNetworkOnly=true. Quotes share `source_key` with RTs. |
| **PR-D** | Lightweight hook | `classify_serve_card` / `record_serve_card_type` (+ unit tests). Full Studio telemetry = product-side. |
| **Out** | — | F2b, F4, E, SpamHighRecall, AgeFilter, OON RT enable untouched. |

## How to test

### Standalone (green on any machine — REQUIRED)

```bash
cd /workspace/x-algo-cache/merge-ready/spacexai-pack-tests
cargo test
```

**Result (this box):** **22 passed** (5 lib unit + 17 e2e integration), 0 failed.  
Zero private deps. README explains monorepo gap.

### Full home-mixer

Needs SpaceXAI private monorepo (`xai_*` crates + Cargo workspace). Public OSS tree has no runnable home-mixer `Cargo.toml`. After merge into monorepo:

```bash
# illustrative — monorepo paths may differ
cargo test -p home-mixer retweet_deduplication
cargo test -p home-mixer controlled_source_reexposure
cargo test -p home-mixer serve_card_classify
```

### Algorithm smoke (Python sim)

```bash
python3 /workspace/x-algo-cache/e2e_smoke_f5/smoke_controlled_reexposure.py
# historically 22/22 PASS
```

## Patch export

`/workspace/x-algo-cache/merge-ready/0001-spacexai-f1-f5-a-d.patch`  
Apply from repo root: `git apply` / `git am` as appropriate.

## Interaction note (F5 ↔ PreviouslyServed)

- Enable=false **or** K≤1 → PreviouslyServed = today (drop on served related ids); F5 filter `enable()` false.  
- Enable && K>1 → PreviouslyServed pass-through; F5 owns K / gap / IN-only / prefer-original / ≤1 per `source_key`.  
- Wired in `phoenix_candidate_pipeline.rs` immediately after PreviouslyServedPostsFilter.  
- AgeFilter / OONRetweetReplyFilter / VF unchanged.

## Blockers

- Cannot compile/run full home-mixer unit tests on this box (missing private crates) — mitigated by standalone crate.  
- F5 request-gap uses `served_history` request indices when present; with gap default **0** the gate is a no-op until treatment arms set gap>0.  
- Serve-count approximation from flat `served_ids` is sufficient for K gating; monorepo may refine with richer history later.
