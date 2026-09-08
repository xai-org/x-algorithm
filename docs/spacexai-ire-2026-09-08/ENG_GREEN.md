# ENG GREEN — SpaceXAI merge-ready pack
**Reviewer:** Head of Engineering (IRE) · **2026-09-08**  
**Base:** `902a06f` · **Patch:** `0001-spacexai-f1-f5-a-d.patch`  
**External send:** still **HOLD** until Falcon go

## Verdict
**ENG GREEN** to land / propose. Acceptance criteria met.

| # | Criterion | Result |
|---|-----------|--------|
| 1 | PR-A FS only; defaults 4/0/20 | PASS — comment + T1–T3 arms; defaults unchanged |
| 2 | PR-F1 prefer-original FS-gated default false | PASS — first-wins off; Following parity on |
| 3 | PR-F5 Enable=false / K=1; IN-only; source_key RT→quoted→tweet_id; ≤1/slate; after PreviouslyServed; no Age/OON/VF | PASS |
| 4 | PR-D classify hooks only | PASS — `serve_card_classify.rs` |
| 5 | Out list untouched | PASS |
| 6 | Tests green + EXP± | PASS — standalone **22/22** (re-run by HoE); Python smoke **22/22** |

## Non-blockers (treatment / monorepo follow-ups)
1. **F5 v1 scopes to PreviouslyServed only** — `PreviouslySeen*` still drops impressed related ids. Correct blast radius for landable default-off; treatment arms may need a measured seen-policy later (not open F2b).  
2. **D hooks fire on F5 keep path** — full serve-mix baseline when F5 off still needs product Under-the-Hood / monorepo counter wire.  
3. **OSS cannot run full home-mixer cargo** — private `xai_*`; standalone crate is the required green bar here.

## Go
Falcon go → send from GitHub `falconortiz` with patch + mixed cover. Until then: HOLD.
