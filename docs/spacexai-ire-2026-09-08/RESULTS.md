# e2e_smoke_f5 RESULTS — 2026-09-08
**Command:** `python3 smoke_controlled_reexposure.py`  
**Outcome:** **22 PASS / 0 FAIL**

## EXP+ (niche re-exposure)
| Check | Result |
|-------|--------|
| Current: first serve original | PASS |
| Current: 2nd request burned (related/source) | PASS |
| Current: same-slate 3 RTs → 1 | PASS |
| F5: 2nd/3rd IN nudge within K=3 | PASS |
| F5: 4th blocked at K | PASS |
| Attribution = original source_id | PASS |

## EXP− (snowball)
| Check | Result |
|-------|--------|
| Current 20 RTs → 1 | PASS |
| Current 20 quotes survive RT-dedup | PASS (risk → need quote cap in F5) |
| F5 ≤1/slate + K=3 across requests | PASS |
| F5 rejects OON RT path | PASS |
| Following prefer-native / For You RT-first gap (F1) | PASS |

## Viability
F5 **not in tree** — implement as FS A/B on top of existing filters. Building blocks all present @ `902a06f`.
