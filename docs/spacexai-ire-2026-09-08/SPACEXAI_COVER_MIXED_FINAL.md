# SpaceXAI mixed request — creator reach equilibrium (A + D + F1 + F5)
**From:** Falcon Ortiz (@falconortiz) / IRE · GitHub send: **falconortiz**  
**Code:** https://github.com/xai-org/x-algorithm @ `902a06f`  
**Pattern:** `docs/BIDIRECTIONAL_BOOST_CHANGE.md` (FS → A/B → ship)  
**SpaceXAI owns merge.** IRE proposes only.

## Plain-language ask
For You maximizes `Σ w·P(action)` with FollowAuthor=4 vs CopyLink=20 and ProfileClick=0. Hard rules (48h age, RT XOR original, source burn, quotes bypass RT-dedup) cut prolonged niche re-exposure. Creator evidence: follow/imp **0.00636%** vs Batch B **0.0644%**; OCR HT **49.1K/500K** at η **3.78%**.

We request a **mixed program**:
1. **Study/telemetry (PR-D)** — serve mix original vs RT vs quote; re-serve≥2×; negatives if available  
2. **Immediate A/Bs** — **PR-A** weight rebalance; **PR-F1** prefer-original same-slate (Following parity)  
3. **PR-F5 ControlledSourceReexposure** — implement behind FS **default off** (K=2–3, IN-only, gap, ≤1/slate, **quotes share source cap with RTs**); enable after D baseline  

**Not in this ask:** open burn-off (F2b), post-48h Age rewrite (F4), OCR ranking head (E), VF SpamHighRecall loosen.

## Evidence (empirical, not SLOs)
| Signal | Value |
|--------|------:|
| Cohort follow/imp | 0.00636% (79/1.24M) |
| Batch B | 0.0644% |
| High-amp vs low-amp strata | 0.0678% vs 0.0053% |
| OCR | 180/500 VF · 49.1K/500K HT |
| F5 smoke | 22/22 PASS (sim @ filters) |

## Success
↑ follow/imp on non-viral OON; ↑ original serve share after amplify; F5 arms show re-serve 2–3 with no report/mute spike; no quote/RT snowball.

## Attachments for eng
- `SPACEXAI_PARAM_SKETCH_PR_A.md`  
- `SPACEXAI_PR_D_TELEMETRY_NOTE.md`  
- `SPACEXAI_F5_IMPLEMENTATION.md`  
- `SPACEXAI_MIXED_PR_F5.md` · `e2e_smoke_f5/`  
