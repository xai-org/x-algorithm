# Mixed SpaceXAI request — study + solutions + implementable F5
**Account evidence:** @falconortiz Studio + API 2026-09-08  
**Code:** https://github.com/xai-org/x-algorithm @ `902a06f`  
**Sender when approved:** GitHub **falconortiz** · SpaceXAI owns merge  
**Pattern:** `docs/BIDIRECTIONAL_BOOST_CHANGE.md` (FS → A/B → ship)  
**Status:** HOLD until Falcon go  
**Eng:** Head of Engineering (IRE) — hardened packaging

---

## 1. POV — is a mixed PR the right vehicle?

**Yes — as one coherent cover / program with layered diffs, not one mega-commit.**

| Layer | Role | Land now? |
|-------|------|-----------|
| **Study (PR-D)** | Serve mix original vs RT vs quote; re-serve≥2×; negatives if exposed | **Yes** — instrumentation |
| **Solutions (PR-A + PR-F1)** | FollowAuthor/ProfileClick A/B; prefer-original For You dedup | **Yes** — FS A/B on |
| **Structure (PR-F5)** | ControlledSourceReexposure — full FS + filter sketch | **Yes code / default off** — treatment arms only after D baseline |

**Not in this ask as substitutes:** open F2b (burn off) or F4 (post-48h age prolong). F5 is the controlled alternative.  
**Not pertinent:** study-only with no F5 sketch · F5-only without A/D/F1 · enabling F5 treatment before serve-mix baseline.

**Projection:** Land D + A/F1 A/B + F5 params/filter **flag off** in one program. Keep F5 treatment off for ~1–2 experiment cycles until serve-mix baseline exists, then turn K=2/3 arms. Same “creator reach equilibrium” story — three layers, one cover.

---

## 2. Problem statement (metrics-backed)

1. **Conversion:** cohort follow/imp **0.00636%** vs Batch B **0.0644%** (~10×); FollowAuthor=4 vs CopyLink=20; ProfileClick=0.  
2. **OCR:** 180/500 VF · 49.1K/500K HT; rolling floor ~**5,556** verified HT/day at η=3.78% or NEVER.  
3. **Hard TL:** Age 48h · RT XOR original · source burn after serve · quotes bypass RT-dedup.  
4. **Proxy:** high-amp strata follow/imp **0.0678%** vs low-amp megaviral **0.0053%**; serve % RT-vs-original **unmeasurable** today.  
5. **Smoke:** current burn blocks niche re-see; F5 restores K=2–3 IN nudges without 20-RT flood (`e2e_smoke_f5/` 22/22 PASS sim).

Hypothesis floors only — not global SLOs.

---

## 3. F5 structure (implementable, default off)

### 3.1 Placement
Policy **override on seen/served source burn** for a source_id when F5 allows re-serve — **not** an AgeFilter rewrite and **not** OON RT enable.

Order of intent:
1. Same-slate ≤1 (existing RT dedup + **F1** prefer-original).  
2. Seen/served burn as today when F5 off (K≡1).  
3. When F5 on: re-admit source for a later request if K/gap/IN rules pass. Prefer **original** candidate if present; else IN amplifier card of that source.

### 3.2 Feature Store params (proposed)

| Param | Type | Default (ship) | Treatment arms |
|-------|------|----------------|----------------|
| `EnableControlledSourceReexposure` | bool | **false** | true |
| `ControlledSourceReexposureK` | u32 | **1** (≡ today’s no re-see) | **2**, **3** |
| `ControlledSourceReexposureGapRequests` | u32 | 1 | 1–3 |
| `ControlledSourceReexposureInNetworkOnly` | bool | **true** | keep true for first ship |
| `ControlledSourceReexposureIncludeQuotes` | bool | **true** | **fixed true** — no false arm (HoE) |

### 3.3 `source_key` (quotes required)

```
if retweeted_tweet_id.is_some() -> retweeted_tweet_id
else if quoted_tweet_id.is_some() -> quoted_tweet_id   # REQUIRED — smoke EXP−
else -> tweet_id
```

Quotes and RTs of the same original **share one cap bucket**.

### 3.4 Acceptance
- Same request: ≤1 card per source (original / RT / quote).  
- Across requests: ≤K serves per source / viewer / window.  
- OON RT never used as F5 nudge.  
- Attribution / logging: `get_original_tweet_id` + card_type ∈ {original, rt, quote}.  
- No material ↑ report / mute / block vs control (needs PR-D).

### 3.5 Files (sketch — SpaceXAI owns merge)
- Filter / policy near `previously_seen_*` / `previously_served_*` (re-admission)  
- Params: `home-mixer/params/param.rs`  
- Counters: Under the Hood / side effects (PR-D)  
- Tests: mirror `e2e_smoke_f5/` EXP±  

Detail: `SPACEXAI_F5_IMPLEMENTATION.md`

---

## 4. Mixed PR table of contents (external)
1. Cover one-pager (soft misalignment + hard rules + F5 north star)  
2. **PR-D** study / telemetry  
3. **PR-A** param sketch  
4. **PR-F1** prefer-original (Following parity)  
5. **PR-F5** implementation note — flag default **off**  
6. Explicit non-asks: open F2b / F4 / E · no OCR invent · no merge claim · no VF SpamHighRecall loosen  

Companions already drafted: `SPACEXAI_COVER_TRANCHE1.md`, `SPACEXAI_PARAM_SKETCH_PR_A.md`, `SPACEXAI_PR_D_TELEMETRY_NOTE.md`, `SPACEXAI_RT_QUOTE_REQUEST.md`

---

## 5. Risk / blast radius
| Risk | Mitigation |
|------|------------|
| Quote wall | Same source_key as RT; IncludeQuotes fixed true |
| Snowball | K + gap + ≤1/slate |
| Blind A/B | F5 treatment off until D serve-mix baseline |
| Safety VF | Do not loosen SpamHighRecall |
| Creator floors as SLOs | Label empirical only |

## 6. Success projection (hypothesis, not SLO)
If η moves toward ~10% and follow/imp toward Batch B on non-viral mix, OCR HT path becomes more reachable; F5 targets **quality re-exposure**, not raw viral. Celebrity strata stay excluded from baseline reporting.
