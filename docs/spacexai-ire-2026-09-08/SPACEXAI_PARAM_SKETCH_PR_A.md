# Param sketch — PR-A FollowAuthor / ProfileClick rebalance

**Mirrors:** `docs/BIDIRECTIONAL_BOOST_CHANGE.md`  
**Repo:** https://github.com/xai-org/x-algorithm @ `902a06f`  
**Primary file:** `home-mixer/params/param.rs`  
**Scorer (unchanged):** `home-mixer/scorers/ranking_scorer.rs` apply terms ~418–447  
**No new hydrator.** FS keys already exist; weights already consumed.

---

## Current defaults (`home-mixer/params/param.rs`)

Verified @ `902a06f`:

```rust
param!(
    ProfileClickWeight,
    f64,
    "rust_home_mixer_profile_click_weight",
    0.0
);
param!(
    ShareViaCopyLinkWeight,
    f64,
    "rust_home_mixer_share_via_copy_link_weight",
    20.0
);
param!(
    FollowAuthorWeight,
    f64,
    "rust_home_mixer_follow_author_weight",
    4.0
);
```

FS keys (already wired):

- `rust_home_mixer_follow_author_weight`
- `rust_home_mixer_profile_click_weight`
- `rust_home_mixer_share_via_copy_link_weight`

---

## Proposed A/B arms (Feature Store)

| Arm | FollowAuthor | ProfileClick | CopyLink | Intent |
|-----|-------------:|-------------:|---------:|--------|
| **control** | 4 | 0 | 20 | ship today |
| **T1** | **12** | **1.0** | 20 | follow ~3×; profile on |
| **T2** | **20** | **2.0** | 20 | follow weight = CopyLink (equal-P) |
| **T3** | 20 | 2.0 | **10** | rebalance share vs follow |

**Assignment (bidirectional July 2026 spirit):** small % on T1/T2/T3; majority control until early read. Prefer T1→T2 before enabling T3 (T3 also moves share volume).

---

## Ship path (recommended)

1. **FS A/B only** — do **not** commit default bumps in `param.rs` until a winning arm.  
2. After win, document default change the way bidirectional did (reply boost → 15/20).  

### Illustrative post-win default bump (not the A/B vehicle)

```diff
diff --git a/home-mixer/params/param.rs b/home-mixer/params/param.rs
--- a/home-mixer/params/param.rs
+++ b/home-mixer/params/param.rs
@@
 param!(
     ProfileClickWeight,
     f64,
     "rust_home_mixer_profile_click_weight",
-    0.0
+    1.0   // replace with winning arm; FS override during A/B
 );
@@
 param!(
     ShareViaCopyLinkWeight,
     f64,
     "rust_home_mixer_share_via_copy_link_weight",
-    20.0
+    20.0  // T3 only would set 10.0; keep 20 until T3 wins
 );
@@
 param!(
     FollowAuthorWeight,
     f64,
     "rust_home_mixer_follow_author_weight",
-    4.0
+    12.0  // or 20.0 if T2/T3 wins
 );
```

---

## Scorer surface (unchanged)

`ranking_scorer.rs` already includes:

- `ProfileClickWeight * profile_click_score` (~426)  
- `ShareViaCopyLinkWeight * share_via_copy_link_score` (~430–433)  
- `FollowAuthorWeight * follow_author_score` (~447)  

PR-A adds **no** gates, filters, or hydrators. Follow remains one term in Σ, not a hard gate (that is PR-B, deferred).

---

## Guardrails / metrics

**Primary**

- follow/imp (OON vs IN split if available)  
- profile_click → follow (when funnel exists)

**Safety**

- report / mute / block rate  
- copy-link / share volume (especially T3)

**Hypothesis floors** (creator empirical — label as such, not global maxima)

- follow/imp Batch B pocket **0.0644%**

---

## Companion

- Cover: `SPACEXAI_COVER_TRANCHE1.md`  
- Telemetry: `SPACEXAI_PR_D_TELEMETRY_NOTE.md`

## Out of scope this PR

- OON low-P(follow) discount (PR-B)  
- Author yield dampener (PR-C)  
- Verified/OCR ranking term (PR-E)
