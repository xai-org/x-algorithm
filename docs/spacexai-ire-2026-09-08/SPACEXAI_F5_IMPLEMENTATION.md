# PR-F5 — ControlledSourceReexposure implementation note
**Status:** Landable with mixed PR as FS + filter sketch · treatment default **off**  
**Depends on:** PR-D counters before enabling K>1 arms  
**Companions:** `SPACEXAI_MIXED_PR_F5.md`, `e2e_smoke_f5/` (22/22 PASS sim), HoE quote-cap harden  
**Repo:** x-algorithm @ `902a06f` · SpaceXAI owns merge

## Algorithm (per viewer request)
```
slate = candidates after hydration
slate = RetweetDedup + prefer-original when F1 on
slate = drop OON RTs (existing OONRetweetReplyFilter — unchanged)
for each candidate:
  source = source_key(candidate)
  if source in served_or_impressed_related:
    if not EnableControlledSourceReexposure: drop   # today
    elif serves[source] >= K: drop
    elif requests_since_last[source] < Gap: drop
    elif InNetworkOnly and amplifier not IN: drop
    else: keep as re-exposure card
  else:
    keep  # first exposure
# Prefer original over amp when both re-admitted for same source
enforce ≤1 card per source in slate
log serve(source, card_type ∈ {original, rt, quote})
```

## source_key (quotes REQUIRED)
```
if retweeted_tweet_id: return retweeted_tweet_id
if quoted_tweet_id:    return quoted_tweet_id
return tweet_id
```
RTs and quotes of the same original share one K bucket. Do not ship a false arm on IncludeQuotes.

## Rollout
1. Land params + logging (D) with **Enable=false** and/or **K=1** (behavior ≡ today)  
2. Shadow: compute would-allow rate from serve mix  
3. After baseline: A/B **K=2** vs **K=3**, small %; InNetworkOnly=true  
4. Guardrails: report / mute / block; duplicate-fatigue proxies  
5. Ship winning K or keep off  

## Out of scope
AgeFilter change · OON RT enable · VF / SpamHighRecall loosen · OCR ranking head · open F2b burn-off · F4 post-48h prolong
