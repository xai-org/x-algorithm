# SpaceXAI request — RT/quote slot vs parallel original reach

**Companion to:** Tranche 1 (PR-A FollowAuthor/ProfileClick rebalance + PR-D follow telemetry)  
**Code:** https://github.com/xai-org/x-algorithm @ `902a06f`  
**Pattern:** `docs/BIDIRECTIONAL_BOOST_CHANGE.md` (FS A/B → ship)  
**Sender (when approved):** GitHub **falconortiz** · IRE · SpaceXAI owns merge  
**Status:** HOLD until Falcon greenlights joint send with Tranche 1

---

## Plain language

For You treats a **repost and its original as one content slot** (`RetweetDeduplicationFilter`: key = source tweet id, **first wins**). A celebrity RT can therefore **replace** the original in a viewer’s slate instead of stacking.

Following already prefers the native tweet (`FollowingRetweetDeduplicationFilter`). For You does not.

Quotes are separate posts and can run in parallel, but follow credit goes to the **quoter** (`follow_author` on quote author; `QuotedClickWeight` default **0.05**).

There is **no** published knob that resets remaining reach to the last amplifier’s follower count. The “re-adjust after a smaller RT” feeling is explained by first-wins dedup + weaker IN distribution + optional seen/served related-id burn (`related_post_ids` = tweet + retweeted + reply-to; **not** quoted).

## Ask (narrow)

1. **PR-F1 (code):** Prefer original over RT in For You retweet dedup when both present — parity with Following. FS bool.  
2. **Telemetry (fold into PR-D):** % of serves as original vs RT-of-original for amplified posts; quote follow vs nested-author profile click.  
3. **Optional later (not this send):** seen/served policy so an RT impression does not permanently exclude the original without a fatigue model.

## Success

↑ original `tweet_id` serve share after celebrity RT; follow/imp on original not degraded; no duplicate-fatigue spike.

## Evidence (hypothesis, one creator)

@falconortiz `2076699535296856102` (Elon RT): ~1.1M imp → 24 follows (**0.0022%**); dominated cohort impressions with weak follow yield. Consistent with slot substitution + celebrity IN audience — **not** a counter reset, and **not** a global SLO.

## Out of scope

- Removing OON RT drop entirely  
- OCR/verified ranking  
- QuoteWeight / PR-B/C/E  
- Claiming single-creator rates as global SLOs

---

## Addendum — prolonged HT exposure vs amplifier volume

**Gap:** Creators expect quote/RT volume to validate niche utility and **extend parallel** original Home exposure. Published formula does the opposite pressure:

1. `AgeFilter` hard-caps eligibility at **48h** on `candidate.tweet_id` (`MAX_POST_AGE`). Quote/RT counts do not extend the original. New RT snowflakes can remain eligible after the original ages out.
2. Score uses `QuoteWeight·P(quote)` / `RetweetWeight·P(retweet)` — predicted actions — **not** `f(quote_count)` as a prolongation validator.
3. Served history records `source_tweet_id` with the RT (`ExcludeServedTweetIdsDuration` default 10m) → original can be excluded right after amplifier serve.

**PR-F4 (deferred design note — not in this send):** amplification volume as positive evidence for **keeping** the original eligible/parallel; do not let post-48h distribution collapse solely onto amplifier cards if product wants prolonged original HT. Age/prolong policy is separate from **PR-F1** (same-slate prefer-original). Source burn on RT serve stays under **PR-F2b** (also deferred).
