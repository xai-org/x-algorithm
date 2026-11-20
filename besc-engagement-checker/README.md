# BESC Engagement Checker

Score a draft post before you publish it on X — grounded in the **real, open-sourced
For You algorithm** that lives in this repository, not vibes-based social media advice.

Paste a draft, pick your media type, optionally attach a link, and get:

- A live **BESC Score** (0-100) computed from the same `RankingScorer` formula X uses:
  `Final Score = Σ (weight_i × P(action_i))`, using the actual production weights
  mirrored in [`home-mixer/params/param.rs`](../home-mixer/params/param.rs) (reply = 5.0,
  share-via-copy-link = 20.0, report = -234.0, etc).
- A full **signal breakdown** — every action (like, reply, repost, quote, share, block,
  mute, report...) with its real weight, an estimated probability, and its contribution
  to the final score.
- **Visibility-filtering risk flags** tied to actual rules in
  [`visibility-filtering/rules/`](../visibility-filtering/rules/) and
  [`botmaker-rules/scarecrow/bot/`](../botmaker-rules/scarecrow/bot/) — shortened/redirect
  links, copy-paste boilerplate phrasing, sensitive-media interstitials, and more.
- Actionable, weight-ranked **tips** to raise your score.

## Honesty about what this is

X's real ranking uses **Phoenix**, a learned model that reads a viewer's actual action
history to predict `P(action | post, viewer)`. We don't have that model, your account's
history, or a specific viewer here — so this tool estimates action propensities from
surface-level features of your draft (length, hooks, links, formatting, etc.) and then
blends them using the **real production weights**. The weights and the visibility-filtering
rule citations are accurate to this repo; the probability estimates are a transparent
heuristic proxy for Phoenix, not the model itself. Treat the score as directional
coaching, not a guarantee — see `lib/scoring.ts` for the fully commented, from-scratch
implementation and exact source citations for every number.

## Stack

- Next.js 14 (App Router) + TypeScript
- Tailwind CSS + Framer Motion
- Single Node service — API route (`/api/analyze`) and frontend ship together

## Local development

```bash
npm install
npm run dev
```

## Deploying on Railway

This is a standard Next.js app — Railway's Nixpacks builder detects it automatically.

1. Create a new Railway project from this repo (or this subdirectory, if deploying as
   a monorepo service — set the service's **root directory** to `besc-engagement-checker/`).
2. Railway runs `npm install && npm run build` and starts with `npm run start`, which
   binds to Railway's injected `PORT` automatically (see `package.json`'s `start` script).
3. No environment variables or external services are required for the core scorer —
   everything runs server-side in this one process.

`railway.json` pins the build/start commands explicitly if you want them version-controlled.

## Optional: AI-assisted rewrite candidates

The deterministic scorer/optimizer above always runs and never needs any of this. On top
of it, `/api/optimize` can optionally generate creative rewrite candidates with an LLM —
every candidate is still scored and gated through the same deterministic scorer, so this
only ever *suggests*, never decides (see `app/api/optimize/route.ts`).

Two providers are supported; set env vars for either (or both — Gemini is preferred
when both are configured, since it's hosted and doesn't carry the cold-start/CPU risk
described in `ollama-service/README.md`):

- **Gemini** (hosted, recommended): `GEMINI_API_KEY`, optionally `GEMINI_MODEL`
  (defaults to `gemini-2.5-flash`).
- **Ollama** (self-hosted, free): `OLLAMA_URL`, `OLLAMA_MODEL` — see
  `ollama-service/README.md` for standing up the Railway service.

Neither set → `/api/optimize` returns `aiStatus: "disabled"` and just the deterministic result.

## Optional: post tracking / score calibration

The scorer is a heuristic proxy for Phoenix, not Phoenix itself — so the honest
question is whether a higher BESC Score actually corresponds to more reach *for your
account*. Tracking answers that with real data instead of asking you to take it on faith.

Flow: hit **Track** on a draft → publish it on X → hit **Check for results**. The app
pulls your recent timeline, fuzzy-matches the draft against what you actually posted
(tolerant of pre-post edits and X's t.co link rewriting), then fetches the post's real
views and engagement a couple of hours later, once numbers have had time to accumulate.

Once there are enough measured posts it compares the higher-scoring half of your posts
against the lower-scoring half, and reports which specific optimizer fixes correlated
with better numbers for you.

Set `DATABASE_URL` to a Postgres connection string to enable it (Railway: add a Postgres
service and reference its `DATABASE_URL`). The schema is created automatically on first
use — no migration step. Without it, tracking reports itself as unavailable and every
other feature works exactly as before.

Deliberate design choices worth knowing:

- **Nothing is claimed below 6 measured posts**, and per-fix comparisons need at least 3
  posts on each side. Engagement data is noisy enough that a "pattern" drawn from three
  posts would be invented, not observed.
- **Medians, not averages**, so a single post that happens to go viral can't manufacture
  a trend.
- **Fixes are only attributed when the tracked text is exactly the optimizer's output.**
  Edit after optimizing and the fixes aren't recorded for that post — a wrong attribution
  would corrupt the very data the feature exists to produce.
- Syncing is on-demand and idempotent: it only spends API calls on posts that are matched,
  old enough to measure, and not measured recently.
