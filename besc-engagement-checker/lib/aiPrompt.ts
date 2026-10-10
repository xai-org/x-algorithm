import { buildAlgorithmBrief, buildSituationBrief } from "./algorithmContext";
import type { AnalyzeRequest, GenerateRequest } from "./types";

// Both prompts open with the same algorithm brief, assembled from the real
// constants in scoring.ts, so a model is reasoning about the actual mechanics
// rather than being handed a bare weight list and told to "make it engaging".
// That's the difference between a rewrite that tacks on "What do you think?"
// and one that knows a copy-link share outweighs a like 40:1 and writes
// something worth sending to a friend.

export function buildRewritePrompt(
  text: string,
  numVariants: number,
  charLimit: number,
  req?: AnalyzeRequest
): string {
  const situation = req ? `\n\n${buildSituationBrief(req, charLimit)}` : "";
  return `${buildAlgorithmBrief()}${situation}

YOUR TASK
Rewrite the post below so it scores higher against the system above, keeping the author's voice and every factual claim exactly as written. Preserve meaning precisely: never upgrade a pending thing into a finished one, never add specifics that aren't already there, never invent facts, names, numbers or links.

Post:
"""
${text}
"""

Write ${numVariants} tighter rewrites of this SAME post. Lead with the point rather than a wind-up, cut filler and passive voice, and end each with a reply hook that could only belong to this post. Make the variants genuinely different approaches, not reworded twins.
Output only lines starting "VARIANT: ", nothing else.`;
}

export function buildGeneratePrompt(
  context: string,
  numVariants: number,
  charLimit: number,
  req?: GenerateRequest
): string {
  const situation = req ? `\n\n${buildSituationBrief(req, charLimit)}` : "";
  return `${buildAlgorithmBrief()}${situation}

YOUR TASK
Write original X posts from the notes below, engineered to score well against the system above.

CRITICAL — the notes are the only source of truth. Use only the facts, names and numbers that appear in them. Never invent specifics (stats, dates, names, outcomes) to sound concrete; if the notes are vague on a detail, write around it. Never write a URL or link, not even a placeholder like "https://" or "[link]" — links are attached separately, outside the post text.

Write like a person with something specific to say, not like marketing copy. No stock AI phrasing, no hype, no emoji padding.

Notes from the author (a rough idea, not finished text):
"""
${context}
"""

Write ${numVariants} complete posts, each a standalone option taking a genuinely different angle on the same material — a different hook, a different opening move, a different question — not reworded versions of one another.
Output only lines starting "VARIANT: ", nothing else.`;
}

// A flat output-token cap doesn't scale: English runs roughly 4 chars/token,
// so a fixed 250-token budget is already tight for 2-3 short variants and
// can silently truncate the last one — and for a verified account's much
// higher char limit (up to 4,000) it isn't remotely enough for even a
// single full-length variant. Scales with both charLimit and numVariants,
// floored at the old constant so nothing regresses, ceilinged so a verified
// draft times many variants can't blow generation time up unboundedly
// (especially on CPU/Ollama).
export function estimateMaxOutputTokens(charLimit: number, numVariants: number): number {
  const perVariant = Math.ceil(charLimit * 0.35) + 25; // + "VARIANT: " prefix/newline overhead
  return Math.min(2000, Math.max(250, perVariant * numVariants));
}

// A generation that runs out of output tokens mid-URL leaves a fragment like
// "...engagement: https://" — and the deterministic optimizer will then
// happily append a reply hook to it, producing a finished-looking post with
// a dead link buried in the middle (observed in production). A trailing URL
// with no real dotted domain means the model got cut off, not that it wrote
// a short link, so drop the whole variant instead of surfacing the fragment.
function endsWithIncompleteUrl(text: string): boolean {
  const tail = text.match(/(?:https?:\/\/|www\.)\S*$/i);
  if (!tail) return false;
  return !/(?:https?:\/\/|www\.)[^\s/]*[a-z0-9-]\.[a-z]{2,}/i.test(tail[0]);
}

export function parseVariants(raw: string, max: number): string[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^variant:?\s*/i.test(line))
    .map((line) => line.replace(/^variant:?\s*/i, "").trim())
    .filter(Boolean)
    .filter((line) => !endsWithIncompleteUrl(line))
    .slice(0, max);
}
