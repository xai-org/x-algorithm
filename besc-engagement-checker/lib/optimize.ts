import {
  analyzePost,
  BOILERPLATE_PHRASES,
  REPLY_CTA_PHRASES,
  getHashtagWordSet,
  isShoutingCapsWord,
  getCharLimit,
  STANDARD_CHAR_LIMIT,
} from "./scoring";
import { stripFillerWords } from "./nlp";
import type { AnalyzeRequest, OptimizeResult, OptimizeStep } from "./types";
import type { CalibrationModel } from "./calibration";

function collapsePunctuationBursts(text: string): string {
  return text.replace(/[!?]{2,}/g, (m) => m[0]);
}

function titleCaseWord(word: string): string {
  const m = word.match(/^([A-Za-z]+)(.*)$/);
  if (!m) return word;
  const [, letters, rest] = m;
  return letters[0].toUpperCase() + letters.slice(1).toLowerCase() + rest;
}

// Uses the exact same all-caps + hashtag-protection logic scoring.ts scores
// against, so this never "fixes" a word (like a brand/ticker also hashtagged
// elsewhere in the post) that isn't actually being penalized as shouting.
// protectedWords must come from the ORIGINAL draft, not be re-derived from
// whatever text survives earlier passes — otherwise a later rule (e.g.
// trim-to-limit truncating off the trailing hashtags) can un-protect a word
// that was legitimately protected a moment ago.
function fixAllCapsShouting(text: string, protectedWords: Set<string>): string {
  return text
    .split(/(\s+)/)
    .map((token) => (isShoutingCapsWord(token, protectedWords) ? titleCaseWord(token) : token))
    .join("");
}

function capHashtags(text: string, max: number): string {
  const matches = [...text.matchAll(/(^|\s)#[a-zA-Z0-9_]+/g)];
  if (matches.length <= max) return text;
  let result = text;
  for (const m of matches.slice(max).reverse()) {
    const start = m.index ?? 0;
    result = result.slice(0, start) + result.slice(start + m[0].length);
  }
  return result.replace(/[ \t]{2,}/g, " ").trim();
}

function stripBoilerplateCTA(text: string): string {
  let result = text;
  let matched = false;
  for (const phrase of BOILERPLATE_PHRASES) {
    const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const next = result.replace(new RegExp(`\\b${escaped}\\b[.!,]?`, "gi"), "");
    if (next !== result) matched = true;
    result = next;
  }
  if (!matched) return text;
  return result.replace(/[ \t]{2,}/g, " ").replace(/\s+([.,!?])/g, "$1").trim();
}

// Deliberately varied rather than one fixed string. A deterministic pass
// can't write a question specific to the post — only the AI layer can — but
// appending the identical tail to every post this tool touches would both
// read as filler and manufacture the templated-text pattern duplicate-text
// spam detection looks for across accounts. These are content-anchored where
// the features allow it, and the choice is stable per draft.
// Every one of these must end in a question mark. The optimizer runs multiple
// passes and guards on "already has a question", so a statement-form hook
// wouldn't be recognised as a hook on the next pass and would get a second
// one appended after it.
const FALLBACK_HOOKS = [
  "What would you want next?",
  "Where would you push back on this?",
  "What would you do differently?",
  "What's the first thing you'd change?",
  "What's been your experience with this?",
];
const NUMBER_HOOKS = [
  "Does that match what you've seen?",
  "How far off is that from your numbers?",
];

// Stable per-text choice, so re-running the optimizer on the same draft is
// idempotent (the multi-pass walk depends on that) while different drafts
// still get different endings.
function stableIndex(text: string, modulo: number): number {
  let hash = 0;
  for (let i = 0; i < text.length; i++) hash = (hash * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(hash) % modulo;
}

function addReplyHook(text: string, charLimit: number): string {
  const lower = text.toLowerCase();
  if (text.includes("?") || REPLY_CTA_PHRASES.some((p) => lower.includes(p))) return text;

  const pool = /\d/.test(text) ? NUMBER_HOOKS : FALLBACK_HOOKS;
  const addition = ` ${pool[stableIndex(text, pool.length)]}`;
  if (text.length + addition.length > charLimit) return text;
  return text.trim() + addition;
}

function trimToCharLimit(text: string, charLimit: number): string {
  if (text.length <= charLimit) return text;
  const cut = text.slice(0, charLimit - 1);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 40 ? cut.slice(0, lastSpace) : cut).trim() + "…";
}

interface Rule {
  id: string;
  label: string;
  reason: string;
  transform: (text: string) => string;
  /** Apply even if it doesn't raise the score — a hard constraint, not an optimization. */
  forced?: boolean;
}

function buildRules(protectedWords: Set<string>, charLimit: number): Rule[] {
  return [
  {
    id: "collapse-punctuation",
    label: "Toned down !!! / ??? bursts",
    reason: "Repeated punctuation feeds the same spam-style penalty as ALL-CAPS.",
    transform: collapsePunctuationBursts,
  },
  {
    id: "fix-all-caps",
    label: "Fixed ALL-CAPS shouting",
    reason:
      "A high all-caps ratio pushes up Not-interested/Mute/Report propensity, the three most negative weights in the model.",
    transform: (t: string) => fixAllCapsShouting(t, protectedWords),
  },
  {
    id: "cap-hashtags",
    label: "Trimmed hashtags to 2",
    reason: "Hashtags beyond a couple read as stuffing. Reach comes from the ranking model, not tag volume.",
    transform: (t) => capHashtags(t, 2),
  },
  {
    id: "strip-boilerplate",
    label: "Removed templated CTA phrasing",
    reason:
      'Generic phrases like "link in bio" or "RT if you agree" are the exact fingerprint duplicate-text spam detection looks for.',
    transform: stripBoilerplateCTA,
  },
  {
    id: "cut-filler",
    label: "Cut filler/hedge words",
    reason:
      'Words like "very", "just", "actually", and "I think" dilute a claim without adding information. Tighter phrasing reads more direct and holds attention better.',
    transform: stripFillerWords,
  },
  {
    id: "add-reply-hook",
    label: "Added a reply hook",
    reason:
      "Reply is weighted 5.0–20.0 vs 0.5 for a like: 10–40x more valuable. This is a generic fallback — a question tied to the specifics of your post scores substantially higher, so it's worth replacing by hand or with the AI rewrite.",
    transform: (t: string) => addReplyHook(t, charLimit),
  },
  {
    id: "trim-to-limit",
    label: `Trimmed to fit the ${charLimit.toLocaleString()}-character limit`,
    reason: "Posts over the limit can't be published as-is.",
    transform: (t: string) => trimToCharLimit(t, charLimit),
    forced: true,
  },
  ];
}

// Canonical id -> human label for every optimizer fix, so the tracking layer
// can name a stored fix id without persisting its label alongside every row
// (and without drifting from the labels users actually saw).
export function optimizerRuleLabels(): Record<string, string> {
  return Object.fromEntries(
    buildRules(new Set<string>(), STANDARD_CHAR_LIMIT).map((rule) => [rule.id, rule.label])
  );
}

/**
 * Deterministic, meaning-preserving fixer: each rule is only kept if it
 * measurably raises the BESC Score (or is a hard constraint like the char
 * limit), computed via the same analyzePost() used everywhere else — so
 * "optimized" always means provably scores higher, never just "sounds better."
 */
const MAX_PASSES = 3;

// Multiple passes because later rules (e.g. stripping a boilerplate phrase)
// can juxtapose leftover text into a fresh pattern an earlier rule already
// "fixed" once — e.g. "NOW!!! Link in bio ???" -> "NOW! ?" -> "NOW!?" after
// the boilerplate phrase between them is removed. Rules are idempotent, so
// re-running the whole list until nothing changes converges safely.
export function optimizePost(req: AnalyzeRequest, calibration?: CalibrationModel | null): OptimizeResult {
  const score = (r: AnalyzeRequest) => analyzePost(r, calibration);
  const before = score(req);
  const RULES = buildRules(getHashtagWordSet(req.text), getCharLimit(req.isVerified));
  let current: AnalyzeRequest = { ...req };
  let currentScore = before;
  let anyForcedApplied = false;
  const appliedMap = new Map<string, OptimizeStep>();

  // Walk permissively (>=, not >): the displayed score is clamped/rounded,
  // so a badly-spammy draft can sit pinned at 0 for several individual steps
  // in a row even though they're necessary staging toward a real improvement
  // a few steps later. Gating each step strictly would reject all of them
  // and leave the draft untouched. Whether the walk was actually worthwhile
  // gets decided once, below, by comparing the final score to the start.
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    let changedThisPass = false;

    for (const rule of RULES) {
      const nextText = rule.transform(current.text);
      if (nextText === current.text) continue;

      const candidate: AnalyzeRequest = { ...current, text: nextText };
      const candidateScore = score(candidate);

      if (rule.forced || candidateScore.score >= currentScore.score) {
        if (rule.forced) anyForcedApplied = true;
        const existing = appliedMap.get(rule.id);
        if (existing) {
          existing.scoreAfter = candidateScore.score;
        } else {
          appliedMap.set(rule.id, {
            id: rule.id,
            label: rule.label,
            reason: rule.reason,
            scoreBefore: currentScore.score,
            scoreAfter: candidateScore.score,
          });
        }
        current = candidate;
        currentScore = candidateScore;
        changedThisPass = true;
      }
    }

    if (!changedThisPass) break;
  }

  // Only surface the walk's result if it was actually worth it overall — a
  // hard constraint (char limit) always counts; otherwise require a real
  // net score gain, not just a string of individually-flat steps.
  const worthKeeping = anyForcedApplied || currentScore.score > before.score;

  return worthKeeping
    ? {
        originalText: req.text,
        optimizedText: current.text,
        applied: Array.from(appliedMap.values()),
        before,
        after: currentScore,
        aiStatus: "disabled",
      }
    : {
        originalText: req.text,
        optimizedText: req.text,
        applied: [],
        before,
        after: before,
        aiStatus: "disabled",
      };
}
