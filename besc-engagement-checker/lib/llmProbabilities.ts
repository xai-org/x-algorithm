import { buildAlgorithmBrief } from "./algorithmContext";
import { GeminiError } from "./gemini";
import { LLM_ACTIONS, type ActionMultipliers } from "./types";

/**
 * An LLM-based estimate of the action probabilities, feeding the same real
 * weighted scorer everything else uses.
 *
 * The important design choice here is that it asks for *relative* judgements
 * rather than absolute probabilities. Language models are genuinely good at
 * "this post will get roughly twice the replies of a typical one" and
 * genuinely bad at "P(reply) = 0.031" — absolute rates depend on follower
 * count, timing and audience, none of which are in the text. Asking for
 * absolutes would produce confident numbers whose scale is meaningless, and
 * because the 0-100 score is a squashed sum, a wrong scale would shift every
 * grade.
 *
 * Multipliers against the author's own baseline sidestep that entirely, and
 * they slot into exactly the same mechanism the fitted regression already uses
 * (see calibratedRatio in calibration.ts). Whether this beats the heuristics is
 * not assumed — /api/track/evaluate backtests it against real posts.
 */

// Declared in types.ts so the scorer (which the browser bundles) can reference
// them without pulling this module's prompt text into the client bundle.
export { LLM_ACTIONS, type ActionMultipliers } from "./types";

// Clamped hard. One bad generation shouldn't be able to multiply the heaviest
// weight in the model by 50 and produce a nonsense score.
const MIN_MULTIPLIER = 0.25;
const MAX_MULTIPLIER = 4;

const REQUEST_TIMEOUT_MS = 25000;
const DEFAULT_MODEL = "gemini-flash-latest";

// Backtesting means scoring dozens of posts, and one request per post would be
// both slow and needlessly expensive — the algorithm brief is the bulk of the
// prompt and is identical every time. Six at a time keeps each response small
// enough to stay well inside the output budget.
const BATCH_SIZE = 6;
const BATCH_CONCURRENCY = 3;

function buildPrompt(text: string, mediaType: string): string {
  return `${buildAlgorithmBrief()}

YOUR TASK
Estimate how this specific post will perform RELATIVE TO A TYPICAL POST from the same account, for each action below.

Report a multiplier, not a probability:
  1.0  = about the same as this author's typical post
  2.0  = roughly twice as likely
  0.5  = roughly half as likely
Stay within 0.25 and 4.0. Most posts are unremarkable on most actions, so 1.0 is the right answer far more often than not — only move away from it when something about this post genuinely justifies it.

Judge only from the text and format. You do not know the author's follower count, posting time or audience, so do not guess at absolute rates.

For the negative actions (notInterested, muteAuthor, report), a HIGHER multiplier means MORE likely to be reported/muted — that is bad. A clean, ordinary post should be 1.0 or below.

Post (media: ${mediaType}):
"""
${text}
"""

Respond with ONLY a JSON object, no prose or code fences, with exactly these numeric keys:
{"favorite":1.0,"reply":1.0,"retweet":1.0,"quote":1.0,"shareViaCopyLink":1.0,"followAuthor":1.0,"notInterested":1.0,"muteAuthor":1.0,"report":1.0}`;
}

function parseMultipliers(raw: string): ActionMultipliers {
  // Models wrap JSON in code fences often enough that failing on it would be
  // a self-inflicted error rather than a real one.
  const cleaned = raw.replace(/```(?:json)?/gi, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1) throw new GeminiError("No JSON object in the response");

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    throw new GeminiError("Response was not valid JSON");
  }

  const out: ActionMultipliers = {};
  for (const action of LLM_ACTIONS) {
    const value = Number(parsed[action]);
    // Silently skipping a missing/garbage key leaves that action on the
    // heuristic, which is the correct fallback rather than defaulting to 1.0
    // and pretending the model had an opinion.
    if (!Number.isFinite(value) || value <= 0) continue;
    out[action] = Math.max(MIN_MULTIPLIER, Math.min(MAX_MULTIPLIER, value));
  }
  return out;
}

export function llmProbabilitiesConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY);
}

async function callGemini(prompt: string, maxOutputTokens: number): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new GeminiError("GEMINI_API_KEY not configured");
  const model = process.env.GEMINI_MODEL || DEFAULT_MODEL;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            // Near-deterministic: this is a measurement, and the same draft
            // should not score differently on a refresh.
            temperature: 0.1,
            maxOutputTokens,
            responseMimeType: "application/json",
          },
        }),
        signal: controller.signal,
      }
    );
  } catch (err) {
    if (controller.signal.aborted) throw new GeminiError("Probability estimate timed out");
    throw new GeminiError(`Couldn't reach Gemini: ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new GeminiError(`Gemini returned ${res.status}${detail ? `: ${detail}` : ""}`);
  }

  const data = (await res.json()) as {
    candidates?: { content?: { parts?: { text?: string }[] } }[];
  };
  return data.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("") ?? "";
}

export async function estimateActionMultipliers(
  text: string,
  mediaType: string
): Promise<ActionMultipliers> {
  return parseMultipliers(await callGemini(buildPrompt(text, mediaType), 2200));
}

export interface BatchItem {
  text: string;
  mediaType: string;
}

function buildBatchPrompt(items: BatchItem[]): string {
  const posts = items
    .map(
      (item, i) => `POST ${i + 1} (media: ${item.mediaType}):
"""
${item.text}
"""`
    )
    .join("\n\n");

  return `${buildAlgorithmBrief()}

YOUR TASK
For each post below, estimate how it will perform RELATIVE TO A TYPICAL POST from the same account, for each action.

Report multipliers, not probabilities:
  1.0  = about the same as this author's typical post
  2.0  = roughly twice as likely
  0.5  = roughly half as likely
Stay within 0.25 and 4.0. Judge each post independently and on its own merits — do not rank them against each other, and do not assume the set contains a spread of good and bad posts. Most posts are unremarkable on most actions, so 1.0 is the right answer far more often than not.

Judge only from the text and format. You do not know follower counts, posting times or audiences, so do not guess at absolute rates.

For the negative actions (notInterested, muteAuthor, report), a HIGHER multiplier means MORE likely to be reported/muted — that is bad. A clean, ordinary post should be 1.0 or below.

${posts}

Respond with ONLY a JSON array of ${items.length} objects, in the same order as the posts, no prose or code fences. Each object has exactly these numeric keys:
{"favorite":1.0,"reply":1.0,"retweet":1.0,"quote":1.0,"shareViaCopyLink":1.0,"followAuthor":1.0,"notInterested":1.0,"muteAuthor":1.0,"report":1.0}`;
}

function parseBatch(raw: string, expected: number): (ActionMultipliers | null)[] {
  const cleaned = raw.replace(/```(?:json)?/gi, "").trim();
  const start = cleaned.indexOf("[");
  const end = cleaned.lastIndexOf("]");
  if (start === -1 || end === -1) throw new GeminiError("No JSON array in the response");

  let parsed: unknown;
  try {
    parsed = JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    throw new GeminiError("Batch response was not valid JSON");
  }
  if (!Array.isArray(parsed)) throw new GeminiError("Batch response was not an array");

  // A short array means the model dropped posts off the end. Those positions
  // stay null rather than being silently shifted onto the wrong post — a
  // misaligned estimate would quietly corrupt the whole comparison.
  const out: (ActionMultipliers | null)[] = new Array(expected).fill(null);
  for (let i = 0; i < Math.min(expected, parsed.length); i++) {
    try {
      out[i] = parseMultipliers(JSON.stringify(parsed[i]));
    } catch {
      out[i] = null;
    }
  }
  return out;
}

/**
 * Estimates a whole set of posts. Failures come back as null for that position
 * rather than throwing, so one bad chunk costs a few samples instead of the
 * entire evaluation.
 */
export async function estimateActionMultipliersBatch(
  items: BatchItem[]
): Promise<(ActionMultipliers | null)[]> {
  const chunks: { at: number; items: BatchItem[] }[] = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    chunks.push({ at: i, items: items.slice(i, i + BATCH_SIZE) });
  }

  const results: (ActionMultipliers | null)[] = new Array(items.length).fill(null);
  let next = 0;

  async function worker() {
    while (next < chunks.length) {
      const chunk = chunks[next++];
      try {
        const parsed = parseBatch(
          await callGemini(buildBatchPrompt(chunk.items), 700 * chunk.items.length + 1500),
          chunk.items.length
        );
        parsed.forEach((m, i) => {
          results[chunk.at + i] = m;
        });
      } catch (err) {
        console.error(
          "[llmProbabilities] batch failed:",
          err instanceof Error ? err.message : err
        );
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(BATCH_CONCURRENCY, chunks.length) }, () => worker())
  );
  return results;
}
