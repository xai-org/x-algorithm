import { buildGeneratePrompt, buildRewritePrompt, estimateMaxOutputTokens, parseVariants } from "./aiPrompt";
import type { AnalyzeRequest, GenerateRequest } from "./types";

export class GeminiError extends Error {}

// Hosted API, not self-hosted CPU inference — no cold-start/model-loading
// story here, so a much shorter timeout than the Ollama path is safe and
// keeps a real outage from stalling the response for anywhere near as long.
const REQUEST_TIMEOUT_MS = 20000;

// Gemini's flash-tier models can spend a chunk of maxOutputTokens on internal
// reasoning before emitting any visible text, so a budget sized only for the
// visible output gets truncated mid-sentence — which is how a variant ended
// up cut off mid-URL in production. Hosted tokens are cheap and unused budget
// costs nothing (billing is on tokens actually produced), so give generous
// headroom rather than sizing this to the wire.
const THINKING_HEADROOM_TOKENS = 2000;

// A dated model ID (e.g. "gemini-2.5-flash") can get retired out from under
// new API keys without warning — that's exactly what happened here. The
// "-latest" alias is Google's own answer to that: it always resolves to
// their current recommended flash-tier model, with an emailed 2-week notice
// before the alias target changes, instead of a cold 404 in production.
const DEFAULT_MODEL = "gemini-flash-latest";

export function geminiConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY);
}

interface GeminiResponse {
  candidates?: {
    content?: { parts?: { text?: string }[] };
    finishReason?: string;
  }[];
  promptFeedback?: { blockReason?: string };
}

async function callGeminiForVariants(
  prompt: string,
  numVariants: number,
  maxOutputTokens: number
): Promise<string[]> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new GeminiError("GEMINI_API_KEY not configured");
  }
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
          generationConfig: { temperature: 0.8, maxOutputTokens },
        }),
        signal: controller.signal,
      }
    );
  } catch (err) {
    if (controller.signal.aborted) {
      throw new GeminiError(`Gemini request timed out after ${(REQUEST_TIMEOUT_MS / 1000).toFixed(0)}s`);
    }
    throw new GeminiError(`Couldn't reach Gemini: ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new GeminiError(`Gemini returned ${res.status}${detail ? `: ${detail}` : ""}`);
  }

  const data = (await res.json()) as GeminiResponse;
  if (data.promptFeedback?.blockReason) {
    throw new GeminiError(`Gemini blocked the request: ${data.promptFeedback.blockReason}`);
  }

  const raw = data.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("") ?? "";
  return parseVariants(raw, numVariants);
}

export async function generateRewriteCandidatesGemini(
  text: string,
  numVariants = 2,
  charLimit = 280,
  req?: AnalyzeRequest
): Promise<string[]> {
  return callGeminiForVariants(
    buildRewritePrompt(text, numVariants, charLimit, req),
    numVariants,
    estimateMaxOutputTokens(charLimit, numVariants) + THINKING_HEADROOM_TOKENS
  );
}

// More variants than the rewrite path on purpose: generating from scratch
// has no single "obviously correct" direction the way tightening an
// existing draft does, so a wider best-of-N sample genuinely raises the
// odds the top-scoring one is actually good, not just the least-bad option
// out of a couple of tries.
export async function generatePostsFromContextGemini(
  context: string,
  numVariants = 5,
  charLimit = 280,
  req?: GenerateRequest
): Promise<string[]> {
  return callGeminiForVariants(
    buildGeneratePrompt(context, numVariants, charLimit, req),
    numVariants,
    estimateMaxOutputTokens(charLimit, numVariants) + THINKING_HEADROOM_TOKENS
  );
}
