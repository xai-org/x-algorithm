import { buildGeneratePrompt, buildRewritePrompt, estimateMaxOutputTokens, parseVariants } from "./aiPrompt";
import type { AnalyzeRequest, GenerateRequest } from "./types";

export class OllamaError extends Error {}

// Local CPU inference on a self-hosted model can be genuinely slow —
// generous timeout, but still a hard one. Learned this lesson the hard way
// with the Vee3 "loads forever" bug: an external call with no deadline
// leaves the UI spinning with no way out. Sized for the default 14B model on
// CPU; tighten this back up if you drop to a smaller model.
const REQUEST_TIMEOUT_MS = 100000;

// Generation time on CPU scales with token count, and this service has a
// real documented history of hitting REQUEST_TIMEOUT_MS (see git log —
// "loads forever," the cold-start saga). estimateMaxOutputTokens()'s shared
// 2000-token ceiling is sized for a hosted API, not CPU inference — capped
// much lower here so a verified account's large charLimit or a wide
// best-of-N sample can't reintroduce that timeout risk on this path.
const OLLAMA_MAX_OUTPUT_TOKENS = 500;

// Ollama's own CPU thread auto-detection is a known-unreliable in containers
// (it can under-count what's actually schedulable), which is a plausible
// explanation for a service with 24 real vCPUs still hitting the same
// generation timeout regardless of model size. num_thread is a documented
// per-request option on /api/generate, more reliable than guessing at
// server-level env var names that vary across Ollama versions. Override via
// OLLAMA_NUM_THREAD if the actual usable core count differs from vCPUs shown
// in Railway (e.g. due to container CPU quota vs. host core count).
const NUM_THREAD = Number(process.env.OLLAMA_NUM_THREAD) || 24;

export function ollamaConfigured(): boolean {
  return Boolean(process.env.OLLAMA_URL && process.env.OLLAMA_MODEL);
}

const HEALTH_CHECK_TIMEOUT_MS = 8000;

// A generate-request timeout is ambiguous on its own: it can't tell you
// whether the model was genuinely slow, or the request never actually
// reached the server at all (wrong URL, private networking not enabled,
// service down) and just happened to hang until the same deadline. This
// hits Ollama's lightweight /api/tags endpoint first, with its own short
// timeout, so a connectivity failure is reported as exactly that instead of
// masquerading as "the model is slow" — and confirms the configured model
// is actually loaded before spending up to REQUEST_TIMEOUT_MS finding out
// it isn't.
async function checkOllamaHealth(url: string, model: string): Promise<void> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${url.replace(/\/$/, "")}/api/tags`, { signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new OllamaError(
        `Ollama isn't reachable at all: the connectivity check itself timed out after ${(HEALTH_CHECK_TIMEOUT_MS / 1000).toFixed(0)}s. This points at a networking problem (wrong OLLAMA_URL, private networking not enabled between the two services, or the ollama service is down/still starting), not a slow model.`
      );
    }
    throw new OllamaError(
      `Can't reach Ollama at ${url}: ${(err as Error).message}. Check OLLAMA_URL and that private networking is enabled on both services.`
    );
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    throw new OllamaError(
      `Ollama responded to the connectivity check with HTTP ${res.status} — it's reachable but not healthy.`
    );
  }

  const data = (await res.json().catch(() => ({}))) as { models?: { name?: string; model?: string }[] };
  const available = (data.models ?? []).map((m) => m.name ?? m.model ?? "").filter(Boolean);
  if (available.length > 0 && !available.includes(model)) {
    throw new OllamaError(
      `Ollama is reachable, but model "${model}" isn't loaded there. Available: ${available.join(", ")}. Check OLLAMA_MODEL matches exactly what was pulled.`
    );
  }
}

async function callOllamaForVariants(
  prompt: string,
  numVariants: number,
  maxOutputTokens: number
): Promise<string[]> {
  const url = process.env.OLLAMA_URL;
  const model = process.env.OLLAMA_MODEL;
  if (!url || !model) {
    throw new OllamaError("OLLAMA_URL/OLLAMA_MODEL not configured");
  }

  await checkOllamaHealth(url, model);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${url.replace(/\/$/, "")}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: { temperature: 0.8, num_predict: maxOutputTokens, num_thread: NUM_THREAD },
      }),
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new OllamaError(`Ollama request timed out after ${(REQUEST_TIMEOUT_MS / 1000).toFixed(0)}s`);
    }
    throw new OllamaError(`Couldn't reach Ollama: ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new OllamaError(`Ollama returned ${res.status}${detail ? `: ${detail}` : ""}`);
  }

  const data = (await res.json()) as { response?: string };
  return parseVariants(data.response ?? "", numVariants);
}

export async function generateRewriteCandidates(
  text: string,
  numVariants = 2,
  charLimit = 280,
  req?: AnalyzeRequest
): Promise<string[]> {
  return callOllamaForVariants(
    buildRewritePrompt(text, numVariants, charLimit, req),
    numVariants,
    Math.min(OLLAMA_MAX_OUTPUT_TOKENS, estimateMaxOutputTokens(charLimit, numVariants))
  );
}

// Kept lower than Gemini's default (5) — more variants means more output
// tokens means more CPU time on this path, and this service doesn't get to
// spend that budget as freely given its timeout history.
export async function generatePostsFromContext(
  context: string,
  numVariants = 3,
  charLimit = 280,
  req?: GenerateRequest
): Promise<string[]> {
  return callOllamaForVariants(
    buildGeneratePrompt(context, numVariants, charLimit, req),
    numVariants,
    Math.min(OLLAMA_MAX_OUTPUT_TOKENS, estimateMaxOutputTokens(charLimit, numVariants))
  );
}
