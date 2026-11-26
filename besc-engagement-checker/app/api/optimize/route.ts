import { NextRequest, NextResponse } from "next/server";
import { optimizePost } from "@/lib/optimize";
import { parseAnalyzeRequest } from "@/lib/request";
import { analyzePost, getCharLimit } from "@/lib/scoring";
import { loadCalibration } from "@/lib/calibrationStore";
import { dbConfigured } from "@/lib/db";
import { generateRewriteCandidates, ollamaConfigured } from "@/lib/ollama";
import { generateRewriteCandidatesGemini, geminiConfigured } from "@/lib/gemini";
import type { AICandidate, AIStatus, AnalyzeRequest } from "@/lib/types";

export async function POST(req: NextRequest) {
  let body: Partial<AnalyzeRequest> & { handle?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = parseAnalyzeRequest(body);
  if (!parsed) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  // Score everything — mechanical fixes and AI candidates alike — against the
  // same model the live score uses, so the gate that decides which candidates
  // surface is consistent with the number the user is looking at.
  const calibration =
    dbConfigured() && typeof body.handle === "string" && body.handle.trim()
      ? await loadCalibration(body.handle)
      : null;

  const result = optimizePost(parsed, calibration);

  // Gemini (hosted) is preferred over Ollama (self-hosted, CPU-bound) when
  // both are configured — it's faster and doesn't carry the cold-start risk
  // documented in lib/ollama.ts. Ollama stays available as a free fallback.
  if (!geminiConfigured() && !ollamaConfigured()) {
    return NextResponse.json({ ...result, aiStatus: "disabled" satisfies AIStatus });
  }

  try {
    const charLimit = getCharLimit(parsed.isVerified);
    const candidates = geminiConfigured()
      ? await generateRewriteCandidatesGemini(result.optimizedText, 2, charLimit, parsed)
      : await generateRewriteCandidates(result.optimizedText, 2, charLimit, parsed);
    const scored: AICandidate[] = candidates
      .map((text) => ({ text, score: analyzePost({ ...parsed, text }, calibration).score }))
      .filter((c) => c.score > result.after.score)
      .sort((a, b) => b.score - a.score);

    if (scored.length > 0) {
      return NextResponse.json({
        ...result,
        aiStatus: "found" satisfies AIStatus,
        aiCandidates: scored,
      });
    }
    return NextResponse.json({ ...result, aiStatus: "no_improvement" satisfies AIStatus });
  } catch (err) {
    // The AI rewrite is a bonus layer on top of the always-available
    // deterministic optimizer — a failure here should never break the
    // response, just fall through and return the mechanical result with an
    // honest status instead of silently pretending AI wasn't attempted.
    const provider = geminiConfigured() ? "Gemini" : "Ollama";
    console.error(`[optimize] ${provider} rewrite failed:`, err instanceof Error ? err.message : err);
    return NextResponse.json({ ...result, aiStatus: "error" satisfies AIStatus });
  }
}
