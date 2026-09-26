import { NextRequest, NextResponse } from "next/server";
import { optimizePost } from "@/lib/optimize";
import { parseGenerateRequest } from "@/lib/request";
import { getCharLimit } from "@/lib/scoring";
import { loadCalibration } from "@/lib/calibrationStore";
import { dbConfigured } from "@/lib/db";
import { generatePostsFromContext, ollamaConfigured } from "@/lib/ollama";
import { generatePostsFromContextGemini, geminiConfigured } from "@/lib/gemini";
import type { AICandidate, AnalyzeRequest, GenerateRequest, GenerateStatus } from "@/lib/types";

export async function POST(req: NextRequest) {
  let body: Partial<GenerateRequest> & { handle?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = parseGenerateRequest(body);
  if (!parsed) {
    return NextResponse.json({ error: "context is required" }, { status: 400 });
  }

  if (!geminiConfigured() && !ollamaConfigured()) {
    return NextResponse.json({ generateStatus: "disabled" satisfies GenerateStatus });
  }

  try {
    const { context, ...contextFields } = parsed;
    const calibration =
      dbConfigured() && typeof body.handle === "string" && body.handle.trim()
        ? await loadCalibration(body.handle)
        : null;
    const charLimit = getCharLimit(parsed.isVerified);

    // Provider-specific defaults on purpose: Gemini (hosted, cheap) samples
    // more variants for a wider best-of-N pool; Ollama (self-hosted CPU)
    // stays conservative given its timeout history — see lib/ollama.ts.
    const rawCandidates = geminiConfigured()
      ? await generatePostsFromContextGemini(context, undefined, charLimit, parsed)
      : await generatePostsFromContext(context, undefined, charLimit, parsed);

    // Every AI-written candidate still gets run through the deterministic
    // optimizer before it's shown — same "suggest, never decide" contract as
    // the rewrite flow, so a generated draft can't skip the mechanical
    // fixes (hashtag trim, filler cut, etc.) the rest of this app enforces.
    const scored: AICandidate[] = rawCandidates
      .map((text) => {
        const analyzeReq: AnalyzeRequest = { ...contextFields, text };
        const optimized = optimizePost(analyzeReq, calibration);
        return { text: optimized.optimizedText, score: optimized.after.score };
      })
      .sort((a, b) => b.score - a.score);

    if (scored.length === 0) {
      return NextResponse.json({ generateStatus: "empty" satisfies GenerateStatus });
    }
    return NextResponse.json({ generateStatus: "found" satisfies GenerateStatus, candidates: scored });
  } catch (err) {
    const provider = geminiConfigured() ? "Gemini" : "Ollama";
    console.error(`[generate] ${provider} generation failed:`, err instanceof Error ? err.message : err);
    return NextResponse.json({ generateStatus: "error" satisfies GenerateStatus });
  }
}
