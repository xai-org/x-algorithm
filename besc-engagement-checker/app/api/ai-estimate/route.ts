import { NextRequest, NextResponse } from "next/server";
import { analyzePost } from "@/lib/scoring";
import { parseAnalyzeRequest } from "@/lib/request";
import { loadCalibration } from "@/lib/calibrationStore";
import { calibrationStrength } from "@/lib/calibration";
import { dbConfigured } from "@/lib/db";
import { GeminiError } from "@/lib/gemini";
import { estimateActionMultipliers, llmProbabilitiesConfigured } from "@/lib/llmProbabilities";
import { loadEvaluation } from "@/lib/evaluation";
import type { AnalyzeRequest } from "@/lib/types";

/**
 * A second opinion on one draft, from a model that has read the algorithm.
 *
 * Deliberately not part of /api/analyze: that route runs on every keystroke,
 * and a model call per keystroke would be slow, costly and jittery. This is an
 * explicit, on-demand pass.
 */
export const maxDuration = 60;

export async function POST(req: NextRequest) {
  if (!llmProbabilitiesConfigured()) {
    return NextResponse.json(
      { error: "The AI estimator isn't configured on this deployment." },
      { status: 503 }
    );
  }

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

  const handle = typeof body.handle === "string" ? body.handle.trim() : "";
  const calibration = dbConfigured() && handle ? await loadCalibration(handle) : null;

  let multipliers;
  try {
    multipliers = await estimateActionMultipliers(parsed.text, parsed.mediaType);
  } catch (err) {
    const message =
      err instanceof GeminiError ? err.message : "The AI estimator couldn't be reached.";
    console.error("[ai-estimate] failed:", message);
    return NextResponse.json({ error: message }, { status: 502 });
  }

  // Both sides, so the UI can show what the model's read actually changed
  // rather than just replacing a number the user was looking at.
  const baseline = analyzePost(parsed, calibration);
  const withAi = analyzePost(parsed, calibration, multipliers);

  // Whether this estimator has been shown to beat the heuristic on this
  // author's own posts. Surfacing an untested opinion as fact is exactly the
  // failure mode this whole feature exists to avoid.
  const evaluation = dbConfigured() && handle ? await loadEvaluation(handle).catch(() => null) : null;

  return NextResponse.json({
    multipliers,
    baselineScore: baseline.score,
    result: { ...withAi, calibrationStrength: calibrationStrength(calibration) },
    evaluation: evaluation
      ? { llmWins: evaluation.llmWins, n: evaluation.n, evaluatedAt: evaluation.evaluatedAt }
      : null,
  });
}
