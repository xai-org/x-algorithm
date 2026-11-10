import { NextRequest, NextResponse } from "next/server";
import { analyzePost } from "@/lib/scoring";
import { parseAnalyzeRequest } from "@/lib/request";
import { loadCalibration } from "@/lib/calibrationStore";
import { calibrationStrength } from "@/lib/calibration";
import { dbConfigured } from "@/lib/db";
import type { AnalyzeRequest } from "@/lib/types";

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

  // When this author has a model fitted to their own results, score against
  // that instead of the hand-tuned priors. Cached and heavily guarded, so a
  // missing model or unreachable database just falls through to heuristics —
  // scoring must never depend on calibration being available.
  const calibration =
    dbConfigured() && typeof body.handle === "string" && body.handle.trim()
      ? await loadCalibration(body.handle)
      : null;

  return NextResponse.json({
    ...analyzePost(parsed, calibration),
    calibrationStrength: calibrationStrength(calibration),
  });
}
