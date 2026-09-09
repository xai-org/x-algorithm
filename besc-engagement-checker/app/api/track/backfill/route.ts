import { NextRequest, NextResponse } from "next/server";
import { dbConfigured } from "@/lib/db";
import { backfillFromTimeline, invalidateCalibrationCache, refitCalibration } from "@/lib/calibrationStore";
import { calibrationStrength, MIN_SAMPLES_FOR_FIT } from "@/lib/calibration";
import { parseHandle } from "@/lib/twitter-import";
import { Vee3Error } from "@/lib/vee3";

// Backfill walks several pages of timeline and then refits, so it runs well
// past the default serverless slice.
export const maxDuration = 120;

/**
 * Pulls the handle's published history, stores each post with its real
 * engagement numbers, and refits the calibration model from everything
 * available. This is what turns the scorer's guessed action probabilities
 * into ones measured against the author's own results.
 */
export async function POST(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ enabled: false });
  }

  let body: { handle?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const handle = parseHandle(body.handle ?? "");
  if (!handle) {
    return NextResponse.json({ error: "A valid @handle is required" }, { status: 400 });
  }

  try {
    const backfill = await backfillFromTimeline(handle);
    const model = await refitCalibration(handle);
    invalidateCalibrationCache(handle);

    return NextResponse.json({
      enabled: true,
      ...backfill,
      calibration: {
        n: model.n,
        minimumForFit: MIN_SAMPLES_FOR_FIT,
        strength: calibrationStrength(model),
        actions: Object.entries(model.actions).map(([action, m]) => ({
          action,
          cvR2: m!.cvR2,
        })),
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[track/backfill] failed:", message);
    if (err instanceof Vee3Error) {
      return NextResponse.json(
        { error: `Couldn't read your timeline from X right now: ${message}` },
        { status: 502 }
      );
    }
    return NextResponse.json({ error: "Couldn't backfill your post history." }, { status: 500 });
  }
}
