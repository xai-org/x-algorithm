import { NextRequest, NextResponse } from "next/server";
import { dbConfigured } from "@/lib/db";
import { loadEvaluation, runEvaluation } from "@/lib/evaluation";
import { llmProbabilitiesConfigured } from "@/lib/llmProbabilities";
import { parseHandle } from "@/lib/twitter-import";

// Dozens of posts through a batched model call takes longer than the default
// serverless slice allows.
export const maxDuration = 120;

function requireHandle(raw: string) {
  const handle = parseHandle(raw);
  if (!handle) {
    return { error: NextResponse.json({ error: "A valid @handle is required" }, { status: 400 }) };
  }
  return { handle };
}

/** The stored verdict, so the UI can show it without paying for a re-run. */
export async function GET(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ enabled: false, evaluation: null, aiConfigured: false });
  }

  const { handle, error } = requireHandle(req.nextUrl.searchParams.get("handle") ?? "");
  if (error) return error;

  try {
    return NextResponse.json({
      enabled: true,
      aiConfigured: llmProbabilitiesConfigured(),
      evaluation: await loadEvaluation(handle!),
    });
  } catch (err) {
    console.error("[track/evaluate] load failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Couldn't load the last evaluation." }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ enabled: false, evaluation: null, aiConfigured: false });
  }
  if (!llmProbabilitiesConfigured()) {
    return NextResponse.json(
      { error: "The AI estimator isn't configured on this deployment." },
      { status: 503 }
    );
  }

  let body: Record<string, unknown> = {};
  try {
    body = await req.json();
  } catch {
    // An empty body is fine — handle can come from the query string.
  }

  const { handle, error } = requireHandle(
    String(body.handle ?? req.nextUrl.searchParams.get("handle") ?? "")
  );
  if (error) return error;

  try {
    const evaluation = await runEvaluation(handle!, Number(body.limit) || undefined);
    if (evaluation.n === 0) {
      return NextResponse.json(
        {
          error:
            "No published posts with view counts to test against yet — run “Learn from my history” first.",
        },
        { status: 400 }
      );
    }
    return NextResponse.json({ enabled: true, aiConfigured: true, evaluation });
  } catch (err) {
    console.error("[track/evaluate] run failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Couldn't run the evaluation." }, { status: 500 });
  }
}
