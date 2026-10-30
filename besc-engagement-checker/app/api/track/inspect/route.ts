import { NextRequest, NextResponse } from "next/server";
import { inspectTimelineShape, parseHandle } from "@/lib/twitter-import";

/**
 * Diagnostic for metric mapping. Backfill silently recorded 0 likes across an
 * entire history because a missing field is indistinguishable from a real
 * zero, so this reports what the timeline response actually contains.
 */
export async function POST(req: NextRequest) {
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
    return NextResponse.json(await inspectTimelineShape(handle));
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Inspect failed" },
      { status: 502 }
    );
  }
}
