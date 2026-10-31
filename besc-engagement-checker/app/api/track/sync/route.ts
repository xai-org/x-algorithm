import { NextRequest, NextResponse } from "next/server";
import { dbConfigured } from "@/lib/db";
import { buildTrackSummary, listTrackedPosts, syncTrackedPosts } from "@/lib/tracking";
import { parseHandle } from "@/lib/twitter-import";
import { Vee3Error } from "@/lib/vee3";

/**
 * Matches pending drafts against what the handle actually published, then
 * refreshes engagement numbers for anything old enough to have real ones.
 * Driven on demand from the UI rather than a background worker: it's
 * idempotent, only spends API calls on rows that need them, and this way the
 * feature needs no extra infrastructure to run.
 */
export async function POST(req: NextRequest) {
  if (!dbConfigured()) {
    return NextResponse.json({ enabled: false, posts: [], matched: 0, refreshed: 0 });
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
    const { matched, refreshed } = await syncTrackedPosts(handle);
    const posts = await listTrackedPosts(handle);
    return NextResponse.json({
      enabled: true,
      matched,
      refreshed,
      posts,
      summary: buildTrackSummary(posts),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[track/sync] failed:", message);
    // A Vee3 outage is the user's most likely failure here and it's temporary —
    // say so specifically instead of a generic error, since their saved data
    // is fine and retrying later will work.
    if (err instanceof Vee3Error) {
      return NextResponse.json(
        { error: `Couldn't reach X right now to check your posts: ${message}` },
        { status: 502 }
      );
    }
    return NextResponse.json({ error: "Couldn't sync your tracked posts." }, { status: 500 });
  }
}
