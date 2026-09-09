import { NextRequest, NextResponse } from "next/server";
import { dbConfigured } from "@/lib/db";
import {
  MIN_MEASURED_FOR_INSIGHTS,
  buildTrackSummary,
  deleteTrackedPost,
  listTrackedPosts,
  loadBacktest,
  loadTimingInsights,
  saveTrackedPost,
} from "@/lib/tracking";
import { parseHandle } from "@/lib/twitter-import";
import type { MediaType, TrackSummary } from "@/lib/types";

const EMPTY_SUMMARY: TrackSummary = {
  tracked: 0,
  measured: 0,
  minimumForInsights: MIN_MEASURED_FOR_INSIGHTS,
  scoreSplit: null,
  fixInsights: [],
};

// Tracking is the one feature that needs storage. Rather than 500 when no
// database is attached, every endpoint reports enabled:false so the UI can
// explain the situation and the rest of the app carries on untouched.
function disabled() {
  return NextResponse.json({ enabled: false, posts: [], summary: EMPTY_SUMMARY });
}

export async function GET(req: NextRequest) {
  if (!dbConfigured()) return disabled();

  const handle = parseHandle(req.nextUrl.searchParams.get("handle") ?? "");
  if (!handle) {
    return NextResponse.json({ error: "A valid @handle is required" }, { status: 400 });
  }

  try {
    // Browser-style offset, so hour-of-day buckets land in the author's local
    // time — "post in the morning" is only actionable where they actually live.
    const tzOffset = Number(req.nextUrl.searchParams.get("tz")) || 0;
    const [posts, timing, accuracy] = await Promise.all([
      listTrackedPosts(handle),
      loadTimingInsights(handle, tzOffset),
      loadBacktest(handle),
    ]);
    return NextResponse.json({
      enabled: true,
      posts,
      summary: buildTrackSummary(posts),
      timing,
      accuracy,
    });
  } catch (err) {
    console.error("[track] list failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Couldn't load your tracked posts." }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  if (!dbConfigured()) return disabled();

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const handle = parseHandle(String(body.handle ?? ""));
  if (!handle) {
    return NextResponse.json(
      { error: "Add your @handle first — it's how a draft gets matched to the post you publish." },
      { status: 400 }
    );
  }

  const draftText = typeof body.draftText === "string" ? body.draftText.trim().slice(0, 6000) : "";
  if (!draftText) {
    return NextResponse.json({ error: "draftText is required" }, { status: 400 });
  }

  try {
    const post = await saveTrackedPost({
      authorHandle: handle,
      draftText,
      predictedScore: Number(body.predictedScore) || 0,
      predictedGrade: String(body.predictedGrade ?? ""),
      appliedFixIds: Array.isArray(body.appliedFixIds) ? body.appliedFixIds.map(String) : [],
      mediaType: String(body.mediaType ?? "none") as MediaType,
      isReply: Boolean(body.isReply),
      hasMutualFollowAudience: Boolean(body.hasMutualFollowAudience),
      isVerified: Boolean(body.isVerified),
      recentPostsCount: Math.max(0, Math.min(10, Number(body.recentPostsCount) || 0)),
    });
    return NextResponse.json({ enabled: true, post });
  } catch (err) {
    console.error("[track] save failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Couldn't save that draft for tracking." }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  if (!dbConfigured()) return disabled();

  const handle = parseHandle(req.nextUrl.searchParams.get("handle") ?? "");
  const id = Number(req.nextUrl.searchParams.get("id"));
  if (!handle || !Number.isInteger(id)) {
    return NextResponse.json({ error: "handle and id are required" }, { status: 400 });
  }

  try {
    // Scoped by handle as well as id so one handle can't delete another's row.
    await deleteTrackedPost(handle, id);
    // Browser-style offset, so hour-of-day buckets land in the author's local
    // time — "post in the morning" is only actionable where they actually live.
    const tzOffset = Number(req.nextUrl.searchParams.get("tz")) || 0;
    const [posts, timing, accuracy] = await Promise.all([
      listTrackedPosts(handle),
      loadTimingInsights(handle, tzOffset),
      loadBacktest(handle),
    ]);
    return NextResponse.json({
      enabled: true,
      posts,
      summary: buildTrackSummary(posts),
      timing,
      accuracy,
    });
  } catch (err) {
    console.error("[track] delete failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Couldn't remove that tracked post." }, { status: 500 });
  }
}
