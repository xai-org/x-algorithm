import { query } from "./db";
import { fetchTweetMetrics, fetchUserTimeline, type TimelinePost } from "./twitter-import";
import { findBestMatch } from "./textMatch";
import { optimizerRuleLabels } from "./optimize";
import { buildTimingInsights, type TimingInsights, type TimingSample } from "./timing";
import { backtest, type BacktestResult } from "./backtest";
import type {
  CalibrationSide,
  FixInsight,
  MediaType,
  TrackSummary,
  TrackedPost,
  TrackedPostMetrics,
} from "./types";

// Engagement data is extremely noisy at low volume — a single post landing in
// someone's feed at the right moment can swamp any real effect. These gates
// exist so the tool never dresses up 3 data points as a finding. The whole
// point of this feature is being able to trust what it tells you.
export const MIN_MEASURED_FOR_INSIGHTS = 6;
const MIN_PER_SIDE_FOR_FIX = 3;

// Don't measure a post the instant it's published — engagement hasn't
// accumulated yet and an early snapshot would understate it. Also don't
// re-fetch a post we already measured recently; each refresh costs a Vee3 call.
const MIN_AGE_BEFORE_MEASURING_MS = 2 * 60 * 60 * 1000;
const METRICS_STALE_AFTER_MS = 6 * 60 * 60 * 1000;

interface TrackedRow {
  id: number;
  created_at: Date;
  draft_text: string;
  predicted_score: number;
  predicted_grade: string;
  applied_fix_ids: string[];
  is_reply: boolean;
  media_type: string;
  tweet_id: string | null;
  posted_at: Date | null;
  metrics_updated_at: Date | null;
  views: string | number | null;
  likes: string | number | null;
  retweets: string | number | null;
  replies: string | number | null;
  quotes: string | number | null;
  bookmarks: string | number | null;
}

// pg returns BIGINT as a string to avoid precision loss; every metric here is
// far below 2^53 so coercing to number is safe and keeps the API JSON-clean.
function num(value: string | number | null): number {
  return value === null ? 0 : Number(value);
}

function rowToPost(row: TrackedRow): TrackedPost {
  const measured = row.metrics_updated_at !== null;
  return {
    id: row.id,
    createdAt: row.created_at.toISOString(),
    draftText: row.draft_text,
    predictedScore: Number(row.predicted_score),
    predictedGrade: row.predicted_grade,
    appliedFixIds: row.applied_fix_ids ?? [],
    isReply: row.is_reply,
    mediaType: row.media_type as MediaType,
    tweetId: row.tweet_id,
    postedAt: row.posted_at ? row.posted_at.toISOString() : null,
    metricsUpdatedAt: row.metrics_updated_at ? row.metrics_updated_at.toISOString() : null,
    metrics: measured
      ? {
          views: num(row.views),
          likes: num(row.likes),
          retweets: num(row.retweets),
          replies: num(row.replies),
          quotes: num(row.quotes),
          bookmarks: num(row.bookmarks),
        }
      : null,
  };
}

const SELECT_COLUMNS = `id, created_at, draft_text, predicted_score, predicted_grade,
  applied_fix_ids, is_reply, media_type, tweet_id, posted_at, metrics_updated_at,
  views, likes, retweets, replies, quotes, bookmarks`;

export interface SaveTrackedInput {
  authorHandle: string;
  draftText: string;
  predictedScore: number;
  predictedGrade: string;
  appliedFixIds: string[];
  mediaType: MediaType;
  isReply: boolean;
  hasMutualFollowAudience: boolean;
  isVerified: boolean;
  recentPostsCount: number;
}

export async function saveTrackedPost(input: SaveTrackedInput): Promise<TrackedPost> {
  const rows = await query<TrackedRow>(
    `INSERT INTO tracked_posts
       (author_handle, draft_text, predicted_score, predicted_grade, applied_fix_ids,
        media_type, is_reply, has_mutual_audience, is_verified, recent_posts_count)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
     RETURNING ${SELECT_COLUMNS}`,
    [
      input.authorHandle.toLowerCase(),
      input.draftText,
      input.predictedScore,
      input.predictedGrade,
      input.appliedFixIds,
      input.mediaType,
      input.isReply,
      input.hasMutualFollowAudience,
      input.isVerified,
      input.recentPostsCount,
    ]
  );
  return rowToPost(rows[0]);
}

export async function listTrackedPosts(handle: string): Promise<TrackedPost[]> {
  const rows = await query<TrackedRow>(
    `SELECT ${SELECT_COLUMNS} FROM tracked_posts
     WHERE author_handle = $1 ORDER BY created_at DESC LIMIT 100`,
    [handle.toLowerCase()]
  );
  return rows.map(rowToPost);
}

export async function deleteTrackedPost(handle: string, id: number): Promise<void> {
  await query(`DELETE FROM tracked_posts WHERE id = $1 AND author_handle = $2`, [
    id,
    handle.toLowerCase(),
  ]);
}

export interface SyncDeps {
  fetchTimeline: (handle: string) => Promise<TimelinePost[]>;
  fetchMetrics: (tweetId: string) => Promise<TrackedPostMetrics>;
}

const LIVE_DEPS: SyncDeps = {
  fetchTimeline: fetchUserTimeline,
  fetchMetrics: fetchTweetMetrics,
};

/**
 * Pulls the handle's recent timeline, matches still-unpublished drafts against
 * it, then refreshes metrics for anything matched, old enough to have real
 * numbers, and not measured recently. Safe to call repeatedly — it only spends
 * API calls on rows that actually need them.
 *
 * `deps` is injectable so the matching/refresh orchestration can be tested
 * without live network calls; production callers use the default.
 */
export async function syncTrackedPosts(
  handle: string,
  deps: SyncDeps = LIVE_DEPS
): Promise<{ matched: number; refreshed: number }> {
  const posts = await listTrackedPosts(handle);
  let matched = 0;
  let refreshed = 0;

  const unmatched = posts.filter((p) => !p.tweetId);
  if (unmatched.length > 0) {
    const timeline = await deps.fetchTimeline(handle);
    // Never let one published tweet be claimed by two drafts — the partial
    // unique index would reject it anyway, but failing here is cheaper and
    // lets the rest of the sync continue.
    const alreadyClaimed = new Set(posts.map((p) => p.tweetId).filter(Boolean) as string[]);

    for (const draft of unmatched) {
      const available = timeline.filter((t) => !alreadyClaimed.has(t.tweetId));
      const best = findBestMatch(draft.draftText, available);
      if (!best) continue;

      const postedAt = best.candidate.createdAt ? new Date(best.candidate.createdAt) : null;
      await query(
        `UPDATE tracked_posts SET tweet_id = $1, matched_at = NOW(), posted_at = $2
         WHERE id = $3 AND tweet_id IS NULL`,
        [best.candidate.tweetId, postedAt && !Number.isNaN(postedAt.getTime()) ? postedAt : null, draft.id]
      );
      alreadyClaimed.add(best.candidate.tweetId);
      matched++;
    }
  }

  const now = Date.now();
  const current = matched > 0 ? await listTrackedPosts(handle) : posts;
  for (const post of current) {
    if (!post.tweetId) continue;

    const publishedAt = post.postedAt ? new Date(post.postedAt).getTime() : new Date(post.createdAt).getTime();
    if (now - publishedAt < MIN_AGE_BEFORE_MEASURING_MS) continue;

    const lastMeasured = post.metricsUpdatedAt ? new Date(post.metricsUpdatedAt).getTime() : 0;
    if (now - lastMeasured < METRICS_STALE_AFTER_MS) continue;

    try {
      const m = await deps.fetchMetrics(post.tweetId);
      await query(
        `UPDATE tracked_posts
         SET views=$1, likes=$2, retweets=$3, replies=$4, quotes=$5, bookmarks=$6,
             metrics_updated_at = NOW()
         WHERE id = $7`,
        [m.views, m.likes, m.retweets, m.replies, m.quotes, m.bookmarks, post.id]
      );
      refreshed++;
    } catch (err) {
      // One unreachable/deleted tweet must not abort the whole sync.
      console.error(`[tracking] metrics refresh failed for tweet ${post.tweetId}:`, err instanceof Error ? err.message : err);
    }
  }

  return { matched, refreshed };
}

/**
 * Timing uses the author's whole measured history, not the 100-post page the
 * list view shows — it slices the data into far more buckets than the content
 * model does, so it needs every sample available.
 */
export async function loadTimingInsights(
  handle: string,
  tzOffsetMinutes = 0
): Promise<TimingInsights> {
  const rows = await query<{ posted_at: Date | null; views: string | number | null }>(
    `SELECT posted_at, views FROM tracked_posts
     WHERE author_handle = $1 AND views > 0 AND posted_at IS NOT NULL AND is_reply = FALSE
     ORDER BY posted_at DESC LIMIT 2000`,
    [handle.toLowerCase()]
  );
  const samples: TimingSample[] = rows
    .filter((r) => r.posted_at)
    .map((r) => ({ postedAt: r.posted_at!.toISOString(), views: num(r.views) }));
  return buildTimingInsights(samples, tzOffsetMinutes);
}

/**
 * Grades the scorer against everything this author has actually published.
 * Uses the full history rather than the 100-post page, since rank correlation
 * on a truncated sample would answer a different question than the one asked.
 */
export async function loadBacktest(handle: string): Promise<BacktestResult> {
  const rows = await query<{
    predicted_score: number;
    views: string | number | null;
    likes: string | number | null;
    retweets: string | number | null;
    replies: string | number | null;
    quotes: string | number | null;
    bookmarks: string | number | null;
  }>(
    `SELECT predicted_score, views, likes, retweets, replies, quotes, bookmarks
     FROM tracked_posts
     WHERE author_handle = $1 AND views > 0 AND is_reply = FALSE
     ORDER BY created_at DESC LIMIT 2000`,
    [handle.toLowerCase()]
  );

  return backtest(
    rows.map((r) => ({
      predicted: Number(r.predicted_score),
      views: num(r.views),
      engagements:
        num(r.likes) + num(r.replies) + num(r.retweets) + num(r.quotes) + num(r.bookmarks),
    }))
  );
}

export function engagementsOf(m: TrackedPostMetrics): number {
  return m.likes + m.replies + m.retweets + m.quotes + m.bookmarks;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function sideOf(posts: TrackedPost[]): CalibrationSide {
  return {
    n: posts.length,
    medianPredicted: median(posts.map((p) => p.predictedScore)),
    medianViews: median(posts.map((p) => p.metrics!.views)),
    medianEngagements: median(posts.map((p) => engagementsOf(p.metrics!))),
  };
}

/**
 * Pure so it can be reasoned about and tested without a database. Deliberately
 * conservative: medians rather than means (one viral post shouldn't define a
 * "pattern"), and nothing is reported at all below the sample-size gates.
 */
export function buildTrackSummary(posts: TrackedPost[]): TrackSummary {
  const measured = posts.filter((p) => p.metrics !== null);

  const summary: TrackSummary = {
    tracked: posts.length,
    measured: measured.length,
    minimumForInsights: MIN_MEASURED_FOR_INSIGHTS,
    scoreSplit: null,
    fixInsights: [],
  };

  if (measured.length < MIN_MEASURED_FOR_INSIGHTS) return summary;

  // Does a higher predicted score actually correspond to better real numbers?
  // Split at the median predicted score so both halves are comparably sized.
  const byScore = [...measured].sort((a, b) => a.predictedScore - b.predictedScore);
  const mid = Math.floor(byScore.length / 2);
  const lower = byScore.slice(0, mid);
  const higher = byScore.slice(mid);
  if (lower.length > 0 && higher.length > 0) {
    summary.scoreSplit = { lower: sideOf(lower), higher: sideOf(higher) };
  }

  const labels = optimizerRuleLabels();
  const fixIds = new Set(measured.flatMap((p) => p.appliedFixIds));
  for (const fixId of fixIds) {
    const withFix = measured.filter((p) => p.appliedFixIds.includes(fixId));
    const withoutFix = measured.filter((p) => !p.appliedFixIds.includes(fixId));
    if (withFix.length < MIN_PER_SIDE_FOR_FIX || withoutFix.length < MIN_PER_SIDE_FOR_FIX) continue;

    const medianWith = median(withFix.map((p) => engagementsOf(p.metrics!)));
    const medianWithout = median(withoutFix.map((p) => engagementsOf(p.metrics!)));
    summary.fixInsights.push({
      fixId,
      label: labels[fixId] ?? fixId,
      withN: withFix.length,
      withoutN: withoutFix.length,
      medianEngagementsWith: medianWith,
      medianEngagementsWithout: medianWithout,
      // A 0 baseline has no honest ratio (and Infinity would serialize to
      // null through JSON anyway) — report it as uncomputable, not as a
      // spectacular result.
      lift: medianWithout > 0 ? medianWith / medianWithout : null,
    });
  }
  summary.fixInsights.sort((a, b) => (b.lift ?? 0) - (a.lift ?? 0));

  return summary;
}
