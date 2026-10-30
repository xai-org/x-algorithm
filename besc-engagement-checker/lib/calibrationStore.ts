import { query } from "./db";
import { analyzePost, extractFeatures } from "./scoring";
import { fetchTweetMetrics, fetchUserTimelinePage } from "./twitter-import";
import {
  type CalibrationModel,
  type CalibrationSample,
  featureVector,
  fitCalibration,
} from "./calibration";
import type { MediaType } from "./types";

// Enough pages to build a real training set without hammering the API. Each
// page is one Vee3 call, and timelines return roughly 20-40 posts per page.
const MAX_BACKFILL_PAGES = 12;
const BACKFILL_TARGET_POSTS = 400;
const EMPTY_POSTS: Awaited<ReturnType<typeof fetchUserTimelinePage>>["posts"] = [];

export interface BackfillResult {
  fetched: number;
  inserted: number;
  /** Already-known posts whose numbers (and stale predicted scores) were updated. */
  refreshed: number;
  /** Posts whose timeline metrics looked wrong and were re-read from tweet-info. */
  enriched: number;
  usable: number;
}

/**
 * Timeline entries and tweet-info responses don't carry engagement in the same
 * shape, and a field we fail to find reads as a genuine zero. A real backfill
 * recorded 0 likes across an entire 200-post history while replies and reposts
 * came through — an impossible combination for a post with real engagement,
 * and one that silently poisons the model with an always-zero metric.
 *
 * Rather than keep guessing key names from the outside, treat that pattern as
 * a failed read and re-fetch from tweet-info, which is known to return likes
 * correctly (it's what the imported-post stats display).
 */
function metricsLookIncomplete(m: { likes: number; replies: number; retweets: number; bookmarks: number }): boolean {
  return m.likes === 0 && m.replies + m.retweets + m.bookmarks > 0;
}

// Bounded concurrency: enrichment can mean a few hundred calls, which is far
// too slow sequentially for a request, and unbounded would hammer the API.
async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (cursor < items.length) {
        const index = cursor++;
        results[index] = await fn(items[index]);
      }
    })
  );
  return results;
}

/**
 * Walks a handle's published timeline and stores each post as an
 * already-measured row. This is what makes calibration practical: a user's
 * own history is hundreds of real (post → outcome) pairs that already exist,
 * so the model doesn't have to wait for drafts to be tracked one at a time.
 *
 * Only the author's own original posts are kept — replies live under
 * completely different distribution rules (excluded from out-of-network
 * recommendations entirely), so mixing them in would teach the model that
 * whatever replies happen to look like suppresses reach.
 */
export async function backfillFromTimeline(
  handle: string,
  // Injectable for the same reason syncTrackedPosts takes its fetchers: it
  // makes the insert/pagination logic testable without live network calls.
  fetchPage: typeof fetchUserTimelinePage = fetchUserTimelinePage,
  fetchMetrics: typeof fetchTweetMetrics = fetchTweetMetrics
): Promise<BackfillResult> {
  const normalized = handle.toLowerCase();
  let cursor: string | undefined;
  let fetched = 0;
  let inserted = 0;
  let refreshed = 0;
  let enriched = 0;
  const seenCursors = new Set<string>();

  const collected: typeof EMPTY_POSTS = [];
  for (let page = 0; page < MAX_BACKFILL_PAGES && fetched < BACKFILL_TARGET_POSTS; page++) {
    const { posts, nextCursor } = await fetchPage(normalized, cursor);
    if (posts.length === 0) break;
    fetched += posts.length;
    collected.push(...posts);

    // A missing or repeating cursor means there's no more history to walk;
    // without this guard a self-referential cursor would loop until the page cap.
    if (!nextCursor || seenCursors.has(nextCursor)) break;
    seenCursors.add(nextCursor);
    cursor = nextCursor;
  }

  // Repair any post whose timeline metrics look like a failed field read,
  // before any of it reaches the model.
  const needsEnrichment = collected.filter(
    (p) => !p.isReply && p.metrics && metricsLookIncomplete(p.metrics)
  );
  if (needsEnrichment.length > 0) {
    await mapWithConcurrency(needsEnrichment, 6, async (post) => {
      try {
        const fresh = await fetchMetrics(post.tweetId);
        // Only accept the re-read if it actually resolved the problem;
        // otherwise keep what we had rather than overwrite with another zero.
        if (!metricsLookIncomplete(fresh)) {
          post.metrics = fresh;
          enriched++;
        }
      } catch {
        // One unreadable post must not fail the whole backfill.
      }
    });
  }

  {
    const posts = collected;
    for (const post of posts) {
      if (post.isReply) continue;
      const metrics = post.metrics;
      // No views means no measurable rate — storing it would look like data
      // while contributing nothing but a zero-denominator row.
      if (!metrics || metrics.views <= 0) continue;

      // Score the post as the tool would have scored it before publishing.
      // Backfilled rows previously stored 0 here, which made the "does the
      // score predict reach?" comparison split a column of identical zeroes —
      // it reported two halves at "score ~0" and compared noise. With a real
      // predicted score this becomes an actual backtest of the scorer against
      // outcomes it never saw.
      const predicted = analyzePost({
        text: post.text,
        mediaType: post.mediaType ?? "none",
        link: "",
        isReply: false,
        hasMutualFollowAudience: false,
        recentPostsCount: 0,
        nsfw: false,
      });

      const postedAt = post.createdAt ? new Date(post.createdAt) : null;
      const rows = await query<{ inserted: boolean }>(
        `INSERT INTO tracked_posts
           (author_handle, draft_text, predicted_score, predicted_grade, applied_fix_ids,
            media_type, is_reply, source, tweet_id, matched_at, posted_at,
            metrics_updated_at, views, likes, retweets, replies, quotes, bookmarks)
         VALUES ($1,$2,$13,$14,$3,$4,FALSE,'backfill',$5,NOW(),$6,NOW(),$7,$8,$9,$10,$11,$12)
         -- tracked_posts_tweet_idx is a PARTIAL unique index (WHERE tweet_id
         -- IS NOT NULL, so unpublished drafts don't collide on NULL). Postgres
         -- will not infer a partial index for ON CONFLICT unless the statement
         -- repeats its predicate — without this it fails outright with
         -- "no unique or exclusion constraint matching the ON CONFLICT
         -- specification".
         -- Refresh rather than skip. Engagement keeps accruing after a post
         -- is first seen, and a re-run also repairs rows written by an older
         -- version (the first backfill stored predicted_score = 0, which made
         -- the calibration split compare a column of zeroes). predicted_score
         -- is only overwritten for backfilled rows — on a manually tracked
         -- draft it's the real pre-publish prediction and must survive.
         ON CONFLICT (tweet_id) WHERE tweet_id IS NOT NULL DO UPDATE SET
           views = EXCLUDED.views,
           likes = EXCLUDED.likes,
           retweets = EXCLUDED.retweets,
           replies = EXCLUDED.replies,
           quotes = EXCLUDED.quotes,
           bookmarks = EXCLUDED.bookmarks,
           metrics_updated_at = NOW(),
           media_type = EXCLUDED.media_type,
           predicted_score = CASE WHEN tracked_posts.source = 'backfill'
                                  THEN EXCLUDED.predicted_score ELSE tracked_posts.predicted_score END,
           predicted_grade = CASE WHEN tracked_posts.source = 'backfill'
                                  THEN EXCLUDED.predicted_grade ELSE tracked_posts.predicted_grade END
         -- xmax = 0 distinguishes a fresh insert from an update, so the
         -- reported counts stay truthful on a re-run.
         RETURNING (xmax = 0) AS inserted`,
        [
          normalized,
          post.text,
          [],
          post.mediaType ?? "none",
          post.tweetId,
          postedAt && !Number.isNaN(postedAt.getTime()) ? postedAt : null,
          metrics.views,
          metrics.likes,
          metrics.retweets,
          metrics.replies,
          metrics.quotes,
          metrics.bookmarks,
          predicted.score,
          predicted.grade,
        ]
      );
      if (rows[0]?.inserted) inserted++;
      else if (rows.length > 0) refreshed++;
    }
  }

  const usable = await countUsableSamples(normalized);
  return { fetched, inserted, refreshed, enriched, usable };
}

async function countUsableSamples(handle: string): Promise<number> {
  const rows = await query<{ count: string }>(
    `SELECT COUNT(*) AS count FROM tracked_posts
     WHERE author_handle = $1 AND views > 0 AND is_reply = FALSE`,
    [handle.toLowerCase()]
  );
  return Number(rows[0]?.count ?? 0);
}

interface SampleRow {
  draft_text: string;
  media_type: string;
  views: string | number;
  likes: string | number;
  retweets: string | number;
  replies: string | number;
  quotes: string | number;
  bookmarks: string | number;
}

/**
 * Refits from every measured post the handle has and caches the result.
 * Features are re-derived from the stored text rather than persisted, so a
 * change to feature extraction automatically applies to the whole history on
 * the next refit instead of leaving stale vectors behind.
 */
export async function refitCalibration(handle: string): Promise<CalibrationModel> {
  const normalized = handle.toLowerCase();
  const rows = await query<SampleRow>(
    `SELECT draft_text, media_type, views, likes, retweets, replies, quotes, bookmarks
     FROM tracked_posts
     WHERE author_handle = $1 AND views > 0 AND is_reply = FALSE
     ORDER BY created_at DESC LIMIT 1000`,
    [normalized]
  );

  const samples: CalibrationSample[] = rows.map((row) => {
    const features = extractFeatures(row.draft_text, "");
    return {
      features: featureVector(features, row.media_type as MediaType),
      views: Number(row.views),
      counts: {
        favorite: Number(row.likes ?? 0),
        reply: Number(row.replies ?? 0),
        retweet: Number(row.retweets ?? 0),
        quote: Number(row.quotes ?? 0),
        bookmark: Number(row.bookmarks ?? 0),
      },
    };
  });

  const model = fitCalibration(normalized, samples);
  await query(
    `INSERT INTO calibration_models (handle, model, fitted_at)
     VALUES ($1, $2, NOW())
     ON CONFLICT (handle) DO UPDATE SET model = EXCLUDED.model, fitted_at = NOW()`,
    [normalized, JSON.stringify(model)]
  );
  return model;
}

// Scoring runs on every keystroke (debounced), so this must not hit the
// database each time. The model only changes on backfill/sync, and a stale
// read for a minute is harmless.
const CACHE_TTL_MS = 60_000;
const cache = new Map<string, { model: CalibrationModel | null; at: number }>();

export async function loadCalibration(handle: string): Promise<CalibrationModel | null> {
  const normalized = handle.trim().toLowerCase();
  if (!normalized) return null;

  const cached = cache.get(normalized);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.model;

  try {
    const rows = await query<{ model: CalibrationModel }>(
      `SELECT model FROM calibration_models WHERE handle = $1`,
      [normalized]
    );
    const model = rows[0]?.model ?? null;
    cache.set(normalized, { model, at: Date.now() });
    return model;
  } catch {
    // Calibration is an enhancement; if it can't be read, scoring proceeds on
    // the heuristics exactly as it always has.
    cache.set(normalized, { model: null, at: Date.now() });
    return null;
  }
}

export function invalidateCalibrationCache(handle: string): void {
  cache.delete(handle.trim().toLowerCase());
}
