import { callVee3Tool } from "./vee3";
import type { AuthorLookupResult, MediaType, TweetImportResult } from "./types";

const RECENT_WINDOW_MS = 3 * 60 * 60 * 1000;

export function parseTweetId(input: string): string | null {
  const trimmed = input.trim();
  if (/^\d{5,25}$/.test(trimmed)) return trimmed;
  const match = trimmed.match(/(?:twitter|x)\.com\/[^/]+\/status(?:es)?\/(\d+)/i);
  return match ? match[1] : null;
}

type Raw = Record<string, any>;

// Vee3's exact response shape isn't published; this reads several plausible
// key spellings per field so minor API differences don't hard-fail the import.
function pick<T>(obj: Raw | undefined | null, paths: string[], fallback: T): T {
  for (const path of paths) {
    const value = path
      .split(".")
      .reduce<any>((acc, key) => (acc == null ? undefined : acc[key]), obj);
    if (value !== undefined && value !== null) return value as T;
  }
  return fallback;
}

function detectMediaType(raw: Raw): MediaType {
  const media = pick<any[]>(
    raw,
    ["media", "extended_entities.media", "attachments.media", "entities.media"],
    []
  );
  if (!Array.isArray(media) || media.length === 0) return "none";
  const type = String(pick<string>(media[0], ["type", "media_type"], "photo")).toLowerCase();
  if (type.includes("video")) return "video";
  if (type.includes("gif") || type.includes("animated")) return "gif";
  return "photo";
}

// Twitter/X API tweet objects mark a reply via in_reply_to_status_id (legacy
// v1.1 shape, null/absent when not a reply) or a "replied_to" entry in
// referenced_tweets (v2 shape). Checking both since Vee3's exact shape isn't
// published — see the module comment above pick().
export function detectIsReply(raw: Raw): boolean {
  const directId = pick<string | number | null>(
    raw,
    ["in_reply_to_status_id", "in_reply_to_status_id_str", "legacy.in_reply_to_status_id_str"],
    null
  );
  if (directId !== null && directId !== undefined && directId !== "") return true;

  const referenced = pick<any[]>(raw, ["referenced_tweets"], []);
  return Array.isArray(referenced) && referenced.some((r) => pick<string>(r, ["type"], "") === "replied_to");
}

export function detectSensitive(raw: Raw): boolean {
  return Boolean(
    pick<boolean>(raw, ["possibly_sensitive", "possibly_sensitive_editable", "legacy.possibly_sensitive"], false)
  );
}

function firstExternalLink(raw: Raw): string {
  const urls = pick<any[]>(raw, ["entities.urls", "urls"], []);
  if (!Array.isArray(urls) || urls.length === 0) return "";
  return pick<string>(urls[0], ["expanded_url", "url"], "");
}

function hoursSincePosted(raw: Raw): number {
  const createdAt = pick<string>(raw, ["created_at", "createdAt", "timestamp"], "");
  if (!createdAt) return 0;
  const ts = new Date(createdAt).getTime();
  if (Number.isNaN(ts)) return 0;
  return Math.max(0, (Date.now() - ts) / (60 * 60 * 1000));
}

async function countRecentPosts(authorHandle: string, authorRestId: string): Promise<number> {
  if (!authorHandle && !authorRestId) return 0;
  try {
    const args = authorHandle ? { user_name: authorHandle } : { rest_id: authorRestId };
    const timeline = await callVee3Tool<Raw>("x-twitter_user_timeline", args);
    const entries = pick<any[]>(timeline, ["entries", "tweets", "timeline", "posts"], []);
    const cutoff = Date.now() - RECENT_WINDOW_MS;
    const count = entries.filter((entry) => {
      const createdAt = pick<string>(entry, ["created_at", "createdAt"], "");
      const ts = createdAt ? new Date(createdAt).getTime() : NaN;
      return !Number.isNaN(ts) && ts >= cutoff;
    }).length;
    return Math.min(10, count);
  } catch {
    return 0;
  }
}

async function fetchRawTweet(url: string): Promise<{ id: string; raw: Raw }> {
  const id = parseTweetId(url);
  if (!id) {
    throw new Error("Couldn't find a tweet id in that link. Paste a full x.com/…/status/… URL.");
  }
  const raw = await callVee3Tool<Raw>("x-twitter_tweet_info", { id });
  return { id, raw };
}

// Factored out of buildResult so the tracking flow can pull fresh numbers for
// an already-known tweet id without re-deriving the whole import result.
// Every plausible spelling per metric. A timeline entry and a tweet-info
// response don't necessarily use the same shape, and a silently-missed field
// reads as a real zero — which is worse than an error, because it quietly
// trains the calibration model on a metric that's always 0. Learned this the
// hard way: 100 backfilled posts all reported 0 likes.
const METRIC_PATHS = {
  views: [
    "views", "view_count", "views_count", "viewCount", "view.count",
    "public_metrics.impression_count", "impression_count", "legacy.view_count",
    "views.count", "ext_views.count",
  ],
  likes: [
    "favorite_count", "like_count", "likes", "favourite_count", "favoriteCount",
    "likeCount", "favorites", "public_metrics.like_count", "stats.like_count",
    "legacy.favorite_count", "legacy.like_count",
  ],
  retweets: [
    "retweet_count", "retweets", "repost_count", "reposts", "retweetCount",
    "public_metrics.retweet_count", "stats.retweet_count", "legacy.retweet_count",
  ],
  replies: [
    "reply_count", "replies", "replyCount", "public_metrics.reply_count",
    "stats.reply_count", "legacy.reply_count",
  ],
  quotes: [
    "quote_count", "quotes", "quoteCount", "quote_tweet_count",
    "public_metrics.quote_count", "stats.quote_count", "legacy.quote_count",
  ],
  bookmarks: [
    "bookmark_count", "bookmarks", "bookmarkCount",
    "public_metrics.bookmark_count", "stats.bookmark_count", "legacy.bookmark_count",
  ],
} as const;

function metric(raw: Raw, paths: readonly string[]): number {
  return Number(pick<number | string>(raw, [...paths], 0)) || 0;
}

export function extractMetrics(raw: Raw): TweetImportResult["realMetrics"] {
  return {
    views: metric(raw, METRIC_PATHS.views),
    likes: metric(raw, METRIC_PATHS.likes),
    retweets: metric(raw, METRIC_PATHS.retweets),
    replies: metric(raw, METRIC_PATHS.replies),
    quotes: metric(raw, METRIC_PATHS.quotes),
    bookmarks: metric(raw, METRIC_PATHS.bookmarks),
  };
}

/**
 * Reports which numeric fields a timeline entry actually carries, so a
 * mis-mapped metric can be diagnosed from a real response instead of guessed
 * at. Returns key paths and values rather than the whole payload — enough to
 * fix a mapping, without dumping post content into logs.
 */
export async function inspectTimelineShape(handle: string): Promise<{
  sampleKeys: string[];
  numericFields: Record<string, number>;
  extracted: TweetImportResult["realMetrics"];
}> {
  const raw = await callVee3Tool<Raw>("x-twitter_user_timeline", { user_name: handle });
  const entries = pick<any[]>(raw, ["entries", "tweets", "timeline", "posts", "data"], []);
  const entry: Raw = Array.isArray(entries) && entries.length > 0 ? entries[0] : {};

  const numericFields: Record<string, number> = {};
  const walk = (obj: Raw, prefix = "", depth = 0) => {
    if (!obj || typeof obj !== "object" || depth > 3) return;
    for (const [key, value] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (typeof value === "number") numericFields[path] = value;
      else if (value && typeof value === "object" && !Array.isArray(value)) walk(value as Raw, path, depth + 1);
    }
  };
  walk(entry);

  return {
    sampleKeys: Object.keys(entry),
    numericFields,
    extracted: extractMetrics(entry),
  };
}

export interface TimelinePost {
  tweetId: string;
  text: string;
  createdAt?: string;
  /** Present on timeline entries, which are full tweet objects — this is what
   * makes a published timeline usable as training data without waiting for
   * drafts to be tracked one at a time. */
  metrics?: TweetImportResult["realMetrics"];
  mediaType?: MediaType;
  isReply?: boolean;
}

function timelineEntryToPost(entry: Raw): TimelinePost {
  return {
    tweetId: String(pick<string | number>(entry, ["id_str", "id", "rest_id", "tweet_id"], "")),
    text: pick<string>(entry, ["text", "full_text"], ""),
    createdAt: pick<string>(entry, ["created_at", "createdAt"], "") || undefined,
    metrics: extractMetrics(entry),
    mediaType: detectMediaType(entry),
    isReply: detectIsReply(entry),
  };
}

/** Recent posts from a handle's profile, shaped for draft matching. */
export async function fetchUserTimeline(handle: string, cursor?: string): Promise<TimelinePost[]> {
  const page = await fetchUserTimelinePage(handle, cursor);
  return page.posts;
}

/**
 * One page of a handle's timeline, plus the cursor for the next one. Kept
 * separate from fetchUserTimeline so the draft-matching path can stay a
 * single cheap call while backfill walks the whole history.
 */
export async function fetchUserTimelinePage(
  handle: string,
  cursor?: string
): Promise<{ posts: TimelinePost[]; nextCursor?: string }> {
  const args: Record<string, unknown> = { user_name: handle };
  if (cursor) args.cursor = cursor;

  const raw = await callVee3Tool<Raw>("x-twitter_user_timeline", args);
  const entries = pick<any[]>(raw, ["entries", "tweets", "timeline", "posts", "data"], []);
  const nextCursor =
    pick<string>(raw, ["next_cursor", "nextCursor", "cursor", "next"], "") || undefined;

  if (!Array.isArray(entries)) return { posts: [], nextCursor: undefined };

  const posts = entries
    .map(timelineEntryToPost)
    .filter((t) => t.tweetId && t.tweetId !== "undefined" && t.text);

  return { posts, nextCursor };
}

/** Fresh engagement numbers for one already-identified tweet. */
export async function fetchTweetMetrics(tweetId: string): Promise<TweetImportResult["realMetrics"]> {
  const raw = await callVee3Tool<Raw>("x-twitter_tweet_info", { id: tweetId });
  return extractMetrics(raw);
}

function buildResult(raw: Raw, recentPostsCount: number): TweetImportResult {
  const text = pick<string>(raw, ["text", "full_text"], "");
  if (!text) {
    throw new Error("Vee3 didn't return any post text for that link.");
  }

  const author = pick<Raw>(raw, ["author", "user"], {});

  return {
    text,
    mediaType: detectMediaType(raw),
    link: firstExternalLink(raw),
    isReply: detectIsReply(raw),
    nsfw: detectSensitive(raw),
    authorHandle: pick<string>(author, ["user_name", "screen_name", "username"], ""),
    authorName: pick<string>(author, ["name", "display_name"], ""),
    authorFollowers: Number(pick<number>(author, ["followers_count", "followers", "sub_count"], 0)) || 0,
    authorVerified: Boolean(pick<boolean>(author, ["is_blue_verified", "blue_verified", "verified"], false)),
    recentPostsCount,
    postedHoursAgo: hoursSincePosted(raw),
    realMetrics: extractMetrics(raw),
  };
}

export function parseHandle(input: string): string | null {
  const trimmed = input.trim().replace(/^@/, "");
  if (/^[A-Za-z0-9_]{1,15}$/.test(trimmed)) return trimmed;
  const match = trimmed.match(/(?:twitter|x)\.com\/([A-Za-z0-9_]{1,15})\/?$/i);
  return match ? match[1] : null;
}

export async function lookupAuthor(handleInput: string): Promise<AuthorLookupResult> {
  const handle = parseHandle(handleInput);
  if (!handle) {
    throw new Error("That doesn't look like a valid @handle.");
  }

  const raw = await callVee3Tool<Raw>("x-twitter_user_info", { user_name: handle });
  // The account object may come back nested (user/data/result) or flat,
  // depending on the exact shape Vee3 returns for this tool. Confirmed flat
  // against a real response, with the handle under "profile" (not
  // user_name/screen_name/username like the tweet-info author object uses).
  const user = pick<Raw>(raw, ["user", "data", "result"], raw);

  const authorHandle = pick<string>(user, ["profile", "user_name", "screen_name", "username"], "");
  if (!authorHandle) {
    throw new Error("Vee3 didn't return account info for that handle.");
  }

  return {
    authorHandle,
    authorName: pick<string>(user, ["name", "display_name"], ""),
    authorFollowers: Number(pick<number>(user, ["followers_count", "followers", "sub_count"], 0)) || 0,
    authorVerified: Boolean(pick<boolean>(user, ["is_blue_verified", "blue_verified", "verified"], false)),
  };
}

export async function importTweet(url: string): Promise<TweetImportResult> {
  const { raw } = await fetchRawTweet(url);

  const author = pick<Raw>(raw, ["author", "user"], {});
  const authorHandle = pick<string>(author, ["user_name", "screen_name", "username"], "");
  const authorRestId = pick<string>(author, ["rest_id", "id"], "");
  const recentPostsCount = await countRecentPosts(authorHandle, authorRestId);

  return buildResult(raw, recentPostsCount);
}
