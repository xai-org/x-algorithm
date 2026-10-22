import { Pool } from "pg";

// Tracking is strictly additive: everything else in this app (scoring,
// optimizing, generating, importing) is stateless and must keep working
// untouched when no database is attached. Every caller checks this first
// rather than assuming a pool exists, so deploying this code before
// provisioning Postgres degrades to "tracking is off", not a broken app.
export function dbConfigured(): boolean {
  return Boolean(process.env.DATABASE_URL);
}

// Next.js hot-reloads modules in dev, which would otherwise leak a new pool
// (and its connections) on every reload until Postgres refuses more.
const globalForDb = globalThis as unknown as { bescPool?: Pool; bescSchemaReady?: Promise<void> };

function pool(): Pool {
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is not configured");
  }
  if (!globalForDb.bescPool) {
    globalForDb.bescPool = new Pool({
      connectionString: process.env.DATABASE_URL,
      // Railway's private network (*.railway.internal) doesn't use TLS, but
      // any external/public Postgres URL will, and its cert won't chain to a
      // root this container trusts. Managed providers universally expect
      // this flag rather than a custom CA bundle.
      ssl: /railway\.internal|localhost|127\.0\.0\.1/.test(process.env.DATABASE_URL)
        ? undefined
        : { rejectUnauthorized: false },
      max: 5,
      connectionTimeoutMillis: 10000,
      idleTimeoutMillis: 30000,
    });
  }
  return globalForDb.bescPool;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS tracked_posts (
  id                SERIAL PRIMARY KEY,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  author_handle     TEXT        NOT NULL,
  draft_text        TEXT        NOT NULL,
  predicted_score   REAL        NOT NULL,
  predicted_grade   TEXT        NOT NULL,
  media_type        TEXT        NOT NULL DEFAULT 'none',
  is_reply          BOOLEAN     NOT NULL DEFAULT FALSE,
  has_mutual_audience BOOLEAN   NOT NULL DEFAULT FALSE,
  is_verified       BOOLEAN     NOT NULL DEFAULT FALSE,
  recent_posts_count INTEGER    NOT NULL DEFAULT 0,
  applied_fix_ids   TEXT[]      NOT NULL DEFAULT '{}',
  tweet_id          TEXT,
  matched_at        TIMESTAMPTZ,
  posted_at         TIMESTAMPTZ,
  metrics_updated_at TIMESTAMPTZ,
  views             BIGINT,
  likes             BIGINT,
  retweets          BIGINT,
  replies           BIGINT,
  quotes            BIGINT,
  bookmarks         BIGINT
);
-- Added after the table shipped, so it needs an ALTER: CREATE TABLE IF NOT
-- EXISTS won't touch an existing table. Distinguishes drafts the user tracked
-- by hand from posts backfilled out of their published timeline.
ALTER TABLE tracked_posts ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'tracked';

CREATE INDEX IF NOT EXISTS tracked_posts_handle_idx ON tracked_posts (author_handle, created_at DESC);

-- One fitted model per handle, refreshed after a backfill or sync. Cached
-- here rather than refitted per request: fitting walks every post the handle
-- has, which is far too much work for a keystroke-latency scoring call.
CREATE TABLE IF NOT EXISTS calibration_models (
  handle    TEXT PRIMARY KEY,
  model     JSONB       NOT NULL,
  fitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Partial unique index, not a plain one: many rows legitimately sit with a
-- NULL tweet_id (drafts saved but not yet published), and those must not
-- collide with each other — but one published tweet must never be claimed
-- by two different tracked drafts.
CREATE UNIQUE INDEX IF NOT EXISTS tracked_posts_tweet_idx ON tracked_posts (tweet_id) WHERE tweet_id IS NOT NULL;

-- The result of testing an alternative probability estimator against this
-- author's real outcomes. Stored because each run costs real model calls, so
-- the verdict has to outlive the request that paid for it.
CREATE TABLE IF NOT EXISTS estimator_evaluations (
  handle       TEXT PRIMARY KEY,
  evaluation   JSONB       NOT NULL,
  evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`;

// Cached on globalThis so concurrent requests share one in-flight schema
// check instead of each firing their own CREATE TABLE IF NOT EXISTS.
function ensureSchema(): Promise<void> {
  if (!globalForDb.bescSchemaReady) {
    globalForDb.bescSchemaReady = pool()
      .query(SCHEMA)
      .then(() => undefined)
      .catch((err) => {
        // Don't cache a failure — a transient connection error at boot would
        // otherwise permanently poison tracking for this process.
        globalForDb.bescSchemaReady = undefined;
        throw err;
      });
  }
  return globalForDb.bescSchemaReady;
}

export async function query<T = Record<string, unknown>>(
  text: string,
  params: unknown[] = []
): Promise<T[]> {
  await ensureSchema();
  const result = await pool().query(text, params);
  return result.rows as T[];
}
