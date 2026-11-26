import { query } from "./db";
import { analyzePost } from "./scoring";
import { compareEstimators, type BacktestResult, type BacktestSample } from "./backtest";
import { estimateActionMultipliersBatch } from "./llmProbabilities";
import type { MediaType } from "./types";

/**
 * Head-to-head test of the language-model probability head against the
 * heuristic one, on posts this author actually published.
 *
 * The point of this file is that the LLM head does not get adopted because it
 * sounds like a good idea. It gets adopted if it ranks real outcomes better
 * than what's already here, measured on the same posts, through the same
 * scorer, with only the probabilities swapped. If it loses, that's the finding
 * and it's worth just as much — an estimator that sounds smart and predicts
 * nothing is exactly what this tool is supposed to catch.
 *
 * Evaluations are stored so the answer survives the request that produced it:
 * each run costs real API calls, and a verdict nobody can see later would have
 * to be bought again every time someone asks.
 */

// Enough posts for a rank correlation to mean something, few enough that a run
// is a handful of batched calls rather than a bill.
const DEFAULT_SAMPLE = 60;
const MAX_SAMPLE = 120;

export interface EstimatorEvaluation {
  handle: string;
  evaluatedAt: string;
  /** Posts that made it through with a usable estimate from both sides. */
  n: number;
  heuristic: BacktestResult;
  llm: BacktestResult;
  /** True only when the LLM head ranks outcomes better than the heuristic. */
  llmWins: boolean;
  /** How many sampled posts the model failed to return a usable estimate for. */
  skipped: number;
}

interface EvaluationRow {
  draft_text: string;
  media_type: string;
  views: string | number | null;
  likes: string | number | null;
  retweets: string | number | null;
  replies: string | number | null;
  quotes: string | number | null;
  bookmarks: string | number | null;
}

function num(value: string | number | null): number {
  return value === null ? 0 : Number(value);
}

/** Injectable so the comparison can be exercised without spending model calls. */
export interface EvaluationDeps {
  estimate: typeof estimateActionMultipliersBatch;
}

export async function runEvaluation(
  handle: string,
  limit = DEFAULT_SAMPLE,
  deps: EvaluationDeps = { estimate: estimateActionMultipliersBatch }
): Promise<EstimatorEvaluation> {
  const size = Math.max(1, Math.min(MAX_SAMPLE, limit));
  const rows = await query<EvaluationRow>(
    `SELECT draft_text, media_type, views, likes, retweets, replies, quotes, bookmarks
       FROM tracked_posts
      WHERE author_handle = $1 AND views > 0 AND is_reply = FALSE
      ORDER BY posted_at DESC NULLS LAST, created_at DESC
      LIMIT $2`,
    [handle.toLowerCase(), size]
  );

  const multipliers = await deps.estimate(
    rows.map((r) => ({ text: r.draft_text, mediaType: r.media_type || "none" }))
  );

  const heuristicSamples: BacktestSample[] = [];
  const llmSamples: BacktestSample[] = [];
  let skipped = 0;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const estimate = multipliers[i];
    // A post the model didn't return an estimate for is dropped from BOTH
    // sides. Scoring it heuristically on the LLM side would quietly credit the
    // heuristic's work to the challenger and make the comparison meaningless.
    if (!estimate) {
      skipped++;
      continue;
    }

    const req = {
      text: row.draft_text,
      mediaType: (row.media_type || "none") as MediaType,
      link: "",
      isReply: false,
      hasMutualFollowAudience: false,
      recentPostsCount: 0,
      nsfw: false,
    };
    const outcome = {
      views: num(row.views),
      engagements:
        num(row.likes) + num(row.replies) + num(row.retweets) + num(row.quotes) + num(row.bookmarks),
    };

    // No calibration on either side: this is a test of the probability head,
    // and letting a fitted model sit underneath one arm would measure the
    // wrong thing.
    heuristicSamples.push({ predicted: analyzePost(req).score, ...outcome });
    llmSamples.push({ predicted: analyzePost(req, null, estimate).score, ...outcome });
  }

  const comparison = compareEstimators(heuristicSamples, llmSamples);
  const evaluation: EstimatorEvaluation = {
    handle: handle.toLowerCase(),
    evaluatedAt: new Date().toISOString(),
    n: llmSamples.length,
    heuristic: comparison.baseline,
    llm: comparison.challenger,
    // A tie goes to the incumbent: the LLM head costs a call per draft, so
    // "no worse" isn't a reason to switch. compareEstimators enforces that.
    llmWins: comparison.challengerWins,
    skipped,
  };

  await saveEvaluation(evaluation);
  return evaluation;
}

async function saveEvaluation(evaluation: EstimatorEvaluation): Promise<void> {
  await query(
    `INSERT INTO estimator_evaluations (handle, evaluation, evaluated_at)
     VALUES ($1, $2, NOW())
     ON CONFLICT (handle) DO UPDATE
       SET evaluation = EXCLUDED.evaluation, evaluated_at = EXCLUDED.evaluated_at`,
    [evaluation.handle, JSON.stringify(evaluation)]
  );
}

export async function loadEvaluation(handle: string): Promise<EstimatorEvaluation | null> {
  const rows = await query<{ evaluation: EstimatorEvaluation }>(
    `SELECT evaluation FROM estimator_evaluations WHERE handle = $1`,
    [handle.toLowerCase()]
  );
  return rows[0]?.evaluation ?? null;
}
