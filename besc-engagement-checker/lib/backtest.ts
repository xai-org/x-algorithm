/**
 * Measures whether a scoring approach actually predicts real outcomes.
 *
 * This exists so new estimators can be judged instead of assumed. Anyone can
 * attach an LLM to a draft and emit 19 confident-looking action probabilities;
 * the hard part is knowing whether those numbers beat the ones already there.
 * With a real published history and its real engagement, that becomes a
 * measurable question rather than a matter of taste.
 *
 * Rank correlation rather than R², because the score's job is ordering drafts
 * — "is this one better than that one" — not predicting a view count. A scorer
 * can be badly calibrated in absolute terms and still be genuinely useful if
 * it ranks correctly, and rank correlation is also far more robust to the
 * heavy tail that one viral post creates.
 */

export interface BacktestSample {
  /** What the scorer predicted, before the outcome was known. */
  predicted: number;
  views: number;
  engagements: number;
}

export interface BacktestResult {
  n: number;
  minimumForVerdict: number;
  /** Spearman rank correlation against views, in [-1, 1]. */
  viewsCorrelation: number | null;
  /** Against engagements-per-view, which is closer to what wording controls. */
  engagementRateCorrelation: number | null;
  /** Plain-language read of the strongest signal found. */
  verdict: "predictive" | "weak" | "none" | "inverted" | "insufficient";
}

// Rank correlation on small samples is extremely noisy; below this the number
// would be an opinion with a decimal point on it.
export const MIN_SAMPLES_FOR_BACKTEST = 25;
// Thresholds are deliberately unambitious. Social engagement has enormous
// irreducible variance — timing, who happened to reshare, what else was
// happening — so a rank correlation of 0.3 is a genuinely useful signal here,
// not a weak one.
const PREDICTIVE = 0.3;
const WEAK = 0.12;

/** Average ranks, so ties don't distort the correlation. */
function rank(values: number[]): number[] {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(values.length).fill(0);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1].v === order[i].v) j++;
    const shared = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k].i] = shared;
    i = j + 1;
  }
  return ranks;
}

function spearman(xs: number[], ys: number[]): number | null {
  if (xs.length < 3) return null;
  const rx = rank(xs);
  const ry = rank(ys);
  const n = rx.length;
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const mx = mean(rx);
  const my = mean(ry);

  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    num += (rx[i] - mx) * (ry[i] - my);
    dx += (rx[i] - mx) ** 2;
    dy += (ry[i] - my) ** 2;
  }
  // Zero variance means every prediction was identical — that isn't a
  // correlation of 0, it's the absence of a usable signal.
  if (dx === 0 || dy === 0) return null;
  return num / Math.sqrt(dx * dy);
}

export function backtest(samples: BacktestSample[]): BacktestResult {
  const usable = samples.filter((s) => s.views > 0 && Number.isFinite(s.predicted));

  const result: BacktestResult = {
    n: usable.length,
    minimumForVerdict: MIN_SAMPLES_FOR_BACKTEST,
    viewsCorrelation: null,
    engagementRateCorrelation: null,
    verdict: "insufficient",
  };
  if (usable.length < MIN_SAMPLES_FOR_BACKTEST) return result;

  const predicted = usable.map((s) => s.predicted);
  result.viewsCorrelation = spearman(predicted, usable.map((s) => s.views));
  result.engagementRateCorrelation = spearman(
    predicted,
    usable.map((s) => s.engagements / s.views)
  );

  const best = Math.max(result.viewsCorrelation ?? 0, result.engagementRateCorrelation ?? 0);
  const worst = Math.min(result.viewsCorrelation ?? 0, result.engagementRateCorrelation ?? 0);

  if (result.viewsCorrelation === null && result.engagementRateCorrelation === null) {
    result.verdict = "none";
  } else if (best >= PREDICTIVE) {
    result.verdict = "predictive";
  } else if (best >= WEAK) {
    result.verdict = "weak";
  } else if (worst <= -WEAK) {
    // Consistently backwards is a finding in its own right, and a much more
    // actionable one than "no signal" — it means the scorer is penalising
    // something this author's audience actually rewards.
    result.verdict = "inverted";
  } else {
    result.verdict = "none";
  }

  return result;
}

/**
 * The strongest usable signal a result found. A null correlation means no
 * signal was measurable, which is 0 — emphatically not a negative score. An
 * earlier version treated null as -1, which let a challenger that ranked
 * outcomes *backwards* count as a win over a baseline that simply found
 * nothing. Backwards is worse than nothing, not better.
 */
function signalStrength(r: BacktestResult): number {
  return Math.max(r.viewsCorrelation ?? 0, r.engagementRateCorrelation ?? 0);
}

// Rank correlations wobble by a few hundredths on samples this size, so a
// hairline lead isn't evidence of anything.
const WIN_MARGIN = 0.05;

/** Head-to-head, so a proposed estimator has to beat the incumbent on the same posts. */
export function compareEstimators(
  baseline: BacktestSample[],
  challenger: BacktestSample[]
): { baseline: BacktestResult; challenger: BacktestResult; challengerWins: boolean } {
  const b = backtest(baseline);
  const c = backtest(challenger);

  // Three conditions, all required. The challenger has to have found a real
  // signal of its own, beat the incumbent by more than noise, and not be
  // ranking outcomes backwards on the other measure while it does it.
  const challengerWins =
    c.verdict !== "insufficient" &&
    c.verdict !== "inverted" &&
    signalStrength(c) >= WEAK &&
    signalStrength(c) >= signalStrength(b) + WIN_MARGIN;

  return { baseline: b, challenger: c, challengerWins };
}
