import type { FeatureReport, MediaType } from "./types";

/**
 * A small, honest stand-in for the part of Phoenix we can actually reproduce.
 *
 * Phoenix predicts P(action | post, viewer) using a specific viewer's
 * engagement history. We have no viewer — an unpublished draft has no audience
 * — and no API exposes the private history of everyone who might see a post,
 * so that model is out of reach by construction, not by lack of access.
 *
 * What we *can* do is the same shape of thing at the aggregate level: learn
 * P(action | post) from real observed outcomes, then blend the predicted
 * per-action rates with the same production weights the real RankingScorer
 * uses. Views and per-action counts are both observable on published posts,
 * so `likes / views` is a directly measured action rate — real ground truth,
 * not a guess.
 *
 * Everything here is deliberately conservative: ridge regression rather than
 * anything fancy, cross-validated before it's trusted at all, and shrunk
 * toward the hand-tuned heuristics in proportion to how much data backs it.
 * A confidently wrong personalised score would be worse than the honest
 * heuristic it replaces.
 */

/** Actions whose real counts X exposes, so they can be learned rather than guessed. */
export const OBSERVABLE_ACTIONS = ["favorite", "reply", "retweet", "quote", "bookmark"] as const;
export type ObservableAction = (typeof OBSERVABLE_ACTIONS)[number];

export const FEATURE_NAMES = [
  "words",
  "hashtags",
  "mentions",
  "links",
  "emojis",
  "hasNumbers",
  "hookSpecific",
  "hookGeneric",
  "hasMedia",
  "isVideo",
  "allCapsRatio",
  "fillerWords",
  "threadMarker",
] as const;

/** Bounded and roughly unit-scaled so ridge's penalty means the same thing across features. */
export function featureVector(f: FeatureReport, mediaType: MediaType): number[] {
  return [
    Math.min(f.words, 100) / 50,
    Math.min(f.hashtags, 6),
    Math.min(f.mentions, 6),
    Math.min(f.links, 2),
    Math.min(f.emojis, 6) / 3,
    f.hasNumbers ? 1 : 0,
    f.replyHookKind === "specific" ? 1 : 0,
    f.replyHookKind === "generic" ? 1 : 0,
    mediaType !== "none" ? 1 : 0,
    mediaType === "video" ? 1 : 0,
    f.allCapsWordRatio,
    Math.min(f.fillerWords, 6) / 6,
    f.hasThreadMarker ? 1 : 0,
  ];
}

export interface CalibrationSample {
  features: number[];
  views: number;
  counts: Record<ObservableAction, number>;
}

export interface ActionModel {
  /** Intercept first, then one coefficient per FEATURE_NAMES entry. */
  coefficients: number[];
  means: number[];
  stds: number[];
  /** Cross-validated R². At or below zero the fit is worse than the mean and is not used. */
  cvR2: number;
  /** Mean observed rate, per 1,000 views — the fallback this model has to beat. */
  baselineRate: number;
}

export interface CalibrationModel {
  handle: string;
  n: number;
  fittedAt: string;
  actions: Partial<Record<ObservableAction, ActionModel>>;
}

// Below this there isn't enough data for a 13-feature fit to mean anything —
// it would memorise noise and report it as insight.
export const MIN_SAMPLES_FOR_FIT = 40;
// A fit has to explain real out-of-sample variance, not just curve-fit the
// training rows, before it's allowed anywhere near a user's score.
const MIN_CV_R2 = 0.05;
const RIDGE_LAMBDA = 1.0;
// Controls how fast trust shifts from heuristic to fitted: at n = SHRINKAGE_K
// the blend is 50/50.
const SHRINKAGE_K = 120;
// The cross-validated R² at which a fit is trusted as much as its sample size
// allows. Below it, influence scales down proportionally — barely clearing the
// MIN_CV_R2 gate earns barely any say in the score.
const STRONG_FIT_R2 = 0.3;

/** Rates are small, skewed and heavy-tailed; log space keeps one viral post from dominating. */
function toTarget(count: number, views: number): number {
  return Math.log1p((count / Math.max(views, 1)) * 1000);
}
function fromTarget(value: number): number {
  return Math.expm1(Math.max(0, value)) / 1000;
}

function standardize(rows: number[][]): { z: number[][]; means: number[]; stds: number[] } {
  const p = rows[0].length;
  const means = new Array(p).fill(0);
  const stds = new Array(p).fill(1);

  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (const row of rows) sum += row[j];
    means[j] = sum / rows.length;

    let variance = 0;
    for (const row of rows) variance += (row[j] - means[j]) ** 2;
    // A constant feature has no information; leaving std at 1 zeroes it out
    // rather than dividing by ~0 and exploding its coefficient.
    stds[j] = Math.sqrt(variance / rows.length) || 1;
  }

  const z = rows.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
  return { z, means, stds };
}

/** Ridge via normal equations: (XᵀX + λI)β = Xᵀy, with the intercept unpenalised. */
function ridgeSolve(z: number[][], y: number[], lambda: number): number[] {
  const n = z.length;
  const p = z[0].length + 1; // + intercept
  const X = z.map((row) => [1, ...row]);

  const xtx: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const xty: number[] = new Array(p).fill(0);

  for (let i = 0; i < n; i++) {
    for (let a = 0; a < p; a++) {
      xty[a] += X[i][a] * y[i];
      for (let b = a; b < p; b++) xtx[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) xtx[a][b] = xtx[b][a];
  for (let a = 1; a < p; a++) xtx[a][a] += lambda;

  // Gaussian elimination with partial pivoting.
  const aug = xtx.map((row, i) => [...row, xty[i]]);
  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let r = col + 1; r < p; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
    if (Math.abs(aug[pivot][col]) < 1e-12) continue; // singular; leave this coefficient at 0
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];

    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const factor = aug[r][col] / aug[col][col];
      if (!factor) continue;
      for (let c = col; c <= p; c++) aug[r][c] -= factor * aug[col][c];
    }
  }

  return aug.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[p] / row[i]));
}

function predictStandardized(coefficients: number[], zRow: number[]): number {
  let out = coefficients[0];
  for (let j = 0; j < zRow.length; j++) out += coefficients[j + 1] * zRow[j];
  return out;
}

/**
 * k-fold cross-validated R². This is the gate that decides whether a fitted
 * model is used at all, so it has to measure out-of-sample performance —
 * in-sample R² would happily approve a model that memorised the training set.
 */
function crossValidatedR2(rows: number[][], y: number[], lambda: number, k = 5): number {
  const n = rows.length;
  if (n < k * 2) return 0;

  const indices = rows.map((_, i) => i);
  // Deterministic interleave rather than a random shuffle, so a fit is
  // reproducible from the same data.
  const folds: number[][] = Array.from({ length: k }, () => []);
  indices.forEach((idx, i) => folds[i % k].push(idx));

  let ssRes = 0;
  let ssTot = 0;
  const grandMean = y.reduce((a, b) => a + b, 0) / n;

  for (const fold of folds) {
    const testSet = new Set(fold);
    const trainRows = rows.filter((_, i) => !testSet.has(i));
    const trainY = y.filter((_, i) => !testSet.has(i));
    if (trainRows.length < 5) continue;

    const { z, means, stds } = standardize(trainRows);
    const coefficients = ridgeSolve(z, trainY, lambda);

    for (const i of fold) {
      const zRow = rows[i].map((v, j) => (v - means[j]) / stds[j]);
      const predicted = predictStandardized(coefficients, zRow);
      ssRes += (y[i] - predicted) ** 2;
      ssTot += (y[i] - grandMean) ** 2;
    }
  }

  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

export function fitCalibration(handle: string, samples: CalibrationSample[]): CalibrationModel {
  // A post with no recorded views has no measurable rate — including it would
  // silently treat "unknown" as "zero engagement".
  const usable = samples.filter((s) => s.views > 0);
  const model: CalibrationModel = {
    handle: handle.toLowerCase(),
    n: usable.length,
    fittedAt: new Date().toISOString(),
    actions: {},
  };
  if (usable.length < MIN_SAMPLES_FOR_FIT) return model;

  const rows = usable.map((s) => s.features);

  for (const action of OBSERVABLE_ACTIONS) {
    const y = usable.map((s) => toTarget(s.counts[action] ?? 0, s.views));
    // An action nobody ever took carries no signal to learn from.
    if (y.every((v) => v === 0)) continue;

    const cvR2 = crossValidatedR2(rows, y, RIDGE_LAMBDA);
    if (cvR2 < MIN_CV_R2) continue;

    const { z, means, stds } = standardize(rows);
    model.actions[action] = {
      coefficients: ridgeSolve(z, y, RIDGE_LAMBDA),
      means,
      stds,
      cvR2,
      baselineRate: fromTarget(y.reduce((a, b) => a + b, 0) / y.length),
    };
  }

  return model;
}

// A learned rate can't be dropped into the score as an absolute probability.
// Real reply rates are on the order of 1% of views, while the hand-tuned
// priors sit near 0.4 — they were tuned to spread the 0-100 score usefully,
// not to be literal probabilities. Substituting real magnitudes would shift
// every score down by a dozen points the moment someone calibrated, which
// tells them nothing: the score is a relative quality measure, not a
// forecast of literal reply probability.
//
// So the model contributes its *relative* structure — "this post should do
// ~1.8x your typical reply rate" — applied to the existing prior. That keeps
// the scale stable and comparable across users while the ranking of drafts
// comes from real measured outcomes.
const MIN_RATIO = 0.4;
const MAX_RATIO = 2.5;

/**
 * How this post compares to the author's own typical performance for one
 * action. 1.0 means "about average for you". Returns null when there's no
 * trustworthy model, so callers keep the heuristic untouched rather than
 * silently receiving a fabricated number.
 */
export function calibratedRatio(
  model: CalibrationModel | null,
  action: ObservableAction,
  features: number[]
): number | null {
  const actionModel = model?.actions[action];
  if (!model || !actionModel || actionModel.baselineRate <= 0) return null;

  const zRow = features.map((v, j) => (v - actionModel.means[j]) / actionModel.stds[j]);
  const fitted = fromTarget(predictStandardized(actionModel.coefficients, zRow));
  if (!Number.isFinite(fitted)) return null;

  const raw = fitted / actionModel.baselineRate;
  const clamped = Math.max(MIN_RATIO, Math.min(MAX_RATIO, raw));

  // Trust depends on two things, not one. Sample size says how much data
  // stands behind the fit; cross-validated R² says how much of the outcome it
  // actually explains. Weighting on n alone let a model explaining 6% of the
  // variance drive 64% of a real user's score — lots of data behind a fit that
  // barely predicts anything. Both have to be good for the model to take over.
  const sampleWeight = model.n / (model.n + SHRINKAGE_K);
  const qualityWeight = Math.min(1, actionModel.cvR2 / STRONG_FIT_R2);
  return 1 + sampleWeight * qualityWeight * (clamped - 1);
}

/**
 * Heuristic rate adjusted by what this author's real results say about posts
 * like this one. Returns null when nothing trustworthy was learned.
 */
export function calibratedRate(
  model: CalibrationModel | null,
  action: ObservableAction,
  features: number[],
  heuristicRate: number
): number | null {
  const ratio = calibratedRatio(model, action, features);
  if (ratio === null) return null;
  const adjusted = heuristicRate * ratio;
  return Number.isFinite(adjusted) ? Math.max(0, Math.min(1, adjusted)) : null;
}

/**
 * How much of the score is actually coming from real data. Reports the same
 * sample x quality trust the scorer applies, rather than sample size alone —
 * otherwise the UI would claim "64% fitted" while weak models were being
 * heavily discounted internally.
 */
export function calibrationStrength(model: CalibrationModel | null): number {
  const fitted = Object.values(model?.actions ?? {});
  if (!model || fitted.length === 0) return 0;
  const sampleWeight = model.n / (model.n + SHRINKAGE_K);
  const bestQuality = Math.max(...fitted.map((a) => Math.min(1, a.cvR2 / STRONG_FIT_R2)));
  return sampleWeight * bestQuality;
}
