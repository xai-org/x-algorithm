/**
 * Posting-condition analysis: when a post went out, and how long since the
 * last one.
 *
 * This exists because of what the data actually said. On a real 212-post
 * history the fitted content models topped out around R² 0.06-0.11, because
 * that account's engagement *rate* barely moves between posts — 5-10% likes
 * per view almost regardless of wording. What swings wildly is total views
 * (47 to 1,400), and reach is driven by distribution rather than phrasing.
 * Author-diversity decay and the 48-hour eligibility window are both real,
 * cited mechanics that operate on timing, so timing is where the remaining
 * explainable variance plausibly lives.
 *
 * Everything here is correlational and says so. Someone who posts
 * announcements at 9am and idle notes at midnight will show a "9am is better"
 * pattern that is really about content. The UI states that rather than
 * implying causation.
 */

export interface TimingBucket {
  label: string;
  n: number;
  medianViews: number;
  /** Multiple of the author's overall median. null when the baseline is 0. */
  lift: number | null;
}

export interface TimingInsights {
  n: number;
  minimumForInsights: number;
  minimumPerBucket: number;
  overallMedianViews: number;
  hourBlocks: TimingBucket[];
  weekdays: TimingBucket[];
  gaps: TimingBucket[];
  /** The single strongest, well-supported pattern, or null if nothing stands out. */
  best: { kind: "time" | "day" | "gap"; label: string; lift: number; n: number } | null;
}

export interface TimingSample {
  postedAt: string;
  views: number;
}

// Timing splits the data far more finely than the content model does, so it
// needs more of it: 24 hours across too few posts is noise by construction.
export const MIN_SAMPLES_FOR_TIMING = 30;
const MIN_PER_BUCKET = 6;
// Below this a "pattern" is well within the range of ordinary variation.
const MEANINGFUL_LIFT = 1.25;

const HOUR_BLOCKS: { label: string; from: number; to: number }[] = [
  { label: "late night (12am–4am)", from: 0, to: 4 },
  { label: "early morning (4am–8am)", from: 4, to: 8 },
  { label: "morning (8am–12pm)", from: 8, to: 12 },
  { label: "afternoon (12pm–4pm)", from: 12, to: 16 },
  { label: "evening (4pm–8pm)", from: 16, to: 20 },
  { label: "night (8pm–12am)", from: 20, to: 24 },
];

const WEEKDAY_LABELS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

const GAP_BUCKETS: { label: string; maxHours: number }[] = [
  { label: "under 1h after the last post", maxHours: 1 },
  { label: "1–4h after the last post", maxHours: 4 },
  { label: "4–12h after the last post", maxHours: 12 },
  { label: "12h+ after the last post", maxHours: Infinity },
];

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function bucketize(
  groups: Map<string, number[]>,
  order: string[],
  overallMedian: number
): TimingBucket[] {
  return order
    .map((label) => {
      const views = groups.get(label) ?? [];
      const medianViews = median(views);
      return {
        label,
        n: views.length,
        medianViews,
        lift: overallMedian > 0 ? medianViews / overallMedian : null,
      };
    })
    .filter((b) => b.n >= MIN_PER_BUCKET);
}

/**
 * @param tzOffsetMinutes Browser-style offset (Date.getTimezoneOffset), so
 * hour-of-day buckets are the author's local time rather than UTC. "Post at
 * 9am" is only actionable in the timezone they actually live in.
 */
export function buildTimingInsights(
  samples: TimingSample[],
  tzOffsetMinutes = 0
): TimingInsights {
  const usable = samples
    .filter((s) => s.postedAt && s.views > 0)
    .map((s) => ({ at: new Date(s.postedAt), views: s.views }))
    .filter((s) => !Number.isNaN(s.at.getTime()))
    .sort((a, b) => a.at.getTime() - b.at.getTime());

  const insights: TimingInsights = {
    n: usable.length,
    minimumForInsights: MIN_SAMPLES_FOR_TIMING,
    minimumPerBucket: MIN_PER_BUCKET,
    overallMedianViews: median(usable.map((s) => s.views)),
    hourBlocks: [],
    weekdays: [],
    gaps: [],
    best: null,
  };
  if (usable.length < MIN_SAMPLES_FOR_TIMING) return insights;

  const overall = insights.overallMedianViews;
  const localOf = (d: Date) => new Date(d.getTime() - tzOffsetMinutes * 60_000);

  const byHour = new Map<string, number[]>();
  const byDay = new Map<string, number[]>();
  const byGap = new Map<string, number[]>();

  usable.forEach((sample, index) => {
    const local = localOf(sample.at);
    const hour = local.getUTCHours();

    const block = HOUR_BLOCKS.find((b) => hour >= b.from && hour < b.to);
    if (block) {
      byHour.set(block.label, [...(byHour.get(block.label) ?? []), sample.views]);
    }

    const day = WEEKDAY_LABELS[local.getUTCDay()];
    byDay.set(day, [...(byDay.get(day) ?? []), sample.views]);

    // The first post has no predecessor, so it has no gap to attribute.
    if (index > 0) {
      const gapHours = (sample.at.getTime() - usable[index - 1].at.getTime()) / 3_600_000;
      const bucket = GAP_BUCKETS.find((g) => gapHours < g.maxHours);
      if (bucket) byGap.set(bucket.label, [...(byGap.get(bucket.label) ?? []), sample.views]);
    }
  });

  insights.hourBlocks = bucketize(byHour, HOUR_BLOCKS.map((b) => b.label), overall);
  insights.weekdays = bucketize(byDay, WEEKDAY_LABELS, overall);
  insights.gaps = bucketize(byGap, GAP_BUCKETS.map((g) => g.label), overall);

  const candidates: { kind: "time" | "day" | "gap"; bucket: TimingBucket }[] = [
    ...insights.hourBlocks.map((b) => ({ kind: "time" as const, bucket: b })),
    ...insights.weekdays.map((b) => ({ kind: "day" as const, bucket: b })),
    ...insights.gaps.map((b) => ({ kind: "gap" as const, bucket: b })),
  ];

  let best: TimingInsights["best"] = null;
  for (const { kind, bucket } of candidates) {
    if (bucket.lift === null || bucket.lift < MEANINGFUL_LIFT) continue;
    if (!best || bucket.lift > best.lift) {
      best = { kind, label: bucket.label, lift: bucket.lift, n: bucket.n };
    }
  }
  insights.best = best;

  return insights;
}
