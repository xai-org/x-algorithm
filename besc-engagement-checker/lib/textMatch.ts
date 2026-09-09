// Matching a saved draft against what actually got published is fuzzy by
// nature: X rewrites every link to a t.co shortlink, and people routinely
// tweak a word or two in the composer before hitting post. Exact comparison
// would miss almost everything, so this uses a Dice coefficient over
// character bigrams — tolerant of small edits, still decisive about whether
// two short texts are the same post.

export function normalizeForMatch(text: string): string {
  return text
    .toLowerCase()
    // Links are stripped, not compared: X rewrites them to t.co on publish,
    // so the published copy never matches the drafted one character-for-character.
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/www\.\S+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function bigramCounts(s: string): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < s.length - 1; i++) {
    const gram = s.slice(i, i + 2);
    counts.set(gram, (counts.get(gram) ?? 0) + 1);
  }
  return counts;
}

/** 0 = nothing in common, 1 = identical after normalization. */
export function similarity(a: string, b: string): number {
  const na = normalizeForMatch(a);
  const nb = normalizeForMatch(b);
  if (!na || !nb) return 0;
  if (na === nb) return 1;
  if (na.length < 2 || nb.length < 2) return na === nb ? 1 : 0;

  const ga = bigramCounts(na);
  const gb = bigramCounts(nb);

  let overlap = 0;
  let totalA = 0;
  let totalB = 0;
  for (const count of ga.values()) totalA += count;
  for (const [gram, count] of gb) {
    totalB += count;
    const inA = ga.get(gram);
    if (inA) overlap += Math.min(inA, count);
  }

  return (2 * overlap) / (totalA + totalB);
}

// Deliberately strict. A false positive here is worse than a miss: it would
// attribute someone else's post's real numbers to this draft and silently
// corrupt the calibration data the whole feature exists to produce. An
// unmatched draft just stays pending until the user posts it.
export const MATCH_THRESHOLD = 0.72;

export interface MatchCandidate {
  tweetId: string;
  text: string;
  createdAt?: string;
}

export function findBestMatch<T extends MatchCandidate>(
  draftText: string,
  candidates: T[],
  threshold = MATCH_THRESHOLD
): { candidate: T; score: number } | null {
  let best: { candidate: T; score: number } | null = null;
  for (const candidate of candidates) {
    const score = similarity(draftText, candidate.text);
    if (score >= threshold && (!best || score > best.score)) {
      best = { candidate, score };
    }
  }
  return best;
}
