import nlp from "compromise";

// Filler/hedge dictionary — general writing-craft heuristic, not a repo-cited
// algorithm weight. These dilute a post's punch without adding information.
export const FILLER_PHRASES = [
  "very",
  "really",
  "just",
  "actually",
  "basically",
  "literally",
  "kind of",
  "sort of",
  "in order to",
  "i think",
  "i feel like",
  "i believe",
  "at the end of the day",
  "for what it's worth",
  "needless to say",
  "it goes without saying",
];

const WEAK_OPENERS = [
  "i think",
  "i feel like",
  "i believe",
  "just wanted to",
  "so,",
  "well,",
  "honestly,",
  "basically,",
];

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function countFillerWords(text: string): number {
  const lower = text.toLowerCase();
  let count = 0;
  for (const phrase of FILLER_PHRASES) {
    const matches = lower.match(new RegExp(`\\b${escapeRegExp(phrase)}\\b`, "g"));
    if (matches) count += matches.length;
  }
  return count;
}

export function stripFillerWords(text: string): string {
  let result = text;
  let matched = false;

  // "in order to" -> "to" keeps the sentence grammatical; every other filler
  // phrase here is a pure hedge/intensifier that's safe to delete outright.
  const inOrderTo = result.replace(/\bin order to\b/gi, "to");
  if (inOrderTo !== result) matched = true;
  result = inOrderTo;

  for (const phrase of FILLER_PHRASES) {
    if (phrase === "in order to") continue;
    const re = new RegExp(`\\b${escapeRegExp(phrase)}\\b\\s*`, "gi");
    const next = result.replace(re, "");
    if (next !== result) matched = true;
    result = next;
  }
  if (!matched) return text;

  result = result.replace(/[ \t]{2,}/g, " ").replace(/\s+([.,!?])/g, "$1").trim();
  // Re-capitalize the new sentence-initial letter if a leading filler clause was removed.
  result = result.replace(/^([a-z])/, (m) => m.toUpperCase());
  result = result.replace(/([.!?]\s+)([a-z])/g, (_m, sep, ch) => sep + ch.toUpperCase());
  return result;
}

function sentences(text: string): string[] {
  try {
    return nlp(text).sentences().out("array") as string[];
  } catch {
    return text.split(/(?<=[.!?])\s+/).filter(Boolean);
  }
}

export function passiveVoiceSentenceRatio(text: string): number {
  const sents = sentences(text);
  if (sents.length === 0) return 0;
  let passive = 0;
  for (const s of sents) {
    try {
      if (nlp(s).match("#Passive").found) passive++;
    } catch {
      // ignore parse failures on a single sentence
    }
  }
  return passive / sents.length;
}

export function hasWeakOpener(text: string): boolean {
  const first = (sentences(text)[0] || text).trim().toLowerCase();
  if (!first) return false;
  return WEAK_OPENERS.some((w) => first.startsWith(w));
}
