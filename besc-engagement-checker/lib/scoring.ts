import type {
  ActionMultipliers,
  AnalyzeRequest,
  ActionRow,
  FeatureReport,
  RiskFlag,
  ScoreResult,
  Tip,
} from "./types";
import { LLM_ACTIONS } from "./types";
import { countFillerWords, passiveVoiceSentenceRatio, hasWeakOpener } from "./nlp";
import {
  calibratedRate,
  calibratedRatio,
  calibrationStrength,
  featureVector,
  type CalibrationModel,
} from "./calibration";

/**
 * ─────────────────────────────────────────────────────────────────────────
 * GROUND TRUTH: home-mixer/params/param.rs (X "For You" algorithm, this repo)
 * RankingScorer combines Phoenix's per-action probabilities as:
 *     Final Score = Σ (weight_i × P(action_i))
 * These are the *actual* production default weights mirrored in param.rs.
 * We reuse them verbatim so the composite score below is arithmetically
 * identical in spirit to how X itself blends predicted actions.
 * ─────────────────────────────────────────────────────────────────────────
 */
export const WEIGHTS = {
  favorite: 0.5,
  reply: 5.0,
  // Added to reply weight, but only for an ORIGINAL post (not itself a
  // reply/retweet) shown to a viewer who mutually follows the author —
  // verified against ranking_scorer.rs's bidirectional_boost_eligible(),
  // which requires in_reply_to_tweet_id.is_none() && retweeted_tweet_id.is_none().
  // A post that IS a reply never gets this boost, regardless of who it's to.
  bidirectionalReplyBoost: 15.0,
  retweet: 1.0,
  photoExpand: 0.05,
  videoOpen: 0.05,
  click: 0.4,
  openLink: 0.2,
  vqv: 0.05,
  share: 2.0,
  shareViaDm: 5.0,
  shareViaCopyLink: 20.0,
  quote: 5.0,
  followAuthor: 4.0,
  notInterested: -43.2,
  blockAuthor: -31.2,
  muteAuthor: -58.8,
  report: -234.0,
  notDwelled: -0.02,
} as const;

// home-mixer/scorers/ranking_scorer.rs — diversity_multiplier(decay, floor, k)
export const AUTHOR_DIVERSITY_DECAY = 0.5; // params::AuthorDiversityDecay
export const AUTHOR_DIVERSITY_FLOOR = 0.25; // params::AuthorDiversityFloor
// home-mixer/params/param.rs — OonWeightFactor
export const OON_WEIGHT_FACTOR = 0.75;

// home-mixer/params/param.rs — ColdStartFollowerCap / ColdStartMaxPostAgeSecs / LowImpressionsMaxPositionRatio
export const COLD_START_FOLLOWER_CAP = 1000;
export const COLD_START_MAX_AGE_HOURS = 24; // 86400 secs
export const COLD_START_MAX_POSITION_RATIO = 0.85;
// home-mixer/params/param.rs — ColdStartImpressionThreshold: the boost only
// applies while the post is still under this many views (home-mixer/scorers/
// author_cold_start.rs:165,179), and only while it's already ranking in the
// top COLD_START_MAX_POSITION_RATIO of scored candidates for that viewer —
// the boost lifts an already-decent-scoring post up toward what's currently
// sitting around rank 15 (ColdStartSlotMin/Max), not an automatic top slot.
export const COLD_START_IMPRESSION_THRESHOLD = 1000;

// home-mixer/params/param.rs — TopicOonWeightFactor: out-of-network reach on
// a topic-matched surface is discounted harder (50%) than the generic OON
// discount below (25%).
export const TOPIC_OON_WEIGHT_FACTOR = 0.5;

// home-mixer/params/param.rs — MinVideoDurationMs: a video under this length
// gets its video-quality-view weight forced to 0.0 (home-mixer/util/
// candidates_util.rs:19-40) — the VQV action row never fires at all,
// regardless of how good the video is.
export const MIN_VIDEO_DURATION_MS = 10_000;

// X platform posting limits (not a repo-cited ranking weight — this is the
// literal composer character cap). Free accounts: 280. X Premium: 4,000.
// Premium+ / Verified Organizations: 25,000. We only know "verified: yes/no"
// from available data, not which paid tier, so 4,000 is used as a safe floor
// for any verified account — it never overstates what a given tier allows.
export const STANDARD_CHAR_LIMIT = 280;
export const VERIFIED_CHAR_LIMIT = 4000;

export function getCharLimit(isVerified?: boolean): number {
  return isVerified ? VERIFIED_CHAR_LIMIT : STANDARD_CHAR_LIMIT;
}

export function authorDiversityMultiplier(k: number): number {
  return (
    (1 - AUTHOR_DIVERSITY_FLOOR) * Math.pow(AUTHOR_DIVERSITY_DECAY, k) +
    AUTHOR_DIVERSITY_FLOOR
  );
}

const EMOJI_RE =
  /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{1F1E6}-\u{1F1FF}]/gu;

const URL_RE = /(https?:\/\/[^\s]+)|(\bwww\.[^\s]+)/gi;
const MENTION_RE = /(^|\s)@[a-zA-Z0-9_]{1,15}/g;
export const HASHTAG_RE = /(^|\s)#[a-zA-Z0-9_]+/g;

// Deliberately excludes t.co. X rewrites EVERY link posted to it into a t.co
// shortlink, so treating t.co as a suspicious redirect penalised every post
// containing any link at all — which is the opposite of a spam signal. The
// URL-verdict rules this models are about the reputation of the *destination*
// domain, and X plainly doesn't distrust its own wrapper. A t.co link just
// means the destination is unknowable from the text, which is neutral, not bad.
const SHORTENER_HOSTS = [
  "bit.ly",
  "tinyurl.com",
  "goo.gl",
  "ow.ly",
  "buff.ly",
  "is.gd",
  "cutt.ly",
  "rebrand.ly",
  "shorturl.at",
  "tiny.cc",
];

export const BOILERPLATE_PHRASES = [
  "follow for follow",
  "f4f",
  "follow me for more",
  "like and retweet",
  "rt if you agree",
  "retweet if",
  "link in bio",
  "dm me for",
  "click the link below",
  "check out my link",
  "comment your",
  "tag 3 friends",
  "tag a friend",
];

// abuse-enforcement-service/service-lib/rules/enforcement_post.yaml:39-44 —
// content matching X's "llm_slop_post" classifier gets a RiskyHighVizReply
// label (30-day TTL). This repo doesn't show what that label actually does
// downstream (nothing in visibility-filtering consumes it, only the
// enforcement service's own tests reference it), so treat this as "a label
// gets attached, for a month, for something free to avoid" rather than a
// citable drop/downrank the way the other risk flags are.
export const AI_SLOP_PHRASES = [
  "in today's fast-paced",
  "in the ever-evolving",
  "dive into",
  "delve into",
  "it's important to note",
  "it's worth noting",
  "unlock the power of",
  "game-changer",
  "game changer",
  "seamless integration",
  "embark on a journey",
  "cutting-edge",
  "revolutionize",
  "testament to",
  "navigating the",
];

export const REPLY_CTA_PHRASES = [
  "what do you think",
  "thoughts?",
  "agree or disagree",
  "reply with",
  "reply below",
  "comment below",
  "tell me why",
  "am i wrong",
  "change my mind",
  "unpopular opinion",
];

// Closers that invite a reply without asking anything real. Kept separate
// from REPLY_CTA_PHRASES because they need to score *worse*, not better: a
// bolt-on "What do you think?" gives a reader almost no reason to answer,
// and a tool that appends the identical tail to thousands of posts is
// manufacturing exactly the templated-text pattern duplicate-text spam
// detection looks for (BBQDuplicateTextProd.bot).
export const GENERIC_CLOSERS = [
  "what do you think",
  "what are your thoughts",
  "thoughts",
  "agree or disagree",
  "do you agree",
  "who agrees",
  "reply below",
  "comment below",
  "let me know",
  "am i wrong",
  "change my mind",
  "right or wrong",
];

const QUESTION_STOPWORDS = new Set([
  "what", "who", "when", "where", "why", "how", "which", "do", "does", "did",
  "is", "are", "was", "were", "will", "would", "should", "could", "can", "you",
  "your", "yours", "i", "we", "they", "it", "this", "that", "the", "a", "an",
  "of", "to", "in", "on", "for", "and", "or", "but", "if", "so", "me", "my",
  "any", "one", "next", "most", "more", "here", "there", "about", "think",
]);

export type ReplyHookKind = "specific" | "generic" | "none";

/**
 * Distinguishes a question that carries real content from a bolt-on closer.
 * This drives the reply-probability estimate, and getting it right is what
 * stops the whole tool from converging on "What do you think?" — see the
 * comment on GENERIC_CLOSERS.
 */
export function classifyReplyHook(text: string): ReplyHookKind {
  const lower = text.toLowerCase();
  const questions = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.includes("?"));

  if (questions.length === 0) {
    // A CTA can invite replies without a question mark ("reply with your...").
    return REPLY_CTA_PHRASES.some((p) => lower.includes(p)) ? "generic" : "none";
  }

  for (const question of questions) {
    let remainder = question.toLowerCase();
    for (const closer of GENERIC_CLOSERS) remainder = remainder.split(closer).join(" ");
    const contentWords = remainder
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 1 && !QUESTION_STOPWORDS.has(w));
    // Enough left over, once the boilerplate and function words are gone,
    // that the question is actually asking about *this* post.
    if (contentWords.length >= 3) return "specific";
  }
  return "generic";
}

const SHARE_CTA_PHRASES = [
  "share this",
  "send this to",
  "everyone needs to see this",
  "bookmark this",
  "save this",
];

function normalizeHashtagWord(w: string): string {
  return w.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
}

// A post that hashtags its own brand/ticker (#BESC) and also writes it in
// caps in the body ("BESC Exchange...") isn't shouting — it's just the name.
// Both the ALL-CAPS risk score and the auto-fixer treat any word echoed as a
// hashtag elsewhere in the same post as protected, rather than penalizing or
// rewriting it.
export function getHashtagWordSet(text: string): Set<string> {
  const matches = text.match(HASHTAG_RE) || [];
  return new Set(matches.map(normalizeHashtagWord));
}

export function isShoutingCapsWord(token: string, protectedWords: Set<string>): boolean {
  const isAllCaps = token.length >= 3 && token === token.toUpperCase() && /[A-Z]/.test(token);
  if (!isAllCaps) return false;
  return !protectedWords.has(normalizeHashtagWord(token));
}

export function extractFeatures(text: string, link: string): FeatureReport {
  const trimmed = text.trim();
  const words = trimmed.length ? trimmed.split(/\s+/) : [];
  const chars = trimmed.length;

  const hashtags = (trimmed.match(HASHTAG_RE) || []).length;
  const mentions = (trimmed.match(MENTION_RE) || []).length;
  const textLinks = (trimmed.match(URL_RE) || []).length;
  const links = textLinks + (link.trim() ? 1 : 0);
  const emojis = (trimmed.match(EMOJI_RE) || []).length;

  const hashtagWords = getHashtagWordSet(trimmed);
  const capsWords = words.filter((w) => isShoutingCapsWord(w, hashtagWords));
  const allCapsWordRatio = words.length ? capsWords.length / words.length : 0;

  const exclamationBursts = (trimmed.match(/!{2,}|\?{2,}|!\?|\?!/g) || [])
    .length;
  const hasQuestion = (trimmed.match(/\?/g) || []).length;

  const lower = trimmed.toLowerCase();
  const hasReplyCTA = REPLY_CTA_PHRASES.some((p) => lower.includes(p));
  const replyHookKind = classifyReplyHook(trimmed);
  const hasShareCTA = SHARE_CTA_PHRASES.some((p) => lower.includes(p));
  const hasBoilerplateCTA = BOILERPLATE_PHRASES.some((p) => lower.includes(p));
  const hasAiSlopPhrasing = AI_SLOP_PHRASES.some((p) => lower.includes(p));
  const hasNumbers = /\d/.test(trimmed);
  const hasThreadMarker = /🧵|^1\/|^\(1\/|thread\b/i.test(trimmed);

  let urlRisk = false;
  let urlReason: string | null = null;
  const candidateUrl = link.trim() || (trimmed.match(URL_RE) || [])[0] || "";
  if (candidateUrl) {
    const host = candidateUrl
      .replace(/^https?:\/\//i, "")
      .replace(/^www\./i, "")
      .split(/[/?#]/)[0]
      .toLowerCase();
    if (SHORTENER_HOSTS.includes(host)) {
      urlRisk = true;
      urlReason = `"${host}" is a link-shortener/redirect domain. Redirect chains are exactly what scarecrow's URL-verdict rules (e.g. bot 3226, bot 6754) resolve and re-check before trusting a link.`;
    } else if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) {
      urlRisk = true;
      urlReason =
        "Raw IP-address links have no domain reputation history, so URL-verdict scoring treats them as low quality by default.";
    } else if (/\.(xyz|top|click|country|gq|tk|work)$/i.test(host)) {
      urlRisk = true;
      urlReason = `The "${host.split(".").pop()}" TLD is disproportionately associated with low-quality/BAD URL verdicts in spam-detection datasets.`;
    }
  }

  return {
    chars,
    words: words.length,
    hashtags,
    mentions,
    links,
    emojis,
    allCapsWordRatio,
    exclamationBursts,
    hasQuestion,
    hasReplyCTA,
    replyHookKind,
    hasShareCTA,
    hasBoilerplateCTA,
    hasAiSlopPhrasing,
    hasNumbers,
    hasThreadMarker,
    fillerWords: countFillerWords(trimmed),
    passiveVoiceRatio: passiveVoiceSentenceRatio(trimmed),
    hasWeakOpener: hasWeakOpener(trimmed),
    urlRisk,
    urlReason,
  };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

interface ActionProbabilities {
  favorite: number;
  reply: number;
  retweet: number;
  photoExpand: number;
  videoOpen: number;
  click: number;
  openLink: number;
  vqv: number;
  share: number;
  shareViaDm: number;
  shareViaCopyLink: number;
  quote: number;
  followAuthor: number;
  notInterested: number;
  blockAuthor: number;
  muteAuthor: number;
  report: number;
  notDwelled: number;
}

/**
 * Heuristic estimator standing in for Phoenix's learned P(action | post, viewer).
 * We don't have Phoenix's weights or a viewer's action-sequence here — instead
 * we approximate propensities from surface-level features that correlate with
 * each action, then blend them with the *real* production weights above. This
 * is a transparent proxy, not a claim that we're running the actual model.
 */
function estimateActionProbabilities(
  f: FeatureReport,
  req: AnalyzeRequest
): ActionProbabilities {
  const lengthScore = clamp01(1 - Math.abs(f.words - 30) / 80); // sweet spot ~ 15-45 words
  const hasMedia = req.mediaType !== "none";
  const spamPenalty = clamp01(
    f.allCapsWordRatio * 0.6 +
      Math.min(f.hashtags, 8) / 8 * 0.35 +
      Math.min(f.exclamationBursts, 4) / 4 * 0.3 +
      (f.hasBoilerplateCTA ? 0.45 : 0) +
      (f.hasAiSlopPhrasing ? 0.2 : 0) +
      (f.urlRisk ? 0.35 : 0) +
      (f.mentions >= 4 ? 0.2 : 0)
  );

  // Filler words / passive voice / a weak opening line are general writing-craft
  // signals (not repo-cited weights) — a small drag on attention and reply,
  // since a muddier or less direct post gives people less to react to.
  const wordinessPenalty = clamp01(
    Math.min(f.fillerWords, 6) / 6 * 0.12 +
      f.passiveVoiceRatio * 0.15 +
      (f.hasWeakOpener ? 0.08 : 0)
  );

  const favorite = clamp01(
    0.05 +
      lengthScore * 0.15 +
      (hasMedia ? 0.1 : 0) +
      Math.min(f.emojis, 3) * 0.02 +
      (f.hasNumbers ? 0.03 : 0) -
      spamPenalty * 0.15 -
      wordinessPenalty * 0.06
  );

  // A specific question is worth far more than a generic closer, and the gap
  // is deliberate. The old model gave a bolt-on "What do you think?" +0.46
  // (CTA + question mark) against +0.18 for a genuinely specific question —
  // so every optimizer pass and every AI candidate gate was being pulled
  // toward the laziest possible ending. Reality runs the other way: a
  // question only someone who read the post can answer is what actually earns
  // replies, and the identical-tail version courts duplicate-text detection.
  const replyHookCredit =
    f.replyHookKind === "specific" ? 0.4 : f.replyHookKind === "generic" ? 0.15 : 0;

  const reply = clamp01(
    0.02 + replyHookCredit - spamPenalty * 0.1 - wordinessPenalty * 0.08
  );

  const retweet = clamp01(
    0.02 +
      (f.hasNumbers ? 0.05 : 0) +
      (f.hasThreadMarker ? 0.06 : 0) +
      lengthScore * 0.06 -
      spamPenalty * 0.12
  );

  // Copy-link share is the heaviest action in the model, and it requires
  // something concrete worth sending to a specific person. A two-word status
  // update has nothing to pass on, but with no hashtags, links or caps it also
  // trips no penalties — so it was scoring mid-range while genuinely
  // informative posts scored lower. Absence of problems is not the same as
  // presence of substance.
  const substance = clamp01(0.25 + (Math.min(f.words, 20) / 20) * 0.75);

  const quote = clamp01((0.015 + (f.hasReplyCTA ? 0.05 : 0) + retweet * 0.4) * substance);

  const share = clamp01((0.02 + (f.hasShareCTA ? 0.18 : 0) + retweet * 0.5) * substance);
  const shareViaDm = clamp01(share * 0.55);
  const shareViaCopyLink = clamp01(share * 0.4);

  const openLink = clamp01(f.links > 0 ? 0.22 - (f.urlRisk ? 0.12 : 0) : 0.0);
  const click = clamp01(0.08 + openLink * 0.5 + (hasMedia ? 0.05 : 0));
  const photoExpand = clamp01(req.mediaType === "photo" ? 0.28 : 0.02);
  const videoOpen = clamp01(req.mediaType === "video" ? 0.3 : 0.02);
  const vqv = clamp01(req.mediaType === "video" ? 0.22 : 0.01);

  const followAuthor = clamp01(
    0.01 + favorite * 0.08 + reply * 0.06 - spamPenalty * 0.08
  );

  // These carry the three largest weights in the model (up to -234), and the
  // real algorithm pairs those weights with genuinely tiny probabilities —
  // people almost never report a post. Pairing huge weights with inflated
  // probabilities made the negatives swamp everything else: a normal product
  // update with a link and a few hashtags scored a flat 0, while a two-word
  // post with no link scored 15. Real published posts with 300 views, 30 likes
  // and no sign of trouble were being graded "High Risk".
  //
  // Scaled to plausible magnitudes. The ordering is unchanged — spammier
  // drafts still lose more — but a clean post now loses ~0.1 points to
  // negatives instead of ~10, and only a genuinely abusive one approaches the
  // floor.
  const notInterested = clamp01(0.002 + spamPenalty * 0.06);
  const blockAuthor = clamp01(0.0002 + spamPenalty * 0.01);
  const muteAuthor = clamp01(0.0005 + spamPenalty * 0.02);
  const report = clamp01(
    0.00002 + (f.urlRisk ? 0.0003 : 0) + spamPenalty * 0.004
  );
  const notDwelled = clamp01(
    0.35 - lengthScore * 0.15 + spamPenalty * 0.2 + wordinessPenalty * 0.3
  );

  return {
    favorite,
    reply,
    retweet,
    photoExpand,
    videoOpen,
    click,
    openLink,
    vqv,
    share,
    shareViaDm,
    shareViaCopyLink,
    quote,
    followAuthor,
    notInterested,
    blockAuthor,
    muteAuthor,
    report,
    notDwelled,
  };
}

/**
 * Replaces guessed action probabilities with ones fitted to this author's real
 * outcomes, where a trustworthy fit exists. Only the four actions X actually
 * reports counts for can be learned directly. The share family (copy-link, DM,
 * generic share) is never exposed publicly, so it stays heuristic — but it's
 * nudged by the author's observed bookmark rate, which is the closest
 * observable proxy for "worth saving or sending to someone".
 */
function applyCalibration(
  heuristic: ActionProbabilities,
  features: FeatureReport,
  req: AnalyzeRequest,
  model?: CalibrationModel | null
): ActionProbabilities {
  if (!model) return heuristic;

  const vector = featureVector(features, req.mediaType);
  const out = { ...heuristic };

  const favorite = calibratedRate(model, "favorite", vector, heuristic.favorite);
  if (favorite !== null) out.favorite = favorite;
  const reply = calibratedRate(model, "reply", vector, heuristic.reply);
  if (reply !== null) out.reply = reply;
  const retweet = calibratedRate(model, "retweet", vector, heuristic.retweet);
  if (retweet !== null) out.retweet = retweet;
  const quote = calibratedRate(model, "quote", vector, heuristic.quote);
  if (quote !== null) out.quote = quote;

  // Share-family actions are never exposed publicly, so they can't be learned
  // directly. The author's bookmark rate is the closest observable proxy for
  // "worth saving or sending to someone", so its learned ratio carries them.
  const bookmarkRatio = calibratedRatio(model, "bookmark", vector);
  if (bookmarkRatio !== null) {
    out.shareViaCopyLink = clamp01(heuristic.shareViaCopyLink * bookmarkRatio);
    out.shareViaDm = clamp01(heuristic.shareViaDm * bookmarkRatio);
    out.share = clamp01(heuristic.share * bookmarkRatio);
  }

  return out;
}

/**
 * Folds in a language model's read of the draft, expressed in the same
 * "multiple of this author's typical post" units the fitted regression uses.
 *
 * It is deliberately damped by how much real fitted data is already in play.
 * Both estimates are answering the same question from overlapping evidence, so
 * stacking them at full strength would double-count; and where measured
 * outcomes exist they are the better evidence, so the model's opinion should
 * recede as the fit earns trust rather than compete with it.
 */
function applyLlmMultipliers(
  probabilities: ActionProbabilities,
  multipliers: ActionMultipliers | null | undefined,
  model?: CalibrationModel | null
): ActionProbabilities {
  if (!multipliers) return probabilities;

  const room = 1 - calibrationStrength(model ?? null);
  const out = { ...probabilities };
  for (const action of LLM_ACTIONS) {
    const raw = multipliers[action];
    if (raw === undefined) continue;
    const damped = 1 + room * (raw - 1);
    out[action] = clamp01(out[action] * damped);
  }
  // The share family isn't separately judged — copy-link is the observable
  // stand-in for "worth sending to someone", so it carries the other two.
  const shareMultiplier = multipliers.shareViaCopyLink;
  if (shareMultiplier !== undefined) {
    const damped = 1 + room * (shareMultiplier - 1);
    out.shareViaDm = clamp01(out.shareViaDm * damped);
    out.share = clamp01(out.share * damped);
  }
  return out;
}

function buildActionRows(
  p: ActionProbabilities,
  req: AnalyzeRequest
): ActionRow[] {
  const bidirectionalBoostEligible = !req.isReply && req.hasMutualFollowAudience;
  const replyWeight =
    WEIGHTS.reply + (bidirectionalBoostEligible ? WEIGHTS.bidirectionalReplyBoost : 0);

  const rows: ActionRow[] = [
    {
      id: "reply",
      label: "Reply",
      group: "engagement",
      weight: replyWeight,
      probability: p.reply,
      contribution: replyWeight * p.reply,
      weightSource: bidirectionalBoostEligible
        ? "ReplyWeight (5.0) + BidirectionalFollowReplyWeightBoost (15.0) — original posts to a mutual-follow audience only"
        : "ReplyWeight = 5.0",
    },
    {
      id: "quote",
      label: "Quote post",
      group: "engagement",
      weight: WEIGHTS.quote,
      probability: p.quote,
      contribution: WEIGHTS.quote * p.quote,
      weightSource: "QuoteWeight = 5.0",
    },
    {
      id: "shareViaCopyLink",
      label: "Share via copy link",
      group: "engagement",
      weight: WEIGHTS.shareViaCopyLink,
      probability: p.shareViaCopyLink,
      contribution: WEIGHTS.shareViaCopyLink * p.shareViaCopyLink,
      weightSource: "ShareViaCopyLinkWeight = 20.0 (highest single weight)",
    },
    {
      id: "shareViaDm",
      label: "Share via DM",
      group: "engagement",
      weight: WEIGHTS.shareViaDm,
      probability: p.shareViaDm,
      contribution: WEIGHTS.shareViaDm * p.shareViaDm,
      weightSource: "ShareViaDmWeight = 5.0",
    },
    {
      id: "share",
      label: "Share (generic)",
      group: "engagement",
      weight: WEIGHTS.share,
      probability: p.share,
      contribution: WEIGHTS.share * p.share,
      weightSource: "ShareWeight = 2.0",
    },
    {
      id: "retweet",
      label: "Repost",
      group: "engagement",
      weight: WEIGHTS.retweet,
      probability: p.retweet,
      contribution: WEIGHTS.retweet * p.retweet,
      weightSource: "RetweetWeight = 1.0",
    },
    {
      id: "favorite",
      label: "Like",
      group: "engagement",
      weight: WEIGHTS.favorite,
      probability: p.favorite,
      contribution: WEIGHTS.favorite * p.favorite,
      weightSource: "FavoriteWeight = 0.5",
    },
    {
      id: "followAuthor",
      label: "Follow you",
      group: "author",
      weight: WEIGHTS.followAuthor,
      probability: p.followAuthor,
      contribution: WEIGHTS.followAuthor * p.followAuthor,
      weightSource: "FollowAuthorWeight = 4.0",
    },
    {
      id: "openLink",
      label: "Open link",
      group: "clicks",
      weight: WEIGHTS.openLink,
      probability: p.openLink,
      contribution: WEIGHTS.openLink * p.openLink,
      weightSource: "OpenLinkWeight = 0.2",
    },
    {
      id: "click",
      label: "Click post",
      group: "clicks",
      weight: WEIGHTS.click,
      probability: p.click,
      contribution: WEIGHTS.click * p.click,
      weightSource: "ClickWeight = 0.4",
    },
    {
      id: "photoExpand",
      label: "Expand photo",
      group: "clicks",
      weight: WEIGHTS.photoExpand,
      probability: p.photoExpand,
      contribution: WEIGHTS.photoExpand * p.photoExpand,
      weightSource: "PhotoExpandWeight = 0.05",
    },
    {
      id: "videoOpen",
      label: "Open video",
      group: "clicks",
      weight: WEIGHTS.videoOpen,
      probability: p.videoOpen,
      contribution: WEIGHTS.videoOpen * p.videoOpen,
      weightSource: "VideoOpenWeight = 0.05",
    },
    {
      id: "vqv",
      label: "Video quality view",
      group: "attention",
      weight: WEIGHTS.vqv,
      probability: p.vqv,
      contribution: WEIGHTS.vqv * p.vqv,
      weightSource: "VqvWeight = 0.05",
    },
    {
      id: "notDwelled",
      label: "Scrolls past without dwelling",
      group: "negative",
      weight: WEIGHTS.notDwelled,
      probability: p.notDwelled,
      contribution: WEIGHTS.notDwelled * p.notDwelled,
      weightSource: "NotDwelledWeight = -0.02",
    },
    {
      id: "notInterested",
      label: '"Not interested"',
      group: "negative",
      weight: WEIGHTS.notInterested,
      probability: p.notInterested,
      contribution: WEIGHTS.notInterested * p.notInterested,
      weightSource: "NotInterestedWeight = -43.2",
    },
    {
      id: "blockAuthor",
      label: "Block",
      group: "negative",
      weight: WEIGHTS.blockAuthor,
      probability: p.blockAuthor,
      contribution: WEIGHTS.blockAuthor * p.blockAuthor,
      weightSource: "BlockAuthorWeight = -31.2",
    },
    {
      id: "muteAuthor",
      label: "Mute",
      group: "negative",
      weight: WEIGHTS.muteAuthor,
      probability: p.muteAuthor,
      contribution: WEIGHTS.muteAuthor * p.muteAuthor,
      weightSource: "MuteAuthorWeight = -58.8",
    },
    {
      id: "report",
      label: "Report",
      group: "negative",
      weight: WEIGHTS.report,
      probability: p.report,
      contribution: WEIGHTS.report * p.report,
      weightSource: "ReportWeight = -234.0 (single largest weight in the model)",
    },
  ];

  return rows.sort((a, b) => b.contribution - a.contribution);
}

function buildRisks(f: FeatureReport, req: AnalyzeRequest): RiskFlag[] {
  const risks: RiskFlag[] = [
    {
      id: "url-verdict",
      severity: "critical",
      title: "Link may resolve to a low-quality / BAD URL verdict",
      detail: f.urlReason
        ? `${f.urlReason} A "LOW_QUALITY" verdict gets downranked. If re-crawling turns up an actual "UNSAFE" verdict, though, the consequence is categorically worse: visibility-filtering applies MALICIOUS_URL_DROP/DO_NOT_AMPLIFY_DROP, which is a hard non-distribution drop for every non-author viewer, not a downrank. And it doesn't stay contained to this one post — once a domain's verdict flips to BAD, the retroactive-labeling bot applies SPAM to every past tweet you've posted that shares that domain, not just new ones.`
        : "No risky link pattern detected in this draft.",
      source:
        "botmaker-rules/scarecrow/bot/Tweet_Spam_High_Recall_RTF_All_Bad_URL_Sources.bot, LQ_Tweets_With_LQ_URL_Verdict_At_Mention_To_NonFollower_v2.bot, rtf_tweets_on_unsafe_verdict.bot (id 20790), Tweet_Search_Blacklist_RTF_All_UNSAFE_URL_Sources.bot (id 20789), tweet_rtf_or_unrtf_on_bad_verdict.bot (id 2002); visibility-filtering/rules/tweet_label_drops.rs:113-124",
      triggered: f.urlRisk,
    },
    {
      id: "mention-link-combo",
      severity: "warning",
      title: "@-mentioning non-followers alongside a link amplifies URL risk",
      detail:
        "This post @-mentions people while carrying a link with a shaky reputation. The same URL is treated as riskier when it appears in an @-mention to someone who doesn't follow you.",
      source:
        "botmaker-rules/scarecrow/bot/LQ_Tweets_With_LQ_URL_Verdict_At_Mention_To_NonFollower_v2.bot (id 6754)",
      triggered: f.urlRisk && f.mentions >= 1,
    },
    {
      id: "duplicate-boilerplate",
      severity: "warning",
      title: "Reads like templated / copy-paste boilerplate",
      detail:
        'Phrases like "follow for follow", "link in bio", or "RT if you agree" match the kind of generic, repeated-across-many-posts text that duplicate-text spam detection is built to catch. The same COPYPASTA_SPAM detection runs on reply text too, unigram- and CJK-character-matched, not just original posts.',
      source:
        "botmaker-rules/scarecrow/bot/BBQDuplicateTextProd.bot (id 21711), BBQDuplicateTextRepliesProd.bot (id 21599); heuristic proxy, not a literal duplicate-corpus match",
      triggered: f.hasBoilerplateCTA,
    },
    {
      id: "all-caps",
      severity: "warning",
      title: "Heavy use of ALL-CAPS reads as spammy / shouty",
      detail:
        "A high ratio of all-caps words correlates with the kind of low-quality, aggressive formatting that pushes up 'not interested', mute, and report propensity. Those three actions carry the three most negative weights in the entire model (-43.2, -58.8, -234).",
      source: "home-mixer/params/param.rs: NotInterestedWeight, MuteAuthorWeight, ReportWeight",
      triggered: f.allCapsWordRatio > 0.25,
    },
    {
      id: "hashtag-stuffing",
      severity: "warning",
      title: "Hashtag stuffing",
      detail:
        "More than a handful of hashtags is a classic low-quality/spam signal and dilutes what the post is actually about.",
      source: "General spam-heuristic best practice, paired with scarecrow's broader spam-label pipeline",
      triggered: f.hashtags > 4,
    },
    {
      id: "nsfw-interstitial",
      severity: "critical",
      title: "Sensitive media: blurred for followers, fully dropped from recommendations",
      detail:
        "Marking media sensitive costs more than a blur. Followers see it behind a click-through interstitial (NsfwAuthorInterstitialRule) — but for everyone else, TweetNsfwUserDropRule removes it from recommendations entirely, on an out-of-network surface that runs a much longer list of drop rules than followers ever see. That's a hard cap on reach beyond your existing audience, not just suppressed impulse engagement.",
      source:
        "visibility-filtering/rules/nsfw_interstitial.rs: NSFW_HIGH_PRECISION_INTERSTITIAL / NsfwAuthorInterstitialRule (both surfaces); visibility-filtering/rules/tweet_flag_rules.rs:35-44 TWEET_NSFW_USER_DROP, wired OON-only at registry.rs:145",
      triggered: req.nsfw,
    },
    {
      id: "nsfw-account-label",
      severity: "warning",
      title: "Repeated NSFW posts can trip an account-level label",
      detail:
        "This goes beyond a single post's interstitial: if roughly 3 of your last 5 posts (within 60 days) carry an NSFW_HIGH_PRECISION flag, your whole account gets a 7-day NSFW_HIGH_PRECISION label; an unbroken streak of flagged posts can escalate further to a POSSIBLY_NSFW_ACCOUNT label.",
      source: "safety-label-user-agg/postToUserLabelRules.strato:360-428",
      triggered: req.nsfw && req.recentPostsCount >= 2,
    },
    {
      id: "repeat-posting",
      severity: "info",
      title: "Repeat posting in the same window compresses reach",
      detail: `You told us this is post #${
        req.recentPostsCount + 1
      } from you in this window. Author-diversity decay multiplies each additional post from the same author by (1 - 0.25) × 0.5^k + 0.25, so a 2nd post already scores at ${(
        authorDiversityMultiplier(1) * 100
      ).toFixed(0)}% and it floors at 25% by the 4th+.`,
      source:
        "home-mixer/scorers/ranking_scorer.rs: diversity_multiplier(decay=0.5, floor=0.25)",
      triggered: req.recentPostsCount >= 1,
    },
    {
      id: "oon-discount",
      severity: "info",
      title: "Reach beyond your followers is discounted by 25%",
      detail: `Every post is scored at full strength for your own followers. For everyone else (out-of-network recommendation), the final score is multiplied by 0.75, and that surface also runs extra spam/abuse-only drop rules your followers never see. If a post lands on a topic-matched recommendation surface specifically, the discount is steeper still — ${(
        (1 - TOPIC_OON_WEIGHT_FACTOR) *
        100
      ).toFixed(0)}% off, not 25%.`,
      source: "home-mixer/params/param.rs: OonWeightFactor = 0.75, TopicOonWeightFactor = 0.5",
      triggered: true,
    },
    {
      id: "reply-reach-limit",
      severity: "warning",
      title: "Replies structurally reach far less than an original post",
      detail:
        "This is more than a discount. OONRetweetReplyFilter removes replies and reposts from recommendations entirely for anyone who doesn't already follow you — they never enter that candidate pool at all, mutual or not. Even shown to your own followers, a reply/repost still gets rescored at the same 0.75x multiplier as out-of-network content (on by default). And replies are excluded from both the cold-start boost and the bidirectional mutual-follow reply-weight boost below — those only ever apply to original posts.",
      source:
        "home-mixer/filters/oon_retweet_reply_filter.rs; home-mixer/scorers/ranking_scorer.rs:744-753 (EnableOonRescoreForInNetworkRepliesRetweets, default true) and :180-183 (bidirectional_boost_eligible)",
      triggered: req.isReply,
    },
    {
      id: "post-age-cutoff",
      severity: "critical",
      title: "Past 48 hours old, a post stops being eligible for For You ranking at all",
      detail: `You told us this post is ${(req.postedHoursAgo ?? 0).toFixed(
        0
      )}h old. AgeFilter drops any candidate older than 48h before scoring even starts — not a downrank, a hard exclusion from recommendation candidates entirely. It can still be seen by followers scrolling their following feed or your profile, just not surfaced through For You ranking anymore.`,
      source: "home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs:347; home-mixer/params/config.rs:36 (MAX_POST_AGE = 48h)",
      triggered: (req.postedHoursAgo ?? 0) > 48,
    },
  ];

  if (req.authorFollowers !== undefined) {
    const ageHours = req.postedHoursAgo ?? 0;
    const eligible =
      req.authorFollowers <= COLD_START_FOLLOWER_CAP &&
      ageHours <= COLD_START_MAX_AGE_HOURS &&
      !req.isReply;
    risks.push({
      id: "cold-start-boost",
      severity: "info",
      title: eligible ? "Cold-start boost likely applies" : "Cold-start boost doesn't apply here",
      detail: eligible
        ? `With ${req.authorFollowers.toLocaleString()} followers (≤ ${COLD_START_FOLLOWER_CAP.toLocaleString()} cap) and posted ${
            ageHours < 1 ? "just now" : `${ageHours.toFixed(0)}h ago`
          } (≤ ${COLD_START_MAX_AGE_HOURS}h window), this post can get a real boost for some viewers — but it's not an automatic top slot. It first has to already be ranking somewhere in the top ${(
            COLD_START_MAX_POSITION_RATIO * 100
          ).toFixed(
            0
          )}% of that viewer's scored candidates on its own merits; only then does the boost lift its score up to match whatever's currently sitting around rank 15. And it only applies while the post is still under ${COLD_START_IMPRESSION_THRESHOLD.toLocaleString()} total views — once it crosses that, cold-start stops touching it. This is separate from, and on top of, the BESC Score above.`
        : req.isReply
          ? `Cold-start boosting only ever applies to original posts — cold_start_base_eligible() requires the candidate to not itself be a reply or repost. This one's a reply, so it's excluded regardless of follower count or age.`
          : `Cold-start boosting only kicks in under ${COLD_START_FOLLOWER_CAP.toLocaleString()} followers and within a ${COLD_START_MAX_AGE_HOURS}h posting window. At ${req.authorFollowers.toLocaleString()} followers${
              ageHours > COLD_START_MAX_AGE_HOURS ? ` and ${ageHours.toFixed(0)}h old` : ""
            }, this post doesn't currently qualify.`,
      source:
        "home-mixer/scorers/author_cold_start.rs:86-91,130-192 (cold_start_base_eligible requires in_reply_to_tweet_id/retweeted_tweet_id both None); home-mixer/params/param.rs (ColdStartFollowerCap=1000, ColdStartMaxPostAgeSecs=86400, ColdStartImpressionThreshold=1000, LowImpressionsMaxPositionRatio=0.85, ColdStartSlotMin/Max=15/16)",
      triggered: eligible,
    });
  }

  return risks;
}

function buildTips(f: FeatureReport, req: AnalyzeRequest, risks: RiskFlag[]): Tip[] {
  const tips: Tip[] = [];

  if (f.hasWeakOpener) {
    tips.push({
      id: "weak-opener",
      title: "Open with the point, not a wind-up",
      detail:
        'Starting with "I think", "just wanted to", or "so," buries the actual claim. People decide whether to keep reading in the first few words, so lead with the thing that matters.',
      impact: "high",
    });
  }

  if (f.fillerWords >= 2) {
    tips.push({
      id: "cut-filler",
      title: `Cut ${f.fillerWords} filler word${f.fillerWords > 1 ? "s" : ""} ("very", "just", "actually"...)`,
      detail:
        "Hedges and filler words dilute a claim without adding information. Tighter, more direct phrasing reads as more confident and holds attention better.",
      impact: "medium",
    });
  }

  if (f.hasAiSlopPhrasing) {
    tips.push({
      id: "cut-ai-slop",
      title: 'Cut the stock AI phrasing ("delve into", "game-changer", "unlock the power of"...)',
      detail:
        "X's abuse-enforcement pipeline has a classifier category specifically for this (\"llm_slop_post\") that attaches a 30-day label to posts matching it. This repo doesn't show exactly what that label restricts downstream, but a label that exists purely to flag AI-generated-sounding writing isn't neutral — and it's free to just write it in your own words instead.",
      impact: "medium",
    });
  }

  if (req.mediaType === "video") {
    tips.push({
      id: "video-min-duration",
      title: `Make sure the video runs ${(MIN_VIDEO_DURATION_MS / 1000).toFixed(0)}s or longer`,
      detail:
        "Video-quality-view is a real scored action, but its weight is forced to exactly 0.0 for any video under this length — a short clip earns nothing from that signal no matter how good it is.",
      impact: "medium",
    });
  }

  if (f.passiveVoiceRatio > 0.4) {
    tips.push({
      id: "active-voice",
      title: "Switch passive constructions to active voice",
      detail:
        '"The report was written by the team" reads weaker than "The team wrote the report." Active voice is more direct and easier to scan mid-scroll.',
      impact: "medium",
    });
  }

  if (f.replyHookKind === "none") {
    tips.push({
      id: "ask-question",
      title: "Give people a reason to reply, not just like",
      detail:
        "Reply is weighted 5.0–20.0 (with the bidirectional-follow boost), versus 0.5 for a like: 10 to 40× more valuable. End with a genuine question or a clear stance people will want to respond to.",
      impact: "high",
    });
  } else if (f.replyHookKind === "generic") {
    tips.push({
      id: "specific-question",
      title: "Swap the generic closer for a question only your readers can answer",
      detail:
        '"What do you think?" and "Thoughts?" ask nothing, so most people scroll past them. A question tied to the specifics of this post — one that assumes the reader actually read it — is what earns replies, and it avoids looking like the templated tail that duplicate-text spam detection is built to catch.',
      impact: "high",
    });
  }

  if (req.mediaType === "none") {
    tips.push({
      id: "add-media",
      title: "Consider attaching a photo or video",
      detail:
        "Media unlocks photo-expand/video-open and video-quality-view weights that a text-only post can never earn, and media posts tend to hold attention longer.",
      impact: "medium",
    });
  }

  if (f.words < 8) {
    tips.push({
      id: "too-short",
      title: "This post may be too thin to hold attention",
      detail:
        "Very short posts tend to score lower on dwell-related signals. Give the reader enough context to want to stop scrolling.",
      impact: "medium",
    });
  }

  if (f.words > 70 && !req.isVerified) {
    tips.push({
      id: "too-long",
      title: "Trim it down",
      detail:
        "Long, dense posts lose skimmers before they reach your point. Aim for a tight ~15-45 words, or restructure as a thread so each post pulls its own weight.",
      impact: "low",
    });
  } else if (f.words > 400 && req.isVerified) {
    tips.push({
      id: "too-long-verified",
      title: "Even long-form has a skim point",
      detail:
        "X Premium lets you write past 280 characters, but readers still decide fast whether to keep going. Put the actual point in the first couple of sentences instead of relying on length alone to hold attention.",
      impact: "low",
    });
  }

  if (risks.find((r) => r.id === "url-verdict")?.triggered) {
    tips.push({
      id: "fix-link",
      title: "Swap the shortened/redirect link for a direct, reputable URL",
      detail:
        "Post the real destination link directly instead of a shortener. Shortened and redirect-heavy links are exactly what URL-verdict spam checks re-resolve and distrust by default.",
      impact: "high",
    });
  }

  if (risks.find((r) => r.id === "duplicate-boilerplate")?.triggered) {
    tips.push({
      id: "de-boilerplate",
      title: 'Cut the "follow for follow" / "link in bio" style phrasing',
      detail:
        "Rewrite in your own voice. Generic broadcast phrasing is the fingerprint duplicate-text spam detection is built to catch across many accounts posting the same boilerplate.",
      impact: "high",
    });
  }

  if (f.allCapsWordRatio > 0.25 || f.exclamationBursts > 0) {
    tips.push({
      id: "tone-down",
      title: "Dial back the ALL-CAPS / !!! energy",
      detail:
        "Report (-234), Mute (-58.8), and Not-interested (-43.2) are by far the largest weights in the whole model, and they're negative. It takes very little of that reaction to wipe out a lot of likes.",
      impact: "high",
    });
  }

  if (f.hashtags > 4) {
    tips.push({
      id: "cut-hashtags",
      title: "Cut hashtags down to 1-2 relevant ones",
      detail:
        "Beyond a couple of targeted hashtags, more tags read as spam and add nothing. Reach comes from the ranking model, not from hashtag volume.",
      impact: "medium",
    });
  }

  if (!f.hasNumbers && !f.hasThreadMarker && f.words > 15) {
    tips.push({
      id: "make-shareable",
      title: "Give it a concrete hook: a stat, a claim, a story beat",
      detail:
        "Share via copy-link is the single highest-weighted action in the model (20.0). Concrete, specific claims are what people actually screenshot or copy-link to send to a friend.",
      impact: "medium",
    });
  }

  if (!req.isReply && req.hasMutualFollowAudience) {
    tips.push({
      id: "mutual-original-post-bonus",
      title: "Original post to a mutual-follow audience: +15.0 reply-weight bonus applies",
      detail:
        "Verified in ranking_scorer.rs: bidirectional_boost_eligible() requires the post to NOT be a reply or repost. This bonus rewards original content shown to people who follow you back, not replies — it never applies to a reply, even one addressed to a mutual.",
      impact: "low",
    });
  } else if (req.isReply && req.hasMutualFollowAudience) {
    tips.push({
      id: "mutual-reply-no-bonus",
      title: "This bonus doesn't apply here — it's reply-only excluded",
      detail:
        "The +15.0 bidirectional mutual-follow bonus only ever applies to original posts, never to replies, regardless of who they're addressed to. If reach matters more than replying to this specific thread, an original post to the same audience would pick up this bonus and this reply won't.",
      impact: "low",
    });
  }

  if (req.recentPostsCount >= 2) {
    tips.push({
      id: "spread-out",
      title: "Space your posts out",
      detail: `At post #${
        req.recentPostsCount + 1
      } in this window, author-diversity decay has already pushed the multiplier toward the 25% floor for repeat viewers. Spread posts across the day instead of stacking them.`,
      impact: "medium",
    });
  }

  return tips;
}

export function analyzePost(
  req: AnalyzeRequest,
  calibration?: CalibrationModel | null,
  llmMultipliers?: ActionMultipliers | null
): ScoreResult {
  const features = extractFeatures(req.text, req.link);
  const heuristic = estimateActionProbabilities(features, req);
  const probabilities = applyLlmMultipliers(
    applyCalibration(heuristic, features, req, calibration),
    llmMultipliers,
    calibration
  );
  const actions = buildActionRows(probabilities, req);

  const positiveContribution = actions
    .filter((a) => a.contribution > 0)
    .reduce((s, a) => s + a.contribution, 0);
  const negativeContribution = actions
    .filter((a) => a.contribution < 0)
    .reduce((s, a) => s + a.contribution, 0);

  const rawScore = positiveContribution + negativeContribution;

  const diversityMultiplier = authorDiversityMultiplier(req.recentPostsCount);
  const adjustedRaw = rawScore * diversityMultiplier;

  // Squash the unbounded weighted sum into a 0-100 "BESC Score".
  // Calibrated so a strong, clean, high-engagement draft lands ~85-95 and a
  // spam-flagged / low-effort draft lands well under 30.
  const SQUASH_C = 1.9;
  let score: number;
  if (adjustedRaw >= 0) {
    score = 100 * (adjustedRaw / (adjustedRaw + SQUASH_C));
  } else {
    score = Math.max(0, 8 + adjustedRaw); // negative scores compress hard toward 0
  }
  score = Math.round(clamp01(score / 100) * 1000) / 10;

  let grade: string;
  if (score >= 82) grade = "Excellent";
  else if (score >= 58) grade = "Strong";
  else if (score >= 38) grade = "Decent";
  else if (score >= 20) grade = "Weak";
  else grade = "High Risk";

  const risks = buildRisks(features, req);
  const tips = buildTips(features, req, risks);

  return {
    score,
    grade,
    rawScore,
    positiveContribution,
    negativeContribution,
    actions,
    risks,
    tips,
    features,
    authorDiversityMultiplier: diversityMultiplier,
    oonWeightFactor: OON_WEIGHT_FACTOR,
  };
}
