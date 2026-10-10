export type MediaType = "none" | "photo" | "video" | "gif";

export interface AnalyzeRequest {
  text: string;
  mediaType: MediaType;
  link: string;
  /** Is this draft itself a reply to someone else's post (not a quote post)? */
  isReply: boolean;
  /** Does a meaningful share of your followers also follow you back? Only
   * matters when isReply is false — see WEIGHTS.bidirectionalReplyBoost. */
  hasMutualFollowAudience: boolean;
  recentPostsCount: number;
  nsfw: boolean;
  /** Your follower count, if known — drives the cold-start boost check. Omit if unknown. */
  authorFollowers?: number;
  /** Hours since this post went live. Omit/0 for a not-yet-posted draft (posting now). */
  postedHoursAgo?: number;
  /** X Premium / verified checkmark — raises the character limit from 280 to 4,000+. */
  isVerified?: boolean;
}

export interface ActionRow {
  id: string;
  label: string;
  group: "engagement" | "clicks" | "attention" | "author" | "negative";
  weight: number;
  probability: number;
  contribution: number;
  weightSource: string;
}

export type RiskSeverity = "critical" | "warning" | "info";

export interface RiskFlag {
  id: string;
  severity: RiskSeverity;
  title: string;
  detail: string;
  source: string;
  triggered: boolean;
}

export interface Tip {
  id: string;
  title: string;
  detail: string;
  impact: "high" | "medium" | "low";
}

export interface FeatureReport {
  chars: number;
  words: number;
  hashtags: number;
  mentions: number;
  links: number;
  emojis: number;
  allCapsWordRatio: number;
  exclamationBursts: number;
  hasQuestion: number;
  hasReplyCTA: boolean;
  /** Whether the post asks something specific to its own content, a generic closer, or nothing. */
  replyHookKind: "specific" | "generic" | "none";
  hasShareCTA: boolean;
  hasBoilerplateCTA: boolean;
  hasAiSlopPhrasing: boolean;
  hasNumbers: boolean;
  hasThreadMarker: boolean;
  fillerWords: number;
  passiveVoiceRatio: number;
  hasWeakOpener: boolean;
  urlRisk: boolean;
  urlReason: string | null;
}

export interface AuthorLookupResult {
  authorHandle: string;
  authorName: string;
  authorFollowers: number;
  authorVerified: boolean;
}

export interface TweetImportResult {
  text: string;
  mediaType: MediaType;
  link: string;
  /** Was this tweet itself posted as a reply (in_reply_to_status_id present)? */
  isReply: boolean;
  /** Vee3's possibly_sensitive flag, if present. */
  nsfw: boolean;
  authorHandle: string;
  authorName: string;
  authorFollowers: number;
  authorVerified: boolean;
  recentPostsCount: number;
  postedHoursAgo: number;
  realMetrics: {
    views: number;
    likes: number;
    retweets: number;
    replies: number;
    quotes: number;
    bookmarks: number;
  };
}

export interface OptimizeStep {
  id: string;
  label: string;
  reason: string;
  scoreBefore: number;
  scoreAfter: number;
}

export interface AICandidate {
  text: string;
  score: number;
}

export type AIStatus = "disabled" | "error" | "no_improvement" | "found";

/** Same contextual fields as AnalyzeRequest, but a loose idea instead of finished text. */
export interface GenerateRequest {
  context: string;
  mediaType: MediaType;
  link: string;
  isReply: boolean;
  hasMutualFollowAudience: boolean;
  recentPostsCount: number;
  nsfw: boolean;
  authorFollowers?: number;
  postedHoursAgo?: number;
  isVerified?: boolean;
}

export type GenerateStatus = "disabled" | "error" | "empty" | "found";

export interface GenerateResult {
  generateStatus: GenerateStatus;
  /** Each candidate is already the deterministic-optimizer's polished version of what the model wrote, sorted by score descending. */
  candidates?: AICandidate[];
}

export interface OptimizeResult {
  originalText: string;
  optimizedText: string;
  applied: OptimizeStep[];
  before: ScoreResult;
  after: ScoreResult;
  /** Always present so the UI can show real AI status instead of silently omitting the section. */
  aiStatus: AIStatus;
  /** Populated only when aiStatus is "found" — each candidate scores strictly higher than the deterministic result. */
  aiCandidates?: AICandidate[];
}

export interface TrackedPostMetrics {
  views: number;
  likes: number;
  retweets: number;
  replies: number;
  quotes: number;
  bookmarks: number;
}

export interface TrackedPost {
  id: number;
  createdAt: string;
  draftText: string;
  predictedScore: number;
  predictedGrade: string;
  appliedFixIds: string[];
  isReply: boolean;
  mediaType: MediaType;
  /** null until this draft has been matched to a real published post. */
  tweetId: string | null;
  postedAt: string | null;
  metricsUpdatedAt: string | null;
  metrics: TrackedPostMetrics | null;
}

export interface CalibrationSide {
  n: number;
  medianPredicted: number;
  medianViews: number;
  medianEngagements: number;
}

export interface FixInsight {
  fixId: string;
  label: string;
  withN: number;
  withoutN: number;
  medianEngagementsWith: number;
  medianEngagementsWithout: number;
  /**
   * Multiplier vs. posts without the fix; 1.0 = no observed difference.
   * null when the baseline is 0 and no honest ratio exists — the UI shows
   * the absolute medians instead of inventing an infinite lift.
   */
  lift: number | null;
}

export interface TrackSummary {
  tracked: number;
  measured: number;
  /** Minimum measured posts before any calibration claim is made at all. */
  minimumForInsights: number;
  /** null until `measured` clears minimumForInsights — small-N engagement data is noise. */
  scoreSplit: { higher: CalibrationSide; lower: CalibrationSide } | null;
  fixInsights: FixInsight[];
}

export interface TrackRecord {
  posts: TrackedPost[];
  summary: TrackSummary;
}

export interface ScoreResult {
  score: number;
  grade: string;
  rawScore: number;
  positiveContribution: number;
  negativeContribution: number;
  actions: ActionRow[];
  risks: RiskFlag[];
  tips: Tip[];
  features: FeatureReport;
  authorDiversityMultiplier: number;
  oonWeightFactor: number;
  /** 0 = pure heuristic, →1 = score largely driven by a model fitted to this author's real results. */
  calibrationStrength?: number;
}

/**
 * A language model's read of one draft, expressed as multiples of this
 * author's typical post. Both scores come back so the UI can show what the
 * model's opinion actually changed rather than silently replacing a number.
 */
export interface AiEstimateResponse {
  multipliers: Record<string, number>;
  baselineScore: number;
  result: ScoreResult;
  /** Whether this estimator has been measured against the author's real posts, and how it did. */
  evaluation: { llmWins: boolean; n: number; evaluatedAt: string } | null;
}

/**
 * The actions a language model is asked to judge. Fewer than the full weight
 * table: it can only reason about actions a reader would recognise from the
 * text, so link clicks, dwell and video views stay heuristic.
 */
export const LLM_ACTIONS = [
  "favorite",
  "reply",
  "retweet",
  "quote",
  "shareViaCopyLink",
  "followAuthor",
  "notInterested",
  "muteAuthor",
  "report",
] as const;
export type LlmAction = (typeof LLM_ACTIONS)[number];

/** Multiples of this author's typical post, per action. Missing keys stay heuristic. */
export type ActionMultipliers = Partial<Record<LlmAction, number>>;
