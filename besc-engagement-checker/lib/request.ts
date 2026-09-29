import type { AnalyzeRequest, GenerateRequest, MediaType } from "./types";

// Hard payload cap, not the platform post limit — kept above
// VERIFIED_CHAR_LIMIT (4,000, see scoring.ts) so a legitimate long-form
// verified draft doesn't get silently truncated before it's even scored.
const MAX_CHARS = 6000;
// Loose context for generation is naturally shorter than a full draft —
// this is a sanity cap against pasting something enormous, not a platform limit.
const MAX_CONTEXT_CHARS = 2000;
const VALID_MEDIA: MediaType[] = ["none", "photo", "video", "gif"];

function parseMediaType(value: unknown): MediaType {
  return VALID_MEDIA.includes(value as MediaType) ? (value as MediaType) : "none";
}

function parseOptionalNonNegative(value: unknown): number | undefined {
  return value !== undefined && value !== null ? Math.max(0, Number(value) || 0) : undefined;
}

interface SharedContextFields {
  mediaType: MediaType;
  link: string;
  isReply: boolean;
  hasMutualFollowAudience: boolean;
  recentPostsCount: number;
  nsfw: boolean;
  authorFollowers?: number;
  postedHoursAgo?: number;
  isVerified: boolean;
}

function parseSharedContextFields(body: Partial<AnalyzeRequest & GenerateRequest>): SharedContextFields {
  return {
    mediaType: parseMediaType(body.mediaType),
    link: typeof body.link === "string" ? body.link.slice(0, 500) : "",
    isReply: Boolean(body.isReply),
    hasMutualFollowAudience: Boolean(body.hasMutualFollowAudience),
    recentPostsCount: Math.max(0, Math.min(10, Number(body.recentPostsCount) || 0)),
    nsfw: Boolean(body.nsfw),
    authorFollowers: parseOptionalNonNegative(body.authorFollowers),
    postedHoursAgo: parseOptionalNonNegative(body.postedHoursAgo),
    isVerified: Boolean(body.isVerified),
  };
}

export function parseAnalyzeRequest(body: Partial<AnalyzeRequest>): AnalyzeRequest | null {
  const text = typeof body.text === "string" ? body.text.slice(0, MAX_CHARS) : "";
  if (!text.trim()) return null;

  return { text, ...parseSharedContextFields(body) };
}

export function parseGenerateRequest(body: Partial<GenerateRequest>): GenerateRequest | null {
  const context = typeof body.context === "string" ? body.context.slice(0, MAX_CONTEXT_CHARS) : "";
  if (!context.trim()) return null;

  return { context, ...parseSharedContextFields(body) };
}
