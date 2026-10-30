import {
  WEIGHTS,
  AUTHOR_DIVERSITY_DECAY,
  AUTHOR_DIVERSITY_FLOOR,
  OON_WEIGHT_FACTOR,
  TOPIC_OON_WEIGHT_FACTOR,
  COLD_START_FOLLOWER_CAP,
  COLD_START_MAX_AGE_HOURS,
  COLD_START_IMPRESSION_THRESHOLD,
  MIN_VIDEO_DURATION_MS,
  BOILERPLATE_PHRASES,
  AI_SLOP_PHRASES,
  GENERIC_CLOSERS,
} from "./scoring";
import type { AnalyzeRequest, GenerateRequest } from "./types";

/**
 * The working knowledge an LLM needs to write for this ranking system, built
 * from the same constants the scorer uses so it can never drift out of sync
 * with what the tool actually rewards.
 *
 * This is deliberately mechanics, not source code. Pasting ranking_scorer.rs
 * at a model would burn thousands of tokens teaching it Rust; what changes
 * what a good post looks like is the *consequences* — which actions are worth
 * chasing, which are catastrophic, and which structural choices cap reach
 * before the text matters at all.
 */
export function buildAlgorithmBrief(): string {
  const replyBoosted = WEIGHTS.reply + WEIGHTS.bidirectionalReplyBoost;
  const reportInLikes = Math.round(Math.abs(WEIGHTS.report) / WEIGHTS.favorite);
  const secondPostPct = Math.round(
    ((1 - AUTHOR_DIVERSITY_FLOOR) * AUTHOR_DIVERSITY_DECAY + AUTHOR_DIVERSITY_FLOOR) * 100
  );

  return `HOW THIS RANKING SYSTEM ACTUALLY WORKS (from X's open-sourced For You algorithm)

Every post is scored as: Final Score = Σ (weight × probability of each action).
Positive weights, highest first:
  ${WEIGHTS.shareViaCopyLink} share-via-copy-link — the single heaviest action in the model
  ${WEIGHTS.reply} reply (${replyBoosted} on an original post shown to a mutual follower)
  ${WEIGHTS.quote} quote · ${WEIGHTS.shareViaDm} share-via-DM · ${WEIGHTS.followAuthor} follow-author · ${WEIGHTS.share} share · ${WEIGHTS.retweet} repost
  ${WEIGHTS.click} click · ${WEIGHTS.openLink} open-link · ${WEIGHTS.photoExpand} photo-expand · ${WEIGHTS.videoOpen} video-open · ${WEIGHTS.vqv} video-quality-view
  ${WEIGHTS.favorite} like — the LOWEST positive weight. Writing for likes optimises for the least valuable action.
Negative weights:
  ${WEIGHTS.report} report · ${WEIGHTS.muteAuthor} mute · ${WEIGHTS.notInterested} not-interested · ${WEIGHTS.blockAuthor} block · ${WEIGHTS.notDwelled} scrolled-past

WHAT THAT MATH MEANS IN PRACTICE
- One report costs the equivalent of ~${reportInLikes} likes. The negative weights dwarf everything else, so anything that reads as spam, bait or shouting is catastrophic, not merely suboptimal. Never trade goodwill for a cheap reaction.
- A reply is worth 10–40× a like. A copy-link share is worth 40× a like. Those two are what a good post is engineered for.
- Copy-link shares come from posts worth sending to one specific person: a concrete number, a sharp claim, a story beat, something genuinely useful. Vague inspirational posts never earn them.

STRUCTURAL LIMITS THAT CAP REACH BEFORE THE WRITING MATTERS
- Replies and reposts are removed entirely from recommendations to non-followers, and are rescored at ${OON_WEIGHT_FACTOR}× even for followers. They also never receive the mutual-follow reply boost or the new-author boost.
- Reach beyond your own followers is multiplied by ${OON_WEIGHT_FACTOR} (${TOPIC_OON_WEIGHT_FACTOR} on topic-matched surfaces).
- Each additional post in the same window is multiplied by (1 − ${AUTHOR_DIVERSITY_FLOOR}) × ${AUTHOR_DIVERSITY_DECAY}^k + ${AUTHOR_DIVERSITY_FLOOR} — a 2nd post scores ~${secondPostPct}%, flooring at ${AUTHOR_DIVERSITY_FLOOR * 100}%.
- Posts older than 48 hours stop being eligible for ranking at all.
- New-author boost applies only under ${COLD_START_FOLLOWER_CAP.toLocaleString()} followers, within ${COLD_START_MAX_AGE_HOURS}h, and under ${COLD_START_IMPRESSION_THRESHOLD.toLocaleString()} views — and only lifts a post that is already ranking well on its own merits.
- A video under ${MIN_VIDEO_DURATION_MS / 1000}s has its video-quality-view weight forced to exactly 0.

VISIBILITY FILTERING — separate from ranking, and it can drop a post outright
- Shortened/redirect links, raw-IP links and cheap TLDs draw URL-verdict scrutiny. An "unsafe" verdict is a hard drop for every non-follower, and a bad domain verdict retroactively labels past posts too.
- Templated broadcast phrasing ("${BOILERPLATE_PHRASES.slice(0, 3).join('", "')}") matches duplicate-text spam detection, which runs across accounts — identical, templated text is the exact fingerprint it looks for.
- Content that reads as machine-generated filler is classified as "llm_slop_post" and labelled for 30 days. Stock AI phrasing like "${AI_SLOP_PHRASES.slice(0, 4).join('", "')}" is the tell.
- ALL-CAPS, !!!/???, and more than 2 hashtags all push up the three most negative weights.

WRITING THE REPLY HOOK — the highest-leverage sentence in the post
A reply hook is only worth its weight if it asks something a reader can only answer having actually read THIS post. Generic closers are near-worthless: they ask nothing, so people scroll past, and a tool appending the identical tail to thousands of posts manufactures the very templated-text pattern spam detection catches.
NEVER end with, or any close variant of: "${GENERIC_CLOSERS.slice(0, 6).join('", "')}".
Hooks that actually earn replies:
  - A specific either/or drawn from the post's own content ("Ship the migration first or the API rewrite?")
  - A question only someone with relevant experience can answer ("What broke first when you scaled past 10k writes/sec?")
  - An invitation to contradict a concrete claim you just made, with the claim stated plainly enough to argue with
  - Asking for the reader's number/example to compare against a number you gave
The question must be answerable in one line, be genuinely open (not rhetorical), and reference something concrete from the post itself.`;
}

/** Post-specific constraints, so the model writes for this exact situation. */
export function buildSituationBrief(req: AnalyzeRequest | GenerateRequest, charLimit: number): string {
  const lines: string[] = [`- Hard limit: ${charLimit.toLocaleString()} characters.`];

  if (req.isReply) {
    lines.push(
      `- This is a REPLY. It cannot reach non-followers at all and gets no mutual-follow or new-author boost, so write it to earn a reply from people already in the thread rather than for broad reach.`
    );
  } else if (req.hasMutualFollowAudience) {
    lines.push(
      `- Original post to a mutual-follow audience: the reply weight rises to ${WEIGHTS.reply + WEIGHTS.bidirectionalReplyBoost}, the largest situational boost available. A strong, specific question is worth more here than anywhere else.`
    );
  }

  if (req.mediaType === "video") {
    lines.push(
      `- A video is attached: it must run ${MIN_VIDEO_DURATION_MS / 1000}s+ to earn video-quality-view at all. Don't write a caption that duplicates what the video already says.`
    );
  } else if (req.mediaType === "photo" || req.mediaType === "gif") {
    lines.push(`- Media is attached, so the text should add context the image can't carry, not describe it.`);
  } else {
    lines.push(`- Text-only, so the words carry the entire post — no media actions are available to it.`);
  }

  if (req.recentPostsCount >= 1) {
    const mult = Math.round(
      ((1 - AUTHOR_DIVERSITY_FLOOR) * Math.pow(AUTHOR_DIVERSITY_DECAY, req.recentPostsCount) +
        AUTHOR_DIVERSITY_FLOOR) * 100
    );
    lines.push(
      `- This is post #${req.recentPostsCount + 1} in the current window, so it's already multiplied to ~${mult}% by author-diversity decay. It has to be strong enough to be worth that penalty.`
    );
  }

  if (req.nsfw) {
    lines.push(`- Marked sensitive: it is dropped from recommendations entirely, so it only reaches existing followers.`);
  }

  if (!req.link) {
    lines.push(`- No link is attached, and you must not invent one.`);
  }

  return `THIS SPECIFIC POST\n${lines.join("\n")}`;
}
