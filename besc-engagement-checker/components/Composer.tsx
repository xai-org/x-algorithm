"use client";

import { useEffect, useRef, useState } from "react";
import {
  Image as ImageIcon,
  Film,
  Sticker,
  Ban,
  Link2,
  Users,
  Repeat2,
  Download,
  Loader2,
  Wand2,
  Check,
  X,
  Rocket,
  Sparkles,
  BadgeCheck,
  AtSign,
  Undo2,
  Reply,
  Copy,
  ChevronDown,
  Activity,
  SlidersHorizontal,
} from "lucide-react";
import type {
  AiEstimateResponse,
  AnalyzeRequest,
  AuthorLookupResult,
  GenerateRequest,
  GenerateResult,
  MediaType,
  OptimizeResult,
  ScoreResult,
  TweetImportResult,
} from "@/lib/types";
import { getCharLimit } from "@/lib/scoring";
import AiReadPanel from "./AiReadPanel";

// A candidate has to clear the deterministic score by this many BESC-score
// points before it's applied automatically instead of requiring a manual
// "Use this version" click. Below this margin the improvement could be
// noise-level, and this app has a real history of AI rewrites drifting a
// claim's meaning — auto-applying anything and everything isn't worth it.
const AI_AUTO_APPLY_MARGIN = 5;

const MEDIA_OPTIONS: { id: MediaType; label: string; icon: typeof ImageIcon }[] = [
  { id: "none", label: "Text only", icon: Ban },
  { id: "photo", label: "Photo", icon: ImageIcon },
  { id: "video", label: "Video", icon: Film },
  { id: "gif", label: "GIF", icon: Sticker },
];

export default function Composer({
  onResult,
  onLoading,
  onImport,
  handleInput,
  onHandleChange,
  onTracked,
}: {
  onResult: (r: ScoreResult | null) => void;
  onLoading: (b: boolean) => void;
  onImport?: (r: TweetImportResult | null) => void;
  /** Lifted so the same handle drives both author lookup and post tracking. */
  handleInput: string;
  onHandleChange: (handle: string) => void;
  onTracked?: () => void;
}) {
  const [text, setText] = useState(
    "Just shipped something I've been heads-down on for weeks. What's one thing you'd want it to do next?"
  );
  const [mediaType, setMediaType] = useState<MediaType>("photo");
  const [link, setLink] = useState("");
  const [isReply, setIsReply] = useState(false);
  const [hasMutualFollowAudience, setHasMutualFollowAudience] = useState(false);
  const [nsfw, setNsfw] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [recentPostsCount, setRecentPostsCount] = useState(0);
  const [authorFollowers, setAuthorFollowers] = useState("");
  const [postedHoursAgo, setPostedHoursAgo] = useState(0);
  // Mirrors what's handed to onResult, so tracking can record the exact score
  // the user is looking at without the parent having to hand it back down.
  const [lastResult, setLastResult] = useState<ScoreResult | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  function parsedFollowers(): number | undefined {
    if (!authorFollowers.trim()) return undefined;
    const n = Number(authorFollowers);
    return Number.isFinite(n) && n >= 0 ? n : undefined;
  }

  const [importUrl, setImportUrl] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const [lookingUpHandle, setLookingUpHandle] = useState(false);
  const [handleLookupError, setHandleLookupError] = useState<string | null>(null);

  async function lookupHandle() {
    if (!handleInput.trim() || lookingUpHandle) return;
    setLookingUpHandle(true);
    setHandleLookupError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);
    try {
      const res = await fetch("/api/lookup-author", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handle: handleInput.trim() }),
        signal: controller.signal,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Lookup failed");
      const result = data as AuthorLookupResult;
      setAuthorFollowers(String(result.authorFollowers));
      setIsVerified(result.authorVerified);
    } catch (e) {
      const err = e as Error;
      setHandleLookupError(err.name === "AbortError" ? "Lookup timed out. Try again." : err.message);
    } finally {
      clearTimeout(timeoutId);
      setLookingUpHandle(false);
    }
  }

  const [optimizeResult, setOptimizeResult] = useState<OptimizeResult | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  const [optimizeError, setOptimizeError] = useState<string | null>(null);
  const [aiAutoApplied, setAiAutoApplied] = useState(false);
  const [preAiText, setPreAiText] = useState<string | null>(null);

  async function optimize() {
    if (!text.trim() || optimizing) return;
    setOptimizing(true);
    setOptimizeError(null);
    setOptimizeResult(null);
    setAiAutoApplied(false);
    setPreAiText(null);
    const textBeforeOptimize = text;
    try {
      const payload: AnalyzeRequest = {
        text,
        mediaType,
        link,
        isReply,
        hasMutualFollowAudience,
        recentPostsCount,
        nsfw,
        authorFollowers: parsedFollowers(),
        postedHoursAgo,
        isVerified,
      };
      const controller = new AbortController();
      // Kept above lib/ollama.ts's own 100s request timeout so a slow AI
      // call resolves to a graceful "aiStatus: error" fallback from the
      // server, instead of the client aborting first and losing the
      // deterministic result along with it.
      const timeoutId = setTimeout(() => controller.abort(), 115000);
      let res: Response;
      try {
        res = await fetch("/api/optimize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload, handle: handleInput.trim() }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeoutId);
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Optimize failed");
      const result = data as OptimizeResult;
      setOptimizeResult(result);

      const topCandidate = result.aiStatus === "found" ? result.aiCandidates?.[0] : undefined;
      if (topCandidate && topCandidate.score - result.after.score >= AI_AUTO_APPLY_MARGIN) {
        setPreAiText(textBeforeOptimize);
        setText(topCandidate.text);
        setAiAutoApplied(true);
        onImport?.(null);
      }
    } catch (e) {
      const err = e as Error;
      setOptimizeError(err.name === "AbortError" ? "Optimize timed out. Try again." : err.message);
    } finally {
      setOptimizing(false);
    }
  }

  const [aiRead, setAiRead] = useState<AiEstimateResponse | null>(null);
  const [aiReading, setAiReading] = useState(false);
  const [aiReadError, setAiReadError] = useState<string | null>(null);
  // The exact text the AI read applies to. Once the draft moves on, the read
  // is about a post that no longer exists, so it stops being shown rather
  // than sitting there attached to the wrong words.
  const [aiReadText, setAiReadText] = useState<string | null>(null);

  async function getAiRead() {
    if (!text.trim() || aiReading) return;
    setAiReading(true);
    setAiReadError(null);
    setAiRead(null);
    const readOf = text;
    try {
      const payload: AnalyzeRequest = {
        text,
        mediaType,
        link,
        isReply,
        hasMutualFollowAudience,
        recentPostsCount,
        nsfw,
        authorFollowers: parsedFollowers(),
        postedHoursAgo,
        isVerified,
      };
      const res = await fetch("/api/ai-estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, handle: handleInput.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "The AI read failed");
      const estimate = data as AiEstimateResponse;
      setAiRead(estimate);
      setAiReadText(readOf);
      // Show the full breakdown under the AI's probabilities, not just a
      // headline number — the next keystroke reverts it to the deterministic
      // score, which is the right default.
      setLastResult(estimate.result);
      onResult(estimate.result);
    } catch (e) {
      setAiReadError((e as Error).message);
    } finally {
      setAiReading(false);
    }
  }

  // Collapsed by default: these are secondary metadata, and leaving them open
  // buried the writing surface under three screens of toggles. The summary on
  // the button means anything non-default is still visible at a glance.
  const [showContext, setShowContext] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [contextIdea, setContextIdea] = useState("");
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);

  async function generateFromContext() {
    if (!contextIdea.trim() || generating) return;
    setGenerating(true);
    setGenerateError(null);
    setGenerateResult(null);
    try {
      const payload: GenerateRequest = {
        context: contextIdea,
        mediaType,
        link,
        isReply,
        hasMutualFollowAudience,
        recentPostsCount,
        nsfw,
        authorFollowers: parsedFollowers(),
        postedHoursAgo,
        isVerified,
      };
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 115000);
      let res: Response;
      try {
        res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload, handle: handleInput.trim() }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeoutId);
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Generate failed");
      setGenerateResult(data as GenerateResult);
    } catch (e) {
      const err = e as Error;
      setGenerateError(err.name === "AbortError" ? "Generate timed out. Try again." : err.message);
    } finally {
      setGenerating(false);
    }
  }

  function useGeneratedCandidate(candidateText: string) {
    applyText(candidateText);
    setGenerateResult(null);
    setContextIdea("");
    setShowGenerate(false);
  }

  function applyText(newText: string) {
    setText(newText);
    setOptimizeResult(null);
    setAiAutoApplied(false);
    setPreAiText(null);
    onImport?.(null);
  }

  function undoAiAutoApply() {
    if (preAiText !== null) setText(preAiText);
    setAiAutoApplied(false);
    setPreAiText(null);
  }

  async function importFromUrl() {
    if (!importUrl.trim() || importing) return;
    setImporting(true);
    setImportError(null);
    onImport?.(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);
    try {
      const res = await fetch("/api/fetch-tweet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: importUrl.trim() }),
        signal: controller.signal,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Import failed");
      const imported = data as TweetImportResult;
      setText(imported.text);
      setMediaType(imported.mediaType);
      setLink(imported.link);
      setIsReply(imported.isReply);
      // hasMutualFollowAudience is deliberately left alone here — it's an
      // aggregate fact about your overall follower composition, not
      // something a single tweet lookup can tell us. Still yours to set.
      setNsfw(imported.nsfw);
      setRecentPostsCount(imported.recentPostsCount);
      setAuthorFollowers(String(imported.authorFollowers));
      setPostedHoursAgo(imported.postedHoursAgo);
      setIsVerified(imported.authorVerified);
      onImport?.(imported);
    } catch (e) {
      const err = e as Error;
      setImportError(err.name === "AbortError" ? "Import timed out. Try again." : err.message);
    } finally {
      clearTimeout(timeoutId);
      setImporting(false);
    }
  }

  async function analyze() {
    if (!text.trim()) {
      onResult(null);
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    onLoading(true);
    try {
      const payload: AnalyzeRequest = {
        text,
        mediaType,
        link,
        isReply,
        hasMutualFollowAudience,
        recentPostsCount,
        nsfw,
        authorFollowers: parsedFollowers(),
        postedHoursAgo,
        isVerified,
      };
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // handle is sent so scoring can use a model fitted to this author's
        // own results when one exists; it's ignored otherwise.
        body: JSON.stringify({ ...payload, handle: handleInput.trim() }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error("analyze failed");
      const data: ScoreResult = await res.json();
      setLastResult(data);
      onResult(data);
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        onResult(null);
      }
    } finally {
      if (controllerRef.current === controller) onLoading(false);
    }
  }

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(analyze, 350);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, mediaType, link, isReply, hasMutualFollowAudience, nsfw, recentPostsCount, authorFollowers, postedHoursAgo, isVerified]);

  // Surfaces whatever is set to something other than its default, so a
  // collapsed panel never hides a setting that's changing the score.
  const contextSummary = [
    isReply ? "reply" : null,
    hasMutualFollowAudience ? "mutual audience" : null,
    nsfw ? "sensitive" : null,
    isVerified ? "verified" : null,
    link.trim() ? "link" : null,
    recentPostsCount > 0 ? `${recentPostsCount} recent` : null,
    handleInput.trim() ? `@${handleInput.trim().replace(/^@/, "")}` : null,
  ]
    .filter(Boolean)
    .join(", ");

  const charLimit = getCharLimit(isVerified);
  const chars = text.length;
  const overLimit = chars > charLimit;

  const [copied, setCopied] = useState(false);
  const [tracking, setTracking] = useState(false);
  const [trackState, setTrackState] = useState<"idle" | "saved">("idle");
  const [trackError, setTrackError] = useState<string | null>(null);

  async function trackThisPost() {
    if (!text.trim() || tracking || !lastResult) return;
    setTracking(true);
    setTrackError(null);
    try {
      // Only claim fixes that are actually baked into the text being saved.
      // If the user edited after optimizing, the optimized text no longer
      // matches and attributing those fixes to this post would poison the
      // very calibration data this feature exists to produce.
      const appliedFixIds =
        optimizeResult && optimizeResult.optimizedText === text
          ? optimizeResult.applied.map((step) => step.id)
          : [];

      const res = await fetch("/api/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          handle: handleInput.trim(),
          draftText: text,
          predictedScore: lastResult.score,
          predictedGrade: lastResult.grade,
          appliedFixIds,
          mediaType,
          isReply,
          hasMutualFollowAudience,
          isVerified,
          recentPostsCount,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Couldn't track that post");
      if (data.enabled === false) {
        throw new Error("Tracking isn't set up on this deployment yet.");
      }
      setTrackState("saved");
      setTimeout(() => setTrackState("idle"), 2500);
      onTracked?.();
    } catch (e) {
      setTrackError((e as Error).message);
    } finally {
      setTracking(false);
    }
  }

  async function copyDraftToClipboard() {
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can fail (insecure context, denied permission) —
      // not worth surfacing an error for a convenience action.
    }
  }

  return (
    <div className="glass-panel p-5 sm:p-6">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-display text-lg font-semibold">Draft your post</h2>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={trackThisPost}
            disabled={!text.trim() || tracking || !lastResult}
            title={
              handleInput.trim()
                ? "Save this draft, then check back after you publish it to see how it really did"
                : "Add your @handle below first — it's how this finds the post you publish"
            }
            className="flex items-center gap-1 text-xs font-medium text-white/50 transition-colors hover:text-white/80 disabled:cursor-not-allowed disabled:opacity-30"
          >
            {trackState === "saved" ? (
              <Check className="h-3.5 w-3.5 text-besc-300" />
            ) : tracking ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Activity className="h-3.5 w-3.5" />
            )}
            {trackState === "saved" ? "Tracking" : "Track"}
          </button>
          <button
            type="button"
            onClick={copyDraftToClipboard}
            disabled={!text.trim()}
            className="flex items-center gap-1 text-xs font-medium text-white/50 transition-colors hover:text-white/80 disabled:cursor-not-allowed disabled:opacity-30"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-besc-300" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
            {copied ? "Copied" : "Copy"}
          </button>
          <span
            className={`font-mono text-xs tabular-nums ${
              overLimit ? "text-danger" : "text-white/35"
            }`}
          >
            {chars.toLocaleString()}/{charLimit.toLocaleString()}
          </span>
        </div>
      </div>

      <div className="mb-4">
        <button
          type="button"
          onClick={() => setShowGenerate((s) => !s)}
          className="flex w-full items-center justify-between gap-2 rounded-2xl border border-besc-400/40 bg-besc-500/10 px-3.5 py-3 text-sm font-medium text-besc-200 transition-colors hover:bg-besc-500/15"
        >
          <span className="flex items-center gap-1.5">
            <Sparkles className="h-4 w-4 shrink-0" />
            {showGenerate ? "Hide idea generator" : "Don't have a draft? Generate one from an idea"}
          </span>
          <ChevronDown
            className={`h-4 w-4 shrink-0 transition-transform ${showGenerate ? "rotate-180" : ""}`}
          />
        </button>

        {showGenerate && (
          <div className="mt-2.5 rounded-2xl border border-besc-400/30 bg-besc-500/[0.06] p-3.5">
            <textarea
              value={contextIdea}
              onChange={(e) => setContextIdea(e.target.value)}
              placeholder="What do you want to post about? Rough notes are fine."
              rows={3}
              className="w-full resize-none rounded-xl border border-white/10 bg-black/30 p-3 text-[15px] leading-relaxed text-white/85 placeholder:text-white/25 focus:border-besc-400/50 focus:outline-none"
            />
            <button
              type="button"
              onClick={generateFromContext}
              disabled={!contextIdea.trim() || generating}
              className="mt-2.5 flex items-center gap-1.5 rounded-full bg-besc-500 px-3.5 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-besc-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {generating ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Sparkles className="h-3.5 w-3.5" />
              )}
              Generate tweet
            </button>
            {generateError && <p className="mt-2 text-xs text-danger">{generateError}</p>}

            {generateResult?.generateStatus === "disabled" && (
              <p className="mt-2.5 text-[11px] text-white/35">
                AI generation is off. Neither Gemini nor Ollama is configured for this deployment.
              </p>
            )}
            {generateResult?.generateStatus === "error" && (
              <p className="mt-2.5 text-xs text-danger">Generation failed to respond. Try again in a moment.</p>
            )}
            {generateResult?.generateStatus === "empty" && (
              <p className="mt-2.5 text-[11px] text-white/35">
                Didn't get a usable post back from that context. Try adding a bit more detail.
              </p>
            )}
            {generateResult?.generateStatus === "found" && generateResult.candidates && (
              <div className="mt-2.5 space-y-2.5">
                <p className="text-[11px] leading-relaxed text-warn/80">
                  Fully AI-written from your notes, not a rewrite of your own words — there's no
                  ground truth here to check it against. Read every option before using it: verify
                  every name, number, and claim yourself.
                </p>
                {generateResult.candidates.map((candidate, i) => (
                  <div key={i} className="rounded-xl border border-white/10 bg-black/25 p-3">
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="font-mono text-xs font-semibold tabular-nums text-besc-300">
                        {candidate.score.toFixed(1)}
                      </span>
                      <button
                        type="button"
                        onClick={() => useGeneratedCandidate(candidate.text)}
                        className="flex items-center gap-1 rounded-full bg-besc-500 px-3 py-1 text-[11px] font-semibold text-black transition-colors hover:bg-besc-400"
                      >
                        <Check className="h-3 w-3" /> Use this draft
                      </button>
                    </div>
                    <p className="text-[13px] leading-relaxed text-white/70">{candidate.text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
        <Download className="h-4 w-4 shrink-0 text-white/30" />
        <input
          value={importUrl}
          onChange={(e) => setImportUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && importFromUrl()}
          placeholder="Or paste a live x.com/…/status/… link to score a real post"
          className="w-full min-w-0 bg-transparent text-sm text-white/80 placeholder:text-white/25 focus:outline-none"
        />
        <button
          type="button"
          onClick={importFromUrl}
          disabled={!importUrl.trim() || importing}
          className="flex shrink-0 items-center gap-1.5 rounded-full border border-besc-400/40 bg-besc-500/10 px-3 py-1 text-xs font-medium text-besc-200 transition-colors hover:bg-besc-500/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {importing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          Import
        </button>
      </div>
      {importError && <p className="mb-4 -mt-2 text-xs text-danger">{importError}</p>}

      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          onImport?.(null);
          setOptimizeResult(null);
          setAiAutoApplied(false);
          setPreAiText(null);
          setPostedHoursAgo(0);
        }}
        placeholder="What's happening?"
        rows={5}
        className="w-full resize-none rounded-2xl border border-white/10 bg-black/30 p-4 text-[15px] leading-relaxed text-white/90 placeholder:text-white/25 focus:border-besc-400/50 focus:outline-none focus:ring-2 focus:ring-besc-400/20"
      />

      <div className="mt-4 flex flex-wrap gap-2">
        {MEDIA_OPTIONS.map((opt) => {
          const Icon = opt.icon;
          const active = mediaType === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => setMediaType(opt.id)}
              className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                active
                  ? "border-besc-400/50 bg-besc-500/15 text-besc-200"
                  : "border-white/10 bg-white/[0.03] text-white/50 hover:text-white/80"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {opt.label}
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => setShowContext((v) => !v)}
        className="mt-4 flex w-full items-center justify-between gap-2 rounded-2xl border border-white/10 bg-white/[0.02] px-3.5 py-2.5 text-xs font-medium text-white/55 transition-colors hover:text-white/80"
      >
        <span className="flex items-center gap-2">
          <SlidersHorizontal className="h-3.5 w-3.5 shrink-0" />
          Post context &amp; your account
          {contextSummary && <span className="text-white/30">· {contextSummary}</span>}
        </span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 transition-transform ${showContext ? "rotate-180" : ""}`}
        />
      </button>

      {showContext && (
        <div>
      <div className="mt-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
        <Link2 className="h-4 w-4 shrink-0 text-white/30" />
        <input
          value={link}
          onChange={(e) => setLink(e.target.value)}
          placeholder="Optional link you're posting (paste the exact URL)"
          className="w-full min-w-0 bg-transparent text-sm text-white/80 placeholder:text-white/25 focus:outline-none"
        />
      </div>

      <div className="mt-4 grid gap-2.5 sm:grid-cols-2">
        <label className="flex cursor-pointer items-center justify-between gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
          <span className="flex min-w-0 items-center gap-2 text-sm text-white/70">
            <Reply className="h-4 w-4 shrink-0 text-white/35" />
            <span className="truncate">This is a reply</span>
          </span>
          <input
            type="checkbox"
            checked={isReply}
            onChange={(e) => setIsReply(e.target.checked)}
            className="h-4 w-4 shrink-0 accent-besc-500"
          />
        </label>

        <label
          className={`flex items-center justify-between gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5 ${
            isReply ? "cursor-not-allowed opacity-40" : "cursor-pointer"
          }`}
          title="Whether a meaningful share of your followers follow you back. Only matters for original posts — replies never get this bonus."
        >
          <span className="flex min-w-0 items-center gap-2 text-sm text-white/70">
            <Users className="h-4 w-4 shrink-0 text-white/35" />
            <span className="truncate">Mostly mutual-follow audience</span>
          </span>
          <input
            type="checkbox"
            checked={hasMutualFollowAudience}
            disabled={isReply}
            onChange={(e) => setHasMutualFollowAudience(e.target.checked)}
            className="h-4 w-4 shrink-0 accent-besc-500"
          />
        </label>

        <label className="flex cursor-pointer items-center justify-between gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
          <span className="flex min-w-0 items-center gap-2 text-sm text-white/70">
            <Ban className="h-4 w-4 shrink-0 text-white/35" />
            <span className="truncate">Contains sensitive media</span>
          </span>
          <input
            type="checkbox"
            checked={nsfw}
            onChange={(e) => setNsfw(e.target.checked)}
            className="h-4 w-4 shrink-0 accent-besc-500"
          />
        </label>

        <label className="flex cursor-pointer items-center justify-between gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
          <span className="flex min-w-0 items-center gap-2 text-sm text-white/70">
            <BadgeCheck className="h-4 w-4 shrink-0 text-white/35" />
            <span className="truncate">Verified / Premium checkmark</span>
          </span>
          <input
            type="checkbox"
            checked={isVerified}
            onChange={(e) => setIsVerified(e.target.checked)}
            className="h-4 w-4 shrink-0 accent-besc-500"
          />
        </label>
      </div>

      <div className="mt-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
        <AtSign className="h-4 w-4 shrink-0 text-white/30" />
        <input
          value={handleInput}
          onChange={(e) => onHandleChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && lookupHandle()}
          placeholder="Your @handle (auto-fills followers + verified below)"
          className="w-full min-w-0 bg-transparent text-sm text-white/80 placeholder:text-white/25 focus:outline-none"
        />
        <button
          type="button"
          onClick={lookupHandle}
          disabled={!handleInput.trim() || lookingUpHandle}
          className="flex shrink-0 items-center gap-1.5 rounded-full border border-besc-400/40 bg-besc-500/10 px-3 py-1 text-xs font-medium text-besc-200 transition-colors hover:bg-besc-500/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {lookingUpHandle ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          Fetch
        </button>
      </div>
      {handleLookupError && <p className="mt-1.5 text-xs text-danger">{handleLookupError}</p>}

      <div className="mt-2.5 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
        <Rocket className="h-4 w-4 shrink-0 text-white/30" />
        <input
          type="number"
          min={0}
          inputMode="numeric"
          value={authorFollowers}
          onChange={(e) => setAuthorFollowers(e.target.value)}
          placeholder="Or enter your follower count manually (unlocks the cold-start boost check)"
          className="w-full min-w-0 bg-transparent text-sm text-white/80 placeholder:text-white/25 focus:outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
        />
      </div>

      <div className="mt-4 flex items-center justify-between gap-2 rounded-2xl border border-white/10 bg-black/20 px-3.5 py-2.5">
        <span className="flex min-w-0 items-center gap-2 text-sm text-white/70">
          <Repeat2 className="h-4 w-4 shrink-0 text-white/35" />
          <span className="truncate">Posts already sent in this window</span>
        </span>
        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setRecentPostsCount((c) => Math.max(0, c - 1))}
            className="h-6 w-6 rounded-full border border-white/15 text-white/60 hover:text-white"
          >
            −
          </button>
          <span className="w-4 text-center font-mono text-sm tabular-nums">
            {recentPostsCount}
          </span>
          <button
            type="button"
            onClick={() => setRecentPostsCount((c) => Math.min(10, c + 1))}
            className="h-6 w-6 rounded-full border border-white/15 text-white/60 hover:text-white"
          >
            +
          </button>
        </div>
      </div>
        </div>
      )}

      <button
        type="button"
        onClick={optimize}
        disabled={!text.trim() || optimizing}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-2xl border border-besc-400/50 bg-gradient-to-r from-besc-500/20 to-besc-500/5 py-3 text-sm font-semibold text-besc-200 transition-colors hover:from-besc-500/30 hover:to-besc-500/10 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {optimizing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Wand2 className="h-4 w-4" />
        )}
        Optimize for the algorithm
      </button>

      <button
        type="button"
        onClick={getAiRead}
        disabled={!text.trim() || aiReading}
        className="mt-2 flex w-full items-center justify-center gap-2 rounded-2xl border border-white/12 bg-white/[0.03] py-2.5 text-[13px] font-medium text-white/65 transition-colors hover:bg-white/[0.06] hover:text-white/85 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {aiReading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Activity className="h-3.5 w-3.5" />
        )}
        {aiReading ? "Reading your draft…" : "Second opinion from the AI"}
      </button>

      {trackError && <p className="mt-2 text-xs text-danger">{trackError}</p>}
      {optimizeError && <p className="mt-2 text-xs text-danger">{optimizeError}</p>}
      {aiReadError && <p className="mt-2 text-xs text-danger">{aiReadError}</p>}

      {aiRead && aiReadText === text && (
        <AiReadPanel estimate={aiRead} />
      )}

      {optimizeResult && (
        <div className="mt-4 rounded-2xl border border-besc-400/30 bg-besc-500/[0.06] p-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-white/85">
              {optimizeResult.applied.length === 0
                ? "Already clean, no mechanical fixes found"
                : `${optimizeResult.applied.length} fix${optimizeResult.applied.length > 1 ? "es" : ""} applied`}
            </span>
            <span className="flex items-center gap-1.5 font-mono text-sm tabular-nums">
              <span className="text-white/40">{optimizeResult.before.score.toFixed(1)}</span>
              <span className="text-white/25">→</span>
              <span
                className={
                  optimizeResult.after.score > optimizeResult.before.score
                    ? "font-semibold text-besc-300"
                    : "text-white/60"
                }
              >
                {optimizeResult.after.score.toFixed(1)}
              </span>
            </span>
          </div>

          {optimizeResult.applied.length > 0 && (
            <>
              <ul className="mt-3 space-y-2">
                {optimizeResult.applied.map((step) => (
                  <li key={step.id} className="flex items-start gap-2 text-[12.5px] leading-relaxed">
                    <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-besc-300" />
                    <div>
                      <span className="font-medium text-white/80">{step.label}</span>
                      <span className="ml-1.5 font-mono text-[10.5px] text-white/35">
                        +{(step.scoreAfter - step.scoreBefore).toFixed(1)}
                      </span>
                      <p className="text-white/45">{step.reason}</p>
                    </div>
                  </li>
                ))}
              </ul>

              <div className="mt-3 rounded-xl border border-white/10 bg-black/25 p-3 text-[13px] leading-relaxed text-white/70">
                {optimizeResult.optimizedText}
              </div>

              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={() => applyText(optimizeResult.optimizedText)}
                  className="flex items-center gap-1.5 rounded-full bg-besc-500 px-3.5 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-besc-400"
                >
                  <Check className="h-3.5 w-3.5" /> Use this version
                </button>
                <button
                  type="button"
                  onClick={() => setOptimizeResult(null)}
                  className="flex items-center gap-1.5 rounded-full border border-white/15 px-3.5 py-1.5 text-xs font-medium text-white/50 transition-colors hover:text-white/80"
                >
                  <X className="h-3.5 w-3.5" /> Dismiss
                </button>
              </div>
            </>
          )}

          <div className="mt-4 flex items-center gap-1.5 border-t border-white/10 pt-3 text-[11px] text-white/35">
            <Sparkles className="h-3.5 w-3.5 shrink-0" />
            {optimizeResult.aiStatus === "disabled" && "AI rewrites are off. Neither Gemini nor Ollama is configured for this deployment."}
            {optimizeResult.aiStatus === "error" && "AI rewrite failed to respond. Showing the mechanical fixes only."}
            {optimizeResult.aiStatus === "no_improvement" && "AI checked this draft and didn't find a rewrite that scores higher."}
            {optimizeResult.aiStatus === "found" &&
              (aiAutoApplied
                ? "AI found a rewrite that clearly scored higher and applied it to your draft above."
                : "AI found rewrites that score higher than the mechanical fixes above.")}
          </div>

          {optimizeResult.aiStatus === "found" && aiAutoApplied && optimizeResult.aiCandidates && (
            <div className="mt-2.5 rounded-xl border border-warn/40 bg-warn/10 p-3">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="font-mono text-xs font-semibold tabular-nums text-besc-300">
                  Auto-applied · {optimizeResult.aiCandidates[0].score.toFixed(1)} (+
                  {(optimizeResult.aiCandidates[0].score - optimizeResult.after.score).toFixed(1)})
                </span>
                <button
                  type="button"
                  onClick={undoAiAutoApply}
                  className="flex items-center gap-1 rounded-full border border-white/20 px-3 py-1 text-[11px] font-medium text-white/70 transition-colors hover:text-white"
                >
                  <Undo2 className="h-3 w-3" /> Undo
                </button>
              </div>
              <p className="text-[11px] leading-relaxed text-warn/80">
                This scored {AI_AUTO_APPLY_MARGIN}+ points higher than the mechanical fixes, so it
                replaced your draft automatically. AI can rephrase a claim in a way that changes its
                meaning (e.g. turning a pending status into a done one) even when told not to — a
                higher score only means it fits the algorithm&apos;s signals better, not that it&apos;s
                factually accurate. Read it over before posting.
              </p>
            </div>
          )}

          {optimizeResult.aiStatus === "found" && !aiAutoApplied && optimizeResult.aiCandidates && (
            <div className="mt-2.5 space-y-2.5">
              <p className="text-[11px] leading-relaxed text-warn/80">
                Read this before using it. The AI can rephrase a claim in a way that changes its
                meaning (e.g. turning a pending status into a done one) even when told not to.
                A higher score only means it fits the algorithm's signals better, not that it's
                factually accurate.
              </p>
              {optimizeResult.aiCandidates.map((candidate, i) => (
                <div key={i} className="rounded-xl border border-white/10 bg-black/25 p-3">
                  <div className="mb-1.5 flex items-center justify-between">
                    <span className="font-mono text-xs font-semibold tabular-nums text-besc-300">
                      {candidate.score.toFixed(1)}
                    </span>
                    <button
                      type="button"
                      onClick={() => applyText(candidate.text)}
                      className="flex items-center gap-1 rounded-full bg-besc-500 px-3 py-1 text-[11px] font-semibold text-black transition-colors hover:bg-besc-400"
                    >
                      <Check className="h-3 w-3" /> Use this version
                    </button>
                  </div>
                  <p className="text-[13px] leading-relaxed text-white/70">{candidate.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
