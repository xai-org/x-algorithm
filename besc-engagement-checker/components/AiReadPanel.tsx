"use client";

import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { AiEstimateResponse } from "@/lib/types";

const ACTION_LABELS: Record<string, string> = {
  favorite: "Likes",
  reply: "Replies",
  retweet: "Reposts",
  quote: "Quote posts",
  shareViaCopyLink: "Shares / copy link",
  followAuthor: "New follows",
  notInterested: "“Not interested”",
  muteAuthor: "Mutes",
  report: "Reports",
};

// Actions where a bigger number is worse. Rendering them the same green as a
// lift would be actively misleading — "4× more likely to be reported" is the
// worst thing this panel can tell you.
const NEGATIVE_ACTIONS = new Set(["notInterested", "muteAuthor", "report"]);

// Below this the model is effectively saying "typical", and listing a wall of
// 1.05s would bury the two lines that matter.
const NOTABLE = 0.15;

function formatWhen(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  return `${days} days ago`;
}

export default function AiReadPanel({ estimate }: { estimate: AiEstimateResponse }) {
  const delta = estimate.result.score - estimate.baselineScore;

  const notable = Object.entries(estimate.multipliers)
    .filter(([, value]) => Math.abs(value - 1) >= NOTABLE)
    .sort((a, b) => Math.abs(b[1] - 1) - Math.abs(a[1] - 1))
    .slice(0, 6);

  return (
    <div className="mt-4 rounded-2xl border border-white/12 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-white/85">AI read of this draft</span>
        <span className="flex items-center gap-1.5 font-mono text-sm tabular-nums">
          <span className="text-white/40">{estimate.baselineScore.toFixed(1)}</span>
          <span className="text-white/25">→</span>
          <span className={delta >= 0 ? "text-besc-300" : "text-danger"}>
            {estimate.result.score.toFixed(1)}
          </span>
        </span>
      </div>

      <p className="mt-1.5 text-[12.5px] leading-relaxed text-white/45">
        How a model that has read X&apos;s ranking rules expects this post to do against a
        typical post of yours — fed through the same real weights as everything else.
      </p>

      {notable.length === 0 ? (
        <p className="mt-3 flex items-center gap-2 text-[12.5px] text-white/50">
          <Minus className="h-3.5 w-3.5 text-white/30" />
          Reads as an ordinary post on every action — nothing standing out either way.
        </p>
      ) : (
        <ul className="mt-3 space-y-1.5">
          {notable.map(([action, value]) => {
            const up = value > 1;
            const bad = NEGATIVE_ACTIONS.has(action) ? up : !up;
            return (
              <li key={action} className="flex items-center justify-between gap-3 text-[13px]">
                <span className="flex items-center gap-2 text-white/60">
                  {up ? (
                    <ArrowUp className={`h-3.5 w-3.5 ${bad ? "text-danger" : "text-besc-300"}`} />
                  ) : (
                    <ArrowDown className={`h-3.5 w-3.5 ${bad ? "text-danger" : "text-besc-300"}`} />
                  )}
                  {ACTION_LABELS[action] ?? action}
                </span>
                <span className="font-mono tabular-nums text-white/70">
                  {value.toFixed(2)}×
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {/* The honesty line. An LLM will produce confident numbers whether or not
          they predict anything, so the panel says out loud whether this one has
          been tested against the user's own published results. */}
      <p className="mt-3 border-t border-white/8 pt-3 text-[11.5px] leading-relaxed text-white/35">
        {estimate.evaluation === null ? (
          <>
            Untested on your account. Run the estimator test in your track record to find
            out whether this read actually predicts your results — until then it&apos;s an
            informed opinion, not a measurement.
          </>
        ) : estimate.evaluation.llmWins ? (
          <>
            Tested {formatWhen(estimate.evaluation.evaluatedAt)} on {estimate.evaluation.n} of
            your published posts, where it ranked outcomes{" "}
            <span className="text-besc-300/70">better</span> than the heuristic score.
          </>
        ) : (
          <>
            Tested {formatWhen(estimate.evaluation.evaluatedAt)} on {estimate.evaluation.n} of
            your published posts, where it did{" "}
            <span className="text-white/50">not</span> beat the heuristic score. Treat this
            read as a sanity check, not a verdict.
          </>
        )}
      </p>
    </div>
  );
}
