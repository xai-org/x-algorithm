"use client";

import { Brain, Gauge } from "lucide-react";

/**
 * Makes the single most important property of a score visible: whether it's a
 * generic estimate or fitted to this author's own measured results. Without
 * this the calibration work is invisible — the number just silently changes
 * meaning, which is exactly the kind of unexplained authority this tool is
 * supposed to avoid.
 */
export default function CalibrationBadge({
  strength,
  hasHandle,
}: {
  strength: number;
  hasHandle: boolean;
}) {
  const pct = Math.round(strength * 100);

  if (strength <= 0) {
    return (
      <div className="glass-inset flex items-start gap-2.5 px-4 py-3">
        <Gauge className="mt-0.5 h-4 w-4 shrink-0 text-white/35" />
        <div className="min-w-0">
          <p className="text-[12.5px] font-medium text-white/70">Generic estimate</p>
          <p className="mt-0.5 text-[11.5px] leading-relaxed text-white/40">
            {hasHandle ? (
              <>
                Built on the real weights, but the action probabilities are still
                averages. <span className="text-besc-300">Learn from my history</span> below
                fits them to your own results.
              </>
            ) : (
              <>
                Add your @handle below, then let it read your published posts — the score
                stops being an average and starts being about your account.
              </>
            )}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2.5 rounded-2xl border border-besc-400/30 bg-besc-500/[0.08] px-4 py-3">
      <Brain className="mt-0.5 h-4 w-4 shrink-0 text-besc-300" />
      <div className="min-w-0 flex-1">
        <p className="text-[12.5px] font-medium text-besc-100">
          Calibrated to your account
        </p>
        <p className="mt-0.5 text-[11.5px] leading-relaxed text-white/45">
          {pct}% of this score comes from a model fitted to your real published results,
          {" "}{100 - pct}% from the general heuristics. Post more and the fitted share grows.
        </p>
        <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.08]">
          <div
            className="h-full rounded-full bg-besc-400 transition-[width] duration-500"
            style={{ width: `${Math.max(4, pct)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
