"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight } from "lucide-react";
import Background from "@/components/Background";
import SiteHeader from "@/components/SiteHeader";
import Composer from "@/components/Composer";
import ScoreGauge from "@/components/ScoreGauge";
import SignalBreakdown from "@/components/SignalBreakdown";
import RiskPanel from "@/components/RiskPanel";
import TipsPanel from "@/components/TipsPanel";
import SocialLinks from "@/components/SocialLinks";
import RealMetricsPanel from "@/components/RealMetricsPanel";
import TrackRecordPanel from "@/components/TrackRecordPanel";
import CalibrationBadge from "@/components/CalibrationBadge";
import type { ScoreResult, TweetImportResult } from "@/lib/types";

const HANDLE_STORAGE_KEY = "besc:handle";

export default function Home() {
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [imported, setImported] = useState<TweetImportResult | null>(null);
  const [handle, setHandle] = useState("");
  // Bumped whenever a draft is tracked, so the panel below reloads without
  // needing a shared store or a full refresh.
  const [trackVersion, setTrackVersion] = useState(0);

  // Remembered locally so a returning user's track record is just there.
  useEffect(() => {
    const saved = localStorage.getItem(HANDLE_STORAGE_KEY);
    if (saved) setHandle(saved);
  }, []);

  useEffect(() => {
    if (handle.trim()) localStorage.setItem(HANDLE_STORAGE_KEY, handle.trim());
  }, [handle]);

  return (
    <main className="relative min-h-screen">
      <Background />

      <SiteHeader />

      <section className="mx-auto max-w-7xl px-5 pb-6 pt-4 sm:px-8 sm:pb-10">
        <div className="flex flex-col items-start gap-4">
          <span className="animate-rise rounded-full border border-besc-500/25 bg-besc-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-widest text-besc-300">
            Built on X&apos;s actual, open-sourced For You algorithm
          </span>
          <h1
            className="animate-rise text-balance font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-6xl [animation-delay:60ms]"
            style={{ animationFillMode: "backwards" }}
          >
            Write posts the{" "}
            <span className="text-gold">algorithm</span>
            <br className="hidden sm:block" /> actually rewards.
          </h1>
          <p
            className="animate-rise max-w-2xl text-balance text-[15px] leading-relaxed text-white/50 sm:text-base [animation-delay:120ms]"
            style={{ animationFillMode: "backwards" }}
          >
            Draft it, generate it from a rough idea, or paste a live post. Every number
            traces back to a real weight or rule in X&apos;s open-sourced{" "}
            <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[13px] text-white/70">
              RankingScorer
            </code>{" "}
            and{" "}
            <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[13px] text-white/70">
              visibility-filtering
            </code>{" "}
            rules — and once it has read your published results, the score stops being an
            average and starts being about your account.
          </p>

          <div className="animate-rise flex flex-wrap gap-2 [animation-delay:180ms]" style={{ animationFillMode: "backwards" }}>
            {[
              "Live score from the real weights",
              "AI rewrites, gated by the scorer",
              "Write from a rough idea",
              "Learns from your real results",
            ].map((c) => (
              <span
                key={c}
                className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11.5px] font-medium text-white/45"
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-24 sm:px-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:items-start">
        <div className="min-w-0 lg:sticky lg:top-6">
          <Composer
            onResult={setResult}
            onLoading={setLoading}
            onImport={setImported}
            handleInput={handle}
            onHandleChange={setHandle}
            onTracked={() => setTrackVersion((v) => v + 1)}
          />
        </div>

        <div className="min-w-0 space-y-6">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-panel flex min-h-[420px] flex-col items-center justify-center gap-3 p-10 text-center"
              >
                <div
                  className={`h-14 w-14 rounded-2xl border border-white/10 bg-white/[0.03] ${
                    loading ? "animate-pulse" : ""
                  }`}
                />
                <p className="text-sm text-white/40">
                  {loading ? "Analyzing…" : "Start typing to see your live BESC Score."}
                </p>
                {!loading && (
                  <p className="max-w-xs text-[12.5px] leading-relaxed text-white/25">
                    No draft yet? Use the idea generator to write one, or paste a live
                    x.com link to score a post that already exists.
                  </p>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-6"
              >
                {imported && <RealMetricsPanel data={imported} />}

                <div className="glass-panel p-6 sm:p-8">
                  <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-center sm:justify-between">
                    <ScoreGauge score={result.score} grade={result.grade} />
                    <div className="grid w-full grid-cols-2 gap-3 sm:w-auto sm:gap-4">
                      <Stat
                        label="Author diversity ×"
                        value={result.authorDiversityMultiplier.toFixed(2)}
                      />
                      <Stat label="Out-of-network ×" value={result.oonWeightFactor.toFixed(2)} />
                    </div>
                  </div>
                  <div className="mt-5">
                    <CalibrationBadge
                      strength={result.calibrationStrength ?? 0}
                      hasHandle={Boolean(handle.trim())}
                    />
                  </div>
                </div>

                <TrackRecordPanel key={trackVersion} handle={handle} onHandleChange={setHandle} />

                <TipsPanel tips={result.tips} />
                <RiskPanel risks={result.risks} />
                <SignalBreakdown actions={result.actions} />

                <div className="glass-panel flex flex-col items-start gap-2 p-5 text-[13px] leading-relaxed text-white/40 sm:flex-row sm:items-center sm:justify-between">
                  <p className="max-w-xl">
                    This score is a transparent, heuristic estimator built on real
                    production weights. It does not run X&apos;s actual Phoenix model
                    or your real account/viewer history, so treat it as directional
                    coaching, not a guarantee.
                  </p>
                  <a
                    href="#top"
                    className="flex shrink-0 items-center gap-1 font-medium text-besc-300 hover:text-besc-200"
                  >
                    Back to top <ArrowRight className="h-3.5 w-3.5" />
                  </a>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      <footer className="mx-auto flex max-w-7xl flex-col items-center gap-5 px-5 pb-14 pt-6 sm:px-8">
        <div className="relative h-10 w-10 opacity-90">
          <Image src="/besc-logo.png" alt="BESC" fill sizes="40px" className="object-contain" />
        </div>
        <SocialLinks variant="labeled" />
        <p className="max-w-xl text-balance text-center text-xs text-white/25">
          Built by <span className="text-besc-400/80">BESC</span> · weights sourced from{" "}
          <code className="font-mono text-white/35">home-mixer/params/param.rs</code> · not
          affiliated with X Corp.
        </p>
      </footer>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-inset px-4 py-3">
      <p className="text-[11px] uppercase tracking-wide text-white/35">{label}</p>
      <p className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-white/85">{value}</p>
    </div>
  );
}
