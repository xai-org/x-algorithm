"use client";

import { motion } from "framer-motion";
import { ShieldAlert, ShieldCheck, Info, TriangleAlert } from "lucide-react";
import type { RiskFlag } from "@/lib/types";

const SEVERITY: Record<
  RiskFlag["severity"],
  { icon: typeof ShieldAlert; ring: string; bg: string; text: string; label: string }
> = {
  critical: {
    icon: ShieldAlert,
    ring: "border-danger/30",
    bg: "bg-danger/[0.06]",
    text: "text-danger",
    label: "Flag risk",
  },
  warning: {
    icon: TriangleAlert,
    ring: "border-warn/30",
    bg: "bg-warn/[0.06]",
    text: "text-warn",
    label: "Watch out",
  },
  info: {
    icon: Info,
    ring: "border-signal/30",
    bg: "bg-signal/[0.06]",
    text: "text-signal",
    label: "How reach works",
  },
};

export default function RiskPanel({ risks }: { risks: RiskFlag[] }) {
  const active = risks.filter((r) => r.triggered && r.severity !== "info");
  const clean = risks.filter((r) => !r.triggered && r.severity !== "info");
  const info = risks.filter((r) => r.severity === "info" && r.triggered);

  return (
    <div className="glass-panel p-5 sm:p-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold">Visibility-filtering risk</h3>
        {active.length === 0 ? (
          <span className="flex items-center gap-1.5 rounded-full bg-besc-500/10 px-2.5 py-1 text-xs font-medium text-besc-300">
            <ShieldCheck className="h-3.5 w-3.5" /> No flags triggered
          </span>
        ) : (
          <span className="rounded-full bg-danger/10 px-2.5 py-1 text-xs font-medium text-danger">
            {active.length} flag{active.length > 1 ? "s" : ""} triggered
          </span>
        )}
      </div>

      <div className="space-y-3">
        {[...active, ...info].map((r, i) => {
          const s = SEVERITY[r.severity];
          const Icon = s.icon;
          return (
            <motion.div
              key={r.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className={`rounded-2xl border ${s.ring} ${s.bg} p-3.5`}
            >
              <div className="flex items-start gap-2.5">
                <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${s.text}`} />
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-sm font-semibold text-white/90">{r.title}</p>
                    <span className={`text-[10px] font-medium uppercase tracking-wide ${s.text}`}>
                      {s.label}
                    </span>
                  </div>
                  <p className="mt-1 text-[13px] leading-relaxed text-white/55">{r.detail}</p>
                  <p className="mt-1.5 font-mono text-[10.5px] text-white/30">{r.source}</p>
                </div>
              </div>
            </motion.div>
          );
        })}

        {clean.length > 0 && (
          <details className="group rounded-2xl border border-white/[0.06] bg-white/[0.02] p-3.5">
            <summary className="cursor-pointer list-none text-[13px] font-medium text-white/40 hover:text-white/60">
              {clean.length} other check{clean.length > 1 ? "s" : ""} passed clean ▾
            </summary>
            <div className="mt-2.5 space-y-2">
              {clean.map((r) => (
                <div key={r.id} className="flex items-start gap-2 text-[12.5px] text-white/45">
                  <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-besc-400" />
                  <span>{r.title}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
