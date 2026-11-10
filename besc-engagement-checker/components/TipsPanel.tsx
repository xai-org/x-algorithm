"use client";

import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import type { Tip } from "@/lib/types";

const IMPACT_STYLE: Record<Tip["impact"], string> = {
  high: "bg-besc-500/15 text-besc-300 border-besc-500/30",
  medium: "bg-signal/15 text-signal border-signal/30",
  low: "bg-white/10 text-white/50 border-white/15",
};

export default function TipsPanel({ tips }: { tips: Tip[] }) {
  if (tips.length === 0) return null;

  return (
    <div className="glass-panel p-5 sm:p-6">
      <div className="mb-4 flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-volt" />
        <h3 className="font-display text-lg font-semibold">
          How to get more views
        </h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {tips.map((t, i) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="glass-inset flex flex-col gap-2 p-4"
          >
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-semibold text-white/90">{t.title}</p>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${IMPACT_STYLE[t.impact]}`}
              >
                {t.impact}
              </span>
            </div>
            <p className="text-[13px] leading-relaxed text-white/50">{t.detail}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
