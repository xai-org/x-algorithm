"use client";

import { motion } from "framer-motion";
import { Eye, Heart, Repeat2, MessageCircle, Quote, Bookmark, BadgeCheck } from "lucide-react";
import type { TweetImportResult } from "@/lib/types";

export default function RealMetricsPanel({ data }: { data: TweetImportResult }) {
  const stats = [
    { label: "Views", value: data.realMetrics.views, icon: Eye },
    { label: "Likes", value: data.realMetrics.likes, icon: Heart },
    { label: "Reposts", value: data.realMetrics.retweets, icon: Repeat2 },
    { label: "Replies", value: data.realMetrics.replies, icon: MessageCircle },
    { label: "Quotes", value: data.realMetrics.quotes, icon: Quote },
    { label: "Bookmarks", value: data.realMetrics.bookmarks, icon: Bookmark },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel p-5 sm:p-6"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold">Live post performance</h3>
        {data.authorHandle && (
          <span className="flex items-center gap-1 text-xs text-white/40">
            @{data.authorHandle}
            {data.authorVerified && <BadgeCheck className="h-3.5 w-3.5 text-besc-300" />}
          </span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2.5 sm:grid-cols-6">
        {stats.map((s) => (
          <div key={s.label} className="glass-inset flex flex-col items-center gap-1 px-3 py-3 text-center">
            <s.icon className="h-4 w-4 text-white/35" />
            <span className="font-mono text-base font-semibold tabular-nums text-white/85">
              {s.value.toLocaleString()}
            </span>
            <span className="text-[10.5px] uppercase tracking-wide text-white/35">{s.label}</span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[12.5px] leading-relaxed text-white/40">
        Pulled live via Vee3. The BESC Score below is what our heuristic predicts for this text
        and media. Compare it against what the post actually did.
      </p>
    </motion.div>
  );
}
