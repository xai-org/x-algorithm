"use client";

import { useId } from "react";
import { motion } from "framer-motion";

const GRADE_STYLES: Record<
  string,
  { ring: string; text: string; glow: string; metallic: boolean }
> = {
  Excellent: { ring: "#f1df92", text: "text-besc-200", glow: "shadow-glow-lg", metallic: true },
  Strong: { ring: "#e7bb7c", text: "text-besc-300", glow: "shadow-glow", metallic: true },
  Decent: { ring: "#c2924f", text: "text-besc-400", glow: "", metallic: true },
  Weak: { ring: "#ffb547", text: "text-warn", glow: "", metallic: false },
  "High Risk": { ring: "#ff5c7c", text: "text-danger", glow: "", metallic: false },
};

export default function ScoreGauge({
  score,
  grade,
}: {
  score: number;
  grade: string;
}) {
  const style = GRADE_STYLES[grade] ?? GRADE_STYLES.Decent;
  const gradientId = useId();
  const radius = 88;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, score)) / 100;

  return (
    <div className="relative flex flex-col items-center">
      <div className={`relative h-56 w-56 ${style.glow}`}>
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#a06e36" />
              <stop offset="30%" stopColor="#f1df92" />
              <stop offset="55%" stopColor="#fbf3e2" />
              <stop offset="80%" stopColor="#d9ae68" />
              <stop offset="100%" stopColor="#ad8345" />
            </linearGradient>
          </defs>
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="12"
          />
          <motion.circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={style.metallic ? `url(#${gradientId})` : style.ring}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference * (1 - pct) }}
            transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            key={score}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className={`font-display text-6xl font-bold tabular-nums tracking-tight ${
              style.metallic ? "text-gold" : style.text
            }`}
          >
            {score}
          </motion.span>
          <span className="mt-1 text-xs font-medium uppercase tracking-[0.2em] text-white/40">
            BESC Score
          </span>
        </div>
        <div
          className="absolute -inset-2 -z-10 animate-pulse-ring rounded-full border border-current opacity-0"
          style={{ color: style.ring }}
        />
      </div>
      <div
        className={`mt-4 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-sm font-semibold ${style.text}`}
      >
        {grade}
      </div>
    </div>
  );
}
