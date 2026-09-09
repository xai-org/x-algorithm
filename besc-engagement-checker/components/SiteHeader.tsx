"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, Github, Menu, X } from "lucide-react";
import { SOCIAL_LINKS } from "./SocialLinks";

const REPO_URL = "https://github.com/BESCLLC/x-algo-BESC-twitter-engagement-";

export default function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Route changes should never leave the menu hanging open behind the new page.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!open) return;

    function onPointerDown(e: MouseEvent | TouchEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("touchstart", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("touchstart", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const onDocs = pathname === "/docs";

  return (
    <header className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-5 py-6 sm:px-8">
      <Link href="/" className="flex min-w-0 items-center gap-2.5">
        <div className="relative h-9 w-9 shrink-0 drop-shadow-[0_0_14px_rgba(194,146,79,0.45)]">
          <Image src="/besc-logo.png" alt="BESC" fill sizes="36px" className="object-contain" priority />
        </div>
        <span className="truncate font-display text-[15px] font-bold tracking-tight">
          <span className="text-gold">BESC</span>{" "}
          <span className="font-medium text-white/40">Engagement Checker</span>
        </span>
      </Link>

      {/* Room for the full nav — show it rather than hiding links behind a menu. */}
      <div className="hidden items-center gap-2 md:flex">
        <Link
          href="/docs"
          className={`flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
            onDocs
              ? "border-besc-400/40 bg-besc-500/10 text-besc-200"
              : "border-white/10 bg-white/[0.03] text-white/50 hover:text-white/80"
          }`}
        >
          <BookOpen className="h-3.5 w-3.5" />
          Docs
        </Link>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-xs font-medium text-white/50 transition-colors hover:text-white/80"
        >
          <Github className="h-3.5 w-3.5" />
          View the algorithm
        </a>
        <div className="flex items-center gap-2">
          {SOCIAL_LINKS.map((l) => {
            const Icon = l.icon;
            return (
              <a
                key={l.id}
                href={l.href}
                target="_blank"
                rel="noreferrer"
                title={l.label}
                className="flex h-8 w-8 items-center justify-center rounded-full border border-besc-500/20 bg-white/[0.02] text-white/50 transition-colors hover:border-besc-400/40 hover:text-besc-200"
              >
                <Icon className="h-3.5 w-3.5" />
              </a>
            );
          })}
        </div>
      </div>

      {/* Below md the same links would crowd the logo off the row, so they collapse. */}
      <div ref={menuRef} className="relative shrink-0 md:hidden">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          aria-haspopup="menu"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-white/60 transition-colors hover:text-white/90"
        >
          {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              role="menu"
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
              className="absolute right-0 z-50 mt-2 w-60 origin-top-right overflow-hidden rounded-2xl border border-white/10 bg-panel/95 p-1.5 shadow-2xl shadow-black/60 backdrop-blur-xl"
            >
              <Link
                href="/docs"
                role="menuitem"
                onClick={() => setOpen(false)}
                className={`flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[13.5px] font-medium transition-colors ${
                  onDocs ? "bg-besc-500/12 text-besc-200" : "text-white/70 hover:bg-white/[0.05]"
                }`}
              >
                <BookOpen className="h-4 w-4 shrink-0" />
                Docs
              </Link>
              <a
                href={REPO_URL}
                target="_blank"
                rel="noreferrer"
                role="menuitem"
                onClick={() => setOpen(false)}
                className="flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[13.5px] font-medium text-white/70 transition-colors hover:bg-white/[0.05]"
              >
                <Github className="h-4 w-4 shrink-0" />
                View the algorithm
              </a>

              <div className="my-1.5 border-t border-white/[0.07]" />

              {SOCIAL_LINKS.map((l) => {
                const Icon = l.icon;
                return (
                  <a
                    key={l.id}
                    href={l.href}
                    target="_blank"
                    rel="noreferrer"
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[13.5px] font-medium text-white/60 transition-colors hover:bg-white/[0.05] hover:text-besc-200"
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {l.label}
                  </a>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
