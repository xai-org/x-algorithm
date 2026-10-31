import { Globe, X as XIcon, Code2 } from "lucide-react";

export const SOCIAL_LINKS = [
  {
    id: "x",
    label: "@BESCLLC",
    href: "https://x.com/BESCLLC",
    icon: XIcon,
  },
  {
    id: "dev",
    label: "@safudev0702",
    href: "https://x.com/safudev0702",
    icon: Code2,
  },
  {
    id: "web",
    label: "bescfinancial.com",
    href: "https://bescfinancial.com",
    icon: Globe,
  },
];

export default function SocialLinks({
  variant = "compact",
}: {
  variant?: "compact" | "labeled";
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {SOCIAL_LINKS.map((l) => {
        const Icon = l.icon;
        return (
          <a
            key={l.id}
            href={l.href}
            target="_blank"
            rel="noreferrer"
            title={l.label}
            className={`group flex items-center gap-1.5 rounded-full border border-besc-500/20 bg-white/[0.02] text-white/50 transition-colors hover:border-besc-400/40 hover:text-besc-200 ${
              variant === "compact" ? "h-8 w-8 justify-center" : "px-3 py-1.5"
            }`}
          >
            <Icon className="h-3.5 w-3.5 shrink-0" />
            {variant === "labeled" && (
              <span className="text-xs font-medium">{l.label}</span>
            )}
          </a>
        );
      })}
    </div>
  );
}
