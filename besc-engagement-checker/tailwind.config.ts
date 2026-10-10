import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#050403",
        surface: "#0b0906",
        panel: "#131009",
        line: "#241d10",
        besc: {
          50: "#fbf3e2",
          100: "#f6e6bd",
          200: "#f1df92",
          300: "#e7bb7c",
          400: "#d9ae68",
          500: "#c2924f",
          600: "#ad8345",
          700: "#a06e36",
          800: "#7a5527",
          900: "#56391a",
        },
        volt: "#f6e2a1",
        signal: "#b3823f",
        danger: "#ff5c7c",
        warn: "#ffb547",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        sans: ["var(--font-sans)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px -10px rgba(194, 146, 79, 0.55)",
        "glow-lg": "0 0 80px -20px rgba(241, 223, 146, 0.4)",
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(to bottom, transparent, rgba(5,4,3,1)), linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)",
        "gold-sheen":
          "linear-gradient(115deg, #a06e36 0%, #f1df92 28%, #fbf3e2 42%, #d9ae68 58%, #ad8345 78%, #f1df92 100%)",
      },
      keyframes: {
        aurora: {
          "0%, 100%": { transform: "translate(0%, 0%) rotate(0deg)" },
          "50%": { transform: "translate(4%, -6%) rotate(6deg)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.9)", opacity: "0.6" },
          "80%, 100%": { transform: "scale(1.6)", opacity: "0" },
        },
        rise: {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        aurora: "aurora 22s ease-in-out infinite",
        "pulse-ring": "pulse-ring 2.4s cubic-bezier(0.2,0.6,0.4,1) infinite",
        rise: "rise 0.5s cubic-bezier(0.16,1,0.3,1) both",
        shimmer: "shimmer 2.5s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
