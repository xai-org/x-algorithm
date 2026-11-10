import type { Metadata, Viewport } from "next";
import { Sora, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const display = Sora({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "600", "700", "800"],
});

const sans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["400", "500", "600", "700"],
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "700"],
});

// This app's own domain, which is what relative asset paths below resolve
// against — notably the OG/Twitter card image, which lives in this app's
// public/ directory and is therefore only served from here. Pointing this at
// another domain silently breaks every link preview. Overridable so Railway
// preview deploys can advertise their own URL without a code change.
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://xalgo.beschyperchain.com";

export const metadata: Metadata = {
  title: "BESC Engagement Checker: Score your post before you post it",
  description:
    "Paste your draft post and get an instant, transparent score built on the real ranking weights and visibility-filtering rules from X's open-sourced For You algorithm.",
  metadataBase: new URL(SITE_URL),
  alternates: { canonical: "/" },
  openGraph: {
    title: "BESC Engagement Checker",
    description:
      "Score your X post before you post it. Grounded in the real, open-sourced ranking algorithm.",
    type: "website",
    url: SITE_URL,
    siteName: "BESC Engagement Checker",
    images: ["/besc-banner.png"],
  },
  twitter: {
    card: "summary_large_image",
    site: "@BESCLLC",
    creator: "@safudev0702",
    images: ["/besc-banner.png"],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#050403",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body className="bg-void text-white font-sans antialiased selection:bg-besc-400/30 selection:text-white">
        {children}
      </body>
    </html>
  );
}
