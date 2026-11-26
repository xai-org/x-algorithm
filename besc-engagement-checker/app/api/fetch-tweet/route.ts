import { NextRequest, NextResponse } from "next/server";
import { importTweet } from "@/lib/twitter-import";
import { Vee3Error } from "@/lib/vee3";

export async function POST(req: NextRequest) {
  let body: { url?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const url = typeof body.url === "string" ? body.url.trim().slice(0, 500) : "";
  if (!url) {
    return NextResponse.json({ error: "url is required" }, { status: 400 });
  }

  try {
    const imported = await importTweet(url);
    return NextResponse.json(imported);
  } catch (err) {
    const status = err instanceof Vee3Error ? 502 : 400;
    const message = err instanceof Error ? err.message : "Failed to import tweet";
    return NextResponse.json({ error: message }, { status });
  }
}
