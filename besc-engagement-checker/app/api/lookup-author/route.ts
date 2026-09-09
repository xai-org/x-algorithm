import { NextRequest, NextResponse } from "next/server";
import { lookupAuthor } from "@/lib/twitter-import";
import { Vee3Error } from "@/lib/vee3";

export async function POST(req: NextRequest) {
  let body: { handle?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const handle = typeof body.handle === "string" ? body.handle.trim().slice(0, 100) : "";
  if (!handle) {
    return NextResponse.json({ error: "handle is required" }, { status: 400 });
  }

  try {
    const result = await lookupAuthor(handle);
    return NextResponse.json(result);
  } catch (err) {
    const status = err instanceof Vee3Error ? 502 : 400;
    const message = err instanceof Error ? err.message : "Failed to look up account";
    return NextResponse.json({ error: message }, { status });
  }
}
