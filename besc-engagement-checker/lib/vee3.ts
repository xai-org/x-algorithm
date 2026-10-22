import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const VEE3_MCP_URL = "https://mcp.vee3.io/mcp";
const REQUEST_TIMEOUT_MS = 20000;

interface ToolContentBlock {
  type: string;
  text?: string;
}

export class Vee3Error extends Error {}

export async function callVee3Tool<T = unknown>(
  toolName: string,
  args: Record<string, unknown>
): Promise<T> {
  const apiKey = process.env.VEE3_API_KEY;
  if (!apiKey) {
    throw new Vee3Error("VEE3_API_KEY is not configured on the server");
  }

  const transport = new StreamableHTTPClientTransport(new URL(VEE3_MCP_URL), {
    requestInit: { headers: { Authorization: `Bearer ${apiKey}` } },
  });
  const client = new Client({ name: "besc-engagement-checker", version: "1.0.0" });

  // A real AbortController, passed to the SDK's own signal option, so a
  // timeout properly cancels the in-flight request instead of just walking
  // away from it — an external Promise.race that abandons a still-running
  // connect()/callTool() while we call client.close() underneath it is what
  // caused the "Load failed" connection-reset errors after the first attempt
  // at this fix.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    await client.connect(transport, { signal: controller.signal, timeout: REQUEST_TIMEOUT_MS });
    const result = await client.callTool({ name: toolName, arguments: args }, undefined, {
      signal: controller.signal,
      timeout: REQUEST_TIMEOUT_MS,
    });

    const blocks = (result.content ?? []) as ToolContentBlock[];
    const textBlock = blocks.find((b) => b.type === "text" && typeof b.text === "string");

    if (result.isError) {
      const message = blocks.map((b) => b.text).filter(Boolean).join(" ");
      throw new Vee3Error(message || `Vee3 tool "${toolName}" failed`);
    }
    if (!textBlock?.text) {
      throw new Vee3Error(`Vee3 tool "${toolName}" returned no content`);
    }

    try {
      return JSON.parse(textBlock.text) as T;
    } catch {
      return textBlock.text as unknown as T;
    }
  } catch (err) {
    if (controller.signal.aborted) {
      throw new Vee3Error(`Vee3 request timed out after ${(REQUEST_TIMEOUT_MS / 1000).toFixed(0)}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
    await client.close().catch(() => {});
  }
}
