import { raw, ResponseCode, t } from "@serving";
import { loadReport } from "./_lib/load";

const T = t.Object({ month: t.Optional(t.String()) });

export default raw("logged_in.under_the_hood.download", T, async (ctx) => {
	const fs = await ctx.switches();
	if (!fs.isTrue("rweb_under_the_hood_report_enabled")) return ResponseCode.notFound;
	const ct0 = ctx.cookies.ct0;
	const state = await loadReport(ctx, undefined, ct0 ? { headers: { "x-csrf-token": ct0 } } : undefined, ctx.month);
	if (state.status !== "ok" || !state.rawJson) return ResponseCode.notFound;
	const stamp =
		state.info.reportPeriod
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-+|-+$/g, "") || "report";
	return new Response(state.rawJson, {
		headers: {
			"Content-Type": "application/json; charset=utf-8",
			"Content-Disposition": `attachment; filename="x-under-the-hood-${stamp}.json"`,
			"Cache-Control": "private, no-store",
			"X-Content-Type-Options": "nosniff",
		},
	});
});
