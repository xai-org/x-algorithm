import { ActionResponder, jsx } from "@jetfuel";
import { action, ResponseCode, t } from "@serving";
import { TRACE_SETTINGS_SWITCH } from "../../../logged_out/cards/under_the_hood/traceView";

const T = t.Object({
	enabled: t.Union([t.Literal("true"), t.Literal("false")]),
});

export default action("logged_in.under_the_hood.actions.set_home_enabled", T, async (ctx) => {
	const fs = await ctx.switches();
	if (!fs.isTrue(TRACE_SETTINGS_SWITCH)) return ResponseCode.notFound;

	try {
		const result = await ctx.bird.$gql.SetUnderTheHoodHomeEnabled({ enabled: ctx.enabled === "true" });
		if (result.set_under_the_hood_home_enabled) return <ActionResponder />;
	} catch {
		return ResponseCode.error;
	}
	return ResponseCode.error;
});
