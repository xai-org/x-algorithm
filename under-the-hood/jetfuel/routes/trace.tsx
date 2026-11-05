import { jsx, jsx_f, Icon, List2, Nav, Page, Text, View } from "@jetfuel";
import { isProd, page, ResponseCode, t } from "@serving";
import { backgroundPrimary, textPrimary, textSecondary } from "../../../ui/themes";
import previewTrace from "../../logged_out/cards/under_the_hood/previewTrace.json";
import { decodeTrace, parseTrace, previewAllowed } from "../../logged_out/cards/under_the_hood/traceBody";
import { latency, stageList, traceTitle } from "../../logged_out/cards/under_the_hood/traceView";

const T = t.Object({
	t: t.Optional(t.String()),
	preview: t.Optional(t.String()),
});

export default page("logged_in.under_the_hood.trace", T, async (ctx) => {
	const fs = await ctx.switches();
	if (!fs.isTrue("jetfuel_under_the_hood_trace_page_enabled")) return ResponseCode.notFound;
	if (ctx.preview && !previewAllowed(isProd, ctx.isDogfood)) return ResponseCode.notFound;

	const trace = ctx.preview ? parseTrace(previewTrace) : ctx.t ? decodeTrace(ctx.t) : null;
	if (!trace) return ResponseCode.notFound;

	const platform = {
		web: ctx.clientType === "web",
		tray: { clientType: ctx.clientType, clientVersion: ctx.clientVersion, isDogfood: ctx.isDogfood },
	};
	return {
		headers: { "cache-control": "no-store" },
		body: (
			<Page
				mod={`${backgroundPrimary} flex-col`}
				headerBehavior="fixed"
				header={
					<View mod={`flex-row items-center px-2 pt-2 pb-2 safe-area-t ${backgroundPrimary}`}>
						<Nav to=":back" mod="w-10 h-10 items-center justify-center rounded-full">
							<Icon key="arrow_left" size={20} mod={textPrimary} />
						</Nav>
						<Text t={traceTitle()} mod={`${textPrimary} text-lg font-bold ml-2 flex-1 w-min-0`} numberOfLines={1} truncate="end" />
						<Text t={latency(trace.latencyUs)} mod={`text-sm leading-5 shrink-0 px-2 ${textSecondary}`} numberOfLines={1} />
					</View>
				}
			>
				<List2 mod="flex-col w-full">
					<View mod="flex-col w-full px-3 pt-1 pb-8 safe-area-b">{stageList(trace, "uth", 0, platform)}</View>
				</List2>
			</Page>
		),
	};
});
