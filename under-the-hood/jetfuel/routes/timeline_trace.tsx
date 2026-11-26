import { jsx, actions, atom, Button, If, Text, View, l } from "@jetfuel";
import { isProd, page, ResponseCode, t } from "@serving";
import { toastAction } from "../../../ui/components/xdsToast";
import { textPrimary } from "../../../ui/themes";
import { TRACE_SETTINGS_SWITCH, traceTitle } from "../../logged_out/cards/under_the_hood/traceView";
import { Body, Screen, SwitchIndicator } from "./_lib/ui";

const T = t.Object({});

const copy = {
	description: l(
		"Shows how your For You feed is built. When you refresh For You, you'll see a trace of code that found candidate posts for you, filtered, and ranked them. Each step in the trace links to its open-source code.",
		"Description on the For You timeline trace page; candidate posts are posts being considered for the feed.",
	),
	toggle: l("Show timeline trace", "Label of the switch that turns the For You timeline trace on or off."),
	saveFailed: l(
		"Couldn't update the timeline trace. Try again in a moment.",
		"Toast shown when turning the For You timeline trace on or off fails.",
	),
};

export default page("logged_in.under_the_hood.timeline_trace", T, async (ctx) => {
	const fs = await ctx.switches();
	if (!fs.isTrue(TRACE_SETTINGS_SWITCH)) return ResponseCode.notFound;

	let enabled: boolean;
	try {
		const result = await ctx.bird.$gql.UnderTheHoodHomeEnabled(undefined, { noCache: true });
		if (typeof result.under_the_hood_home_enabled !== "boolean") return ResponseCode.error;
		enabled = result.under_the_hood_home_enabled;
	} catch (e) {
		!isProd && console.error("[under_the_hood] setting fetch failed", e);
		return ResponseCode.error;
	}

	const on = atom.bool({ value: enabled, id: "under_the_hood_trace_on" });
	const save = (value: boolean) =>
		actions.post_request({
			url: "/under_the_hood/actions/set_home_enabled",
			body: { enabled: String(value) },
			optimistic: on.set(value),
			error: actions.multi([on.set(!value), toastAction(copy.saveFailed, "error", 2)]),
		});
	const row = (value: boolean) => (
		<Button haptic="light" pressEffect="light" on:press={save(!value)} scribe:press={{ element: "toggle_timeline_trace" }} mod="w-full">
			<View mod="flex-row w-full items-center gap-4 py-3">
				<Text t={copy.toggle} mod={`${textPrimary} text-base flex-1`} />
				<SwitchIndicator on={value} />
			</View>
		</Button>
	);

	return (
		<Screen title={traceTitle()} page="under_the_hood_timeline_trace">
			<Body t={copy.description} />
			<If cond={on.is(true)} else={row(false)}>
				{row(true)}
			</If>
		</Screen>
	);
});
