import { jsx, Icon, Link, Nav, RichText2, Text, View, l, type JetElement } from "@jetfuel";
import { page, t } from "@serving";
import { FormatText } from "../../logged_out/components/richtext";
import { pillSurface, textPrimary, textSecondary } from "../../../ui/themes";
import { canMock, loadReport, type ReportState } from "./_lib/load";
import type { EligibilityCheck } from "./_lib/report";
import { Body, Divider, OPEN_SOURCE_URL, REPORT_PATH, Screen, downloadUrl, pilotCopy } from "./_lib/ui";

const T = t.Object({
	mock: t.Optional(
		t.Union([t.Literal("none"), t.Literal("pilot"), t.Literal("ineligible"), t.Literal("empty"), t.Literal("labels"), t.Literal("launch")]),
	),
});

const copy = {
	title: l("Under the hood", "Title of the Under the hood page, where a user sees which safety labels affected their posts' reach."),
	description: l(
		"See aggregate stats on labels that can affect post visibility in the X algorithm, for your posts and account last month.",
		"Description on the Under the hood page explaining what the report contains.",
	),
	openSource: l(
		"See how these labels affect visibility in our [link/]",
		"Sentence on the Under the hood page; [link/] is replaced by a link labeled 'open-source code'.",
	),
	openSourceLink: l("open-source code ›", "Link text on the Under the hood page that opens X's open-source algorithm repository."),
	unavailable: l("Report is not available yet.", "Shown on the Under the hood page when no report exists for the user."),
	period: (period: string) =>
		l("Report period: {period}", "Label on the Under the hood page naming the month the report covers, e.g. 'August 2026'.", { period }),
	requirements: l(
		"To view this report during our pilot testing period, you need:",
		"Introduces the eligibility checklist on the Under the hood page.",
	),
	view: l("View report", "Button on the Under the hood page that opens the full report."),
	download: l("Download", "Button on the Under the hood page that downloads the report as a JSON file (web only)."),
};

const Check = ({ check }: { check: EligibilityCheck }) => (
	<View mod="flex-row items-center gap-2">
		<Icon
			key={check.value ? "checkmark_circle_fill" : "close_circle_fill"}
			size={18}
			mod={check.value ? "text-green-500" : "text-red-500"}
		/>
		<Text t={check.key} mod={`${textSecondary} text-base leading-snug flex-1`} />
	</View>
);

const eligible = (state: ReportState, reportPath: string, download: string | undefined): JetElement => {
	if (state.status === "pilot") return <Body t={pilotCopy} />;
	if (state.status === "unavailable") return <Body t={copy.unavailable} />;
	const { info, report } = state;
	return (
		<View mod="flex-col gap-4">
			<Divider />
			{info.reportPeriod ? <Body t={copy.period(info.reportPeriod)} /> : null}
			<Body t={copy.requirements} />
			<View mod="flex-col gap-3">
				{info.eligibilityChecks.map((check) => (
					<Check check={check} />
				))}
			</View>
			{report ? (
				<View mod="flex-row items-center gap-3 mt-2">
					<Nav to={reportPath} scribe:press={{ element: "view_report" }} mod="hover:opacity-80">
						<View mod="h-12 px-6 rounded-full items-center justify-center light:bg-black dark:bg-white dim:bg-white">
							<Text t={copy.view} mod="light:text-white dark:text-black dim:text-black font-bold text-base" />
						</View>
					</Nav>
					{download ? (
						<Link url={download} scribe:press={{ element: "download_report" }} mod="hover:opacity-80">
							<View mod={`h-12 px-6 rounded-full items-center justify-center ${pillSurface}`}>
								<Text t={copy.download} mod={`${textPrimary} font-bold text-base`} />
							</View>
						</Link>
					) : null}
				</View>
			) : null}
		</View>
	);
};

export default page("logged_in.under_the_hood", T, async (ctx) => {
	const fs = await ctx.switches();
	const mock = canMock(ctx) ? ctx.mock : undefined;
	const state: ReportState = fs.isTrue("rweb_under_the_hood_report_enabled") ? await loadReport(ctx, mock) : { status: "pilot" };
	const reportPath = mock ? `${REPORT_PATH}?mock=${mock}` : REPORT_PATH;
	return (
		<Screen title={copy.title} page="under_the_hood">
			<View mod={`${pillSurface} w-12 h-12 rounded-xl items-center justify-center`}>
				<Icon key="code" size={24} mod={textPrimary} />
			</View>
			<Text t={copy.title} mod={`${textPrimary} text-2xl font-bold`} />
			<Body t={copy.description} />
			{state.status === "pilot" ? null : (
				<FormatText
					mod={`${textSecondary} text-base leading-snug`}
					locale={ctx.intl.locale}
					textKey={copy.openSource}
					components={{
						link: <RichText2.Text t={copy.openSourceLink} link={OPEN_SOURCE_URL} mod={`${textSecondary} text-base underline`} />,
					}}
				/>
			)}
			{eligible(state, reportPath, state.status === "ok" && state.rawJson ? downloadUrl(ctx) : undefined)}
		</Screen>
	);
});
