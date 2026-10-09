import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import { jsx, DateTime, Icon, Link, Text, View, l, type JetElement, type Localized } from "@jetfuel";
import { page, t } from "@serving";
import { cardSurface, textPrimary, textSecondary } from "../../../ui/themes";
import { TimelineRichText } from "../events/components/TimelineRichText";
import { canMock, loadReport, type Mock } from "./_lib/load";
import type { AccountLabel, PostLabel, Report } from "./_lib/report";
import { Body, Divider, Screen, downloadUrl, pilotCopy } from "./_lib/ui";

const T = t.Object({
	mock: t.Optional(t.Union([t.Literal("none"), t.Literal("pilot"), t.Literal("ineligible"), t.Literal("empty"), t.Literal("labels")])),
});

const copy = {
	title: l(
		"Under the hood report",
		"Title of the Under the hood report page listing the safety labels applied to a user's posts and account.",
	),
	unavailable: l("Report is not available yet.", "Shown on the Under the hood report page when no report exists for the user."),
	postsLabel: l("Posts in period", "Row label in the Under the hood report summary, beside the number of posts."),
	periodLabel: l("Period", "Row label in the Under the hood report summary, beside the date range."),
	generatedLabel: l("Report generated", "Row label in the Under the hood report summary, beside when the report was produced."),
	count: (count: number) => l("{postCount}", "Post count shown as the value in the Under the hood report summary.", { postCount: count }),
	to: l("to", "Word between the start and end dates on the Under the hood report period row."),
	postLabels: l("Post labels", "Section title on the Under the hood report page for labels applied to individual posts."),
	postLabelsBlurb: l(
		"Labels applied to your posts in this period that could limit post visibility.",
		"Explains the post labels section on the Under the hood report page.",
	),
	noPostLabels: l("None in this period", "Empty state for the post labels section on the Under the hood report page."),
	accountLabels: l("Account labels", "Section title on the Under the hood report page for labels applied to the whole account."),
	accountLabelsBlurb: l(
		"Labels applied to your account in this period that could limit post visibility.",
		"Explains the account labels section on the Under the hood report page.",
	),
	noAccountLabels: l("None in this period", "Empty state for the account labels section on the Under the hood report page."),
	postStat: (posts: number, total: number, pct: string) =>
		l("{posts} of {total} posts · {pct}", "Per-label stat on the Under the hood report page, e.g. '3 of 40 posts · 7.50%'.", {
			posts,
			total,
			pct,
		}),
	dayStat: (days: number, total: number, pct: string) =>
		l("{days} of {total} days · {pct}", "Per-label stat on the Under the hood report page, e.g. '5 of 31 days · 16.12%'.", {
			days,
			total,
			pct,
		}),
	about: l("What it means", "Caption above the description of a label on the Under the hood report page."),
	effect: l("Effect on visibility", "Caption above the visibility effect of a label on the Under the hood report page."),
	reason: l("Reason", "Caption above the high-level reason for a restriction on the Under the hood report page."),
	restriction: l("Restriction", "Caption above the specific distribution restriction on the Under the hood report page."),
	affectedTarget: l("Affected element", "Caption above the affected element or media for a restriction on the Under the hood report page."),
	scope: l("Scope", "Caption above whether the restriction is post-level or account-level."),
	download: l("Download this report (JSON)", "Link on the Under the hood report page that downloads the raw report file (web only)."),
};

const quietPill = "border-xs rounded-md px-2 py-1 light:border-gray-400 dark:border-gray-500 dim:border-gray-400";
const postScopePill = "border-xs rounded-md px-2 py-0.5 light:text-blue-700 dark:text-blue-300 light:bg-blue-100 dark:bg-blue-950/60 light:border-blue-300 dark:border-blue-800 text-xs font-semibold";
const accountScopePill = "border-xs rounded-md px-2 py-0.5 light:text-amber-700 dark:text-amber-300 light:bg-amber-100 dark:bg-amber-950/60 light:border-amber-300 dark:border-amber-800 text-xs font-semibold";
const track = "flex-row w-full h-1.5 rounded-full clipped items-center light:bg-gray-200 dark:bg-zinc-700 dim:bg-gray-700";
const fill = "h-1.5 rounded-full light:bg-gray-600 dark:bg-gray-300 dim:bg-gray-300";
const valueMod = `${textPrimary} text-sm`;

const barWidth = (part: number, total: number): string | undefined => {
	if (!(total > 0) || !(part > 0)) return undefined;
	const twelfths = Math.max(1, Math.min(12, Math.round((part / total) * 12)));
	return twelfths >= 12 ? "w-full" : `w-${twelfths}/12`;
};

const ShareBar = ({ part, total }: { part: number; total: number }) => {
	const width = barWidth(part, total);
	return <View mod={track}>{width ? <View mod={`${fill} ${width}`} /> : null}</View>;
};

const LabelCard = ({
	code,
	stat,
	about,
	effect,
	reason,
	restriction,
	scope,
	affectedTarget,
	affectedMedia,
	part,
	total,
}: {
	code: string;
	stat: Localized;
	about: string;
	effect: string;
	reason?: string;
	restriction?: string;
	scope?: string;
	affectedTarget?: string;
	affectedMedia?: string;
	part: number;
	total: number;
}) => (
	<View mod={`${cardSurface} flex-col gap-2 p-4`}>
		<View mod="flex-row flex-wrap items-center justify-between gap-3">
			<View mod="flex-row items-center gap-2">
				<View mod={quietPill}>
					<Text t={code} mod={`${textPrimary} font-mono text-sm`} />
				</View>
				{scope ? (
					<View mod={scope.toLowerCase().includes("account") ? accountScopePill : postScopePill}>
						<Text t={scope} />
					</View>
				) : null}
			</View>
			<Text t={stat} mod={`${textPrimary} text-sm font-bold`} />
		</View>
		<ShareBar part={part} total={total} />
		{reason ? (
			<View mod="flex-col gap-0.5">
				<Text t={copy.reason} mod={`${textSecondary} text-sm font-medium`} />
				<Text t={reason} mod={`${textPrimary} text-sm leading-snug`} />
			</View>
		) : null}
		{affectedMedia || affectedTarget ? (
			<View mod="flex-col gap-0.5">
				<Text t={copy.affectedTarget} mod={`${textSecondary} text-sm font-medium`} />
				<Text t={affectedMedia ?? affectedTarget ?? ""} mod={`${textPrimary} text-sm leading-snug`} />
			</View>
		) : null}
		{restriction ? (
			<View mod="flex-col gap-0.5">
				<Text t={copy.restriction} mod={`${textSecondary} text-sm font-medium`} />
				<Text t={restriction} mod={`${textPrimary} text-sm leading-snug`} />
			</View>
		) : null}
		<View mod="flex-col gap-0.5">
			<Text t={copy.about} mod={`${textSecondary} text-sm`} />
			<Text t={about} mod={`${textPrimary} text-sm leading-snug`} />
		</View>
		<View mod="flex-col gap-0.5">
			<Text t={copy.effect} mod={`${textSecondary} text-sm`} />
			<Text t={effect} mod={`${textPrimary} text-sm leading-snug`} />
		</View>
	</View>
);

const Empty = ({ t }: { t: Localized }) => (
	<View mod="flex-row items-center gap-2">
		<Icon key="checkmark_circle_fill" size={16} mod="text-green-500" />
		<Text t={t} mod={`${textPrimary} text-sm flex-1`} />
	</View>
);

const Section = ({ title, blurb, empty, cards }: { title: Localized; blurb: Localized; empty: Localized; cards: JetElement[] }) => (
	<View mod="flex-col gap-3">
		<View mod="flex-col gap-1">
			<Text t={title} mod={`${textPrimary} text-base font-bold`} />
			<Text t={blurb} mod={`${textSecondary} text-sm leading-snug`} />
		</View>
		{cards.length > 0 ? <View mod="flex-col gap-3">{cards}</View> : <Empty t={empty} />}
	</View>
);

const ReceiptRow = ({ label, value }: { label: Localized; value: JetElement }) => (
	<View mod="flex-col">
		<View mod="flex-row flex-wrap items-center justify-between gap-3 py-2">
			<Text t={label} mod={`${textSecondary} text-sm shrink-0`} />
			{value}
		</View>
		<Divider />
	</View>
);

const body = (report: Report, download: string | undefined, note: TimelineRichTextFragment | undefined): JetElement[] => {
	const tz = report.period.timezone || "UTC";
	const rows: JetElement[] = [
		...(note ? [<TimelineRichText richText={note} mod={`${textSecondary} leading-snug`} />] : []),
		<View mod="flex-col">
			<ReceiptRow label={copy.postsLabel} value={<Text t={copy.count(report.postCount)} mod={valueMod} />} />
			<ReceiptRow
				label={copy.periodLabel}
				value={
					<View mod="flex-row flex-wrap items-center gap-1">
						<DateTime time={new Date(`${report.period.startDate}T00:00:00Z`)} dateStyle="medium" timeZone="UTC" mod={valueMod} />
						<Text t={copy.to} mod={valueMod} />
						<DateTime time={new Date(`${report.period.endDate}T00:00:00Z`)} dateStyle="medium" timeZone="UTC" mod={valueMod} />
						<Text t={tz} mod={valueMod} />
					</View>
				}
			/>
			<ReceiptRow
				label={copy.generatedLabel}
				value={<DateTime time={new Date(report.generatedAt)} dateStyle="medium" timeStyle="long" timeZone="UTC" mod={valueMod} />}
			/>
		</View>,
		<Section
			title={copy.postLabels}
			blurb={copy.postLabelsBlurb}
			empty={copy.noPostLabels}
			cards={report.postLabels.map((label: PostLabel) => (
				<LabelCard
					code={label.label}
					stat={copy.postStat(label.posts, label.totalPostsInMonth, label.percentageOfPosts)}
					about={label.about}
					effect={label.effect}
					reason={label.reason}
					restriction={label.restriction}
					scope={label.scope ?? "Post-level"}
					affectedTarget={label.affectedTarget}
					affectedMedia={label.affectedMedia}
					part={label.posts}
					total={label.totalPostsInMonth}
				/>
			))}
		/>,
		<Section
			title={copy.accountLabels}
			blurb={copy.accountLabelsBlurb}
			empty={copy.noAccountLabels}
			cards={report.accountLabels.map((label: AccountLabel) => (
				<LabelCard
					code={label.label}
					stat={copy.dayStat(label.days, label.daysInPeriod, label.percentageOfDays)}
					about={label.about}
					effect={label.effect}
					reason={label.reason}
					restriction={label.restriction}
					scope={label.scope ?? "Account-level"}
					affectedTarget={label.affectedTarget}
					part={label.days}
					total={label.daysInPeriod}
				/>
			))}
		/>,
	];
	if (download) {
		rows.push(
			<Link url={download} scribe:press={{ element: "download_report" }} mod="w-full hover:opacity-80">
				<View mod="flex-row items-center justify-between py-2 gap-3">
					<Text t={copy.download} mod={`${textPrimary} text-sm flex-1`} />
					<Icon key="incoming" size={18} mod={textSecondary} />
				</View>
			</Link>,
		);
	}
	return rows;
};

export default page("logged_in.under_the_hood.report", T, async (ctx) => {
	const fs = await ctx.switches();
	const mock: Mock | undefined = canMock(ctx) ? ctx.mock : undefined;
	const state = fs.isTrue("rweb_under_the_hood_report_enabled") ? await loadReport(ctx, mock) : { status: "pilot" as const };
	const content =
		state.status === "ok" && state.report ? (
			body(state.report, state.rawJson ? downloadUrl(ctx) : undefined, state.note)
		) : (
			<Body t={state.status === "pilot" ? pilotCopy : copy.unavailable} />
		);
	return (
		<Screen title={copy.title} page="under_the_hood_report">
			{content}
		</Screen>
	);
});
