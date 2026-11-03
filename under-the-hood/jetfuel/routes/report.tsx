import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import { jsx, actions, atom, Button, DateTime, Icon, If, Link, Text, View, l, type JetElement, type Localized } from "@jetfuel";
import { page, t } from "@serving";
import type { InputAction } from "../../../lib/jetfuel/core/actions";
import { cardSurface, textPrimary, textSecondary } from "../../../ui/themes";
import { TimelineRichText } from "../events/components/TimelineRichText";
import { canMock, loadReport, type Mock } from "./_lib/load";
import { postCreatedAt, postUrl, type AccountLabel, type LabelAppeals, type PostLabel, type Report, showsMostRecent } from "./_lib/report";
import { Body, Divider, Screen, downloadUrl, pilotCopy } from "./_lib/ui";

const T = t.Object({
	mock: t.Optional(
		t.Union([t.Literal("none"), t.Literal("pilot"), t.Literal("ineligible"), t.Literal("empty"), t.Literal("labels"), t.Literal("launch")]),
	),
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
		"Labels that could limit post visibility and were on your posts at any point in this period.",
		"Explains the post labels section on the Under the hood report page.",
	),
	noPostLabels: l("None in this period", "Empty state for the post labels section on the Under the hood report page."),
	accountLabels: l("Account labels", "Section title on the Under the hood report page for labels applied to the whole account."),
	accountLabelsBlurb: l(
		"Labels that could limit post visibility and were on your account at any point in this period.",
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
	showPosts: (count: number) =>
		l("Show posts ({count})", "Expands the list of posts that carried a label on the Under the hood report page, e.g. 'Show posts (3)'.", {
			count,
		}),
	showRecentPosts: (count: number) =>
		l(
			"Show posts ({count} most recent)",
			"Expands the list of posts that carried a label on the Under the hood report page when only the most recent of them are listed, e.g. 'Show posts (50 most recent)'.",
			{ count },
		),
	hidePosts: l("Hide posts", "Collapses the list of posts that carried a label on the Under the hood report page."),
	showAllPosts: (count: number) =>
		l(
			"Show all {count} posts",
			"Link under a short list of labeled posts on the Under the hood report page that reveals the rest, e.g. 'Show all 14 posts'. The count is always more than 10.",
			{ count },
		),
	showAllRecentPosts: (count: number) =>
		l(
			"Show all {count} most recent posts",
			"Link under a short list of labeled posts on the Under the hood report page that reveals the rest of the label's most recent posts (the label had more posts than the page lists), e.g. 'Show all 50 most recent posts'. The count is always more than 10.",
			{ count },
		),
	download: l("Download this report (JSON)", "Link on the Under the hood report page that downloads the raw report file (web only)."),
	howToAppeal: l("How to appeal", "Caption above how to appeal a label on the Under the hood report page."),
};

const quietPill = "border-xs rounded-md px-2 py-1 light:border-gray-400 dark:border-gray-500 dim:border-gray-400";
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

const POST_PREVIEW_COUNT = 10;

type Toggle = ReturnType<typeof atom.bool>;

const Disclosure = ({ label, icon, press }: { label: Localized; icon: "chevron_down" | "chevron_up"; press: InputAction }) => (
	<Button on:press={press} scribe:press={{ element: "toggle_labeled_posts" }} mod="w-full hover:opacity-80">
		<View mod="flex-row items-center justify-between gap-3 py-2">
			<Text t={label} mod={`${textPrimary} text-sm font-bold`} />
			<Icon key={icon} size={18} mod={textPrimary} />
		</View>
	</Button>
);

const ShowAll = ({ label, toggle }: { label: Localized; toggle: Toggle }) => (
	<Button on:press={toggle.set(true)} scribe:press={{ element: "show_all_labeled_posts" }} mod="w-full">
		<View mod="flex-row py-2">
			<Text t={label} mod="text-blue-500 hover:text-blue-700 text-sm" />
		</View>
	</Button>
);

const PostRow = ({ id }: { id: string }) => (
	<Link url={postUrl(id)} scribe:press={{ element: "labeled_post" }} mod="w-full hover:opacity-80">
		<View mod="flex-row items-center gap-3 py-2">
			<DateTime time={postCreatedAt(id)} month="short" day="numeric" timeZone="UTC" mod={`${textPrimary} text-sm w-14`} />
			<Text t={id} mod={`${textSecondary} font-mono text-sm flex-1`} numberOfLines={1} truncate="end" />
			<Icon key="arrow_up_right" size={16} mod={textSecondary} />
		</View>
	</Link>
);

const HowToAppeal = ({ richText }: { richText: TimelineRichTextFragment }) => (
	<View mod="flex-col gap-0.5">
		<Text t={copy.howToAppeal} mod={`${textSecondary} text-sm`} />
		<TimelineRichText richText={richText} mod={`${textPrimary} leading-snug`} />
	</View>
);

const appealFor = (appeals: ReadonlyMap<string, TimelineRichTextFragment>, label: string): JetElement | undefined => {
	const richText = appeals.get(label);
	return richText ? <HowToAppeal richText={richText} /> : undefined;
};

const PostList = ({ ids, mostRecent, stateId }: { ids: string[]; mostRecent: boolean; stateId: string }) => {
	const open = atom.bool({ default: false, id: `${stateId}_open` });
	const all = atom.bool({ default: false, id: `${stateId}_all` });
	const rest = ids.slice(POST_PREVIEW_COUNT);
	const show = mostRecent ? copy.showRecentPosts(ids.length) : copy.showPosts(ids.length);
	const showRest = mostRecent ? copy.showAllRecentPosts(ids.length) : copy.showAllPosts(ids.length);
	return (
		<View mod="flex-col">
			<Divider />
			<If cond={open.is(true)} else={<Disclosure label={show} icon="chevron_down" press={open.set(true)} />}>
				<View mod="flex-col">
					<Disclosure label={copy.hidePosts} icon="chevron_up" press={actions.multi([open.set(false), all.set(false)])} />
					{ids.slice(0, POST_PREVIEW_COUNT).map((id) => (
						<PostRow id={id} />
					))}
					{rest.length > 0 ? (
						<If cond={all.is(true)} else={<ShowAll label={showRest} toggle={all} />}>
							<View mod="flex-col">
								{rest.map((id) => (
									<PostRow id={id} />
								))}
							</View>
						</If>
					) : null}
				</View>
			</If>
		</View>
	);
};

const PostsNotShown = ({ reason }: { reason: string }) => (
	<View mod="flex-col">
		<Divider />
		<Text t={reason} mod={`${textSecondary} text-sm leading-snug py-2`} />
	</View>
);

const postsFooter = (label: PostLabel, report: Report): JetElement | undefined => {
	if (label.postIds.length > 0) {
		return (
			<PostList
				ids={label.postIds}
				mostRecent={showsMostRecent(label, report.maxPostIdsPerLabel)}
				stateId={`uth_report_posts_${report.period.startDate}_${label.label}`}
			/>
		);
	}
	return label.postIdsNotShownNote ? <PostsNotShown reason={label.postIdsNotShownNote} /> : undefined;
};

const LabelCard = ({
	code,
	stat,
	about,
	effect,
	part,
	total,
	extra,
	footer,
}: {
	code: string;
	stat: Localized;
	about: string;
	effect: string;
	part: number;
	total: number;
	extra?: JetElement;
	footer?: JetElement;
}) => (
	<View mod={`${cardSurface} flex-col gap-2 ${footer ? "px-4 pt-4 pb-1" : "p-4"}`}>
		<View mod="flex-row flex-wrap items-center justify-between gap-3">
			<View mod={quietPill}>
				<Text t={code} mod={`${textPrimary} font-mono text-sm`} />
			</View>
			<Text t={stat} mod={`${textPrimary} text-sm font-bold`} />
		</View>
		<ShareBar part={part} total={total} />
		<View mod="flex-col gap-0.5">
			<Text t={copy.about} mod={`${textSecondary} text-sm`} />
			<Text t={about} mod={`${textPrimary} text-sm leading-snug`} />
		</View>
		<View mod="flex-col gap-0.5">
			<Text t={copy.effect} mod={`${textSecondary} text-sm`} />
			<Text t={effect} mod={`${textPrimary} text-sm leading-snug`} />
		</View>
		{extra ?? null}
		{footer ?? null}
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

type BodyOptions = {
	download: string | undefined;
	note: TimelineRichTextFragment | undefined;
	showPostIds: boolean;
	appeals: LabelAppeals;
};

const body = (report: Report, { download, note, showPostIds, appeals }: BodyOptions): JetElement[] => {
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
					part={label.posts}
					total={label.totalPostsInMonth}
					extra={appealFor(appeals.posts, label.label)}
					footer={showPostIds ? postsFooter(label, report) : undefined}
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
					part={label.days}
					total={label.daysInPeriod}
					extra={appealFor(appeals.accounts, label.label)}
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
			body(state.report, {
				download: state.rawJson ? downloadUrl(ctx) : undefined,
				note: state.note,
				showPostIds: fs.isTrue("rweb_under_the_hood_report_post_ids_enabled"),
				appeals: state.appeals,
			})
		) : (
			<Body t={state.status === "pilot" ? pilotCopy : copy.unavailable} />
		);
	return (
		<Screen title={copy.title} page="under_the_hood_report">
			{content}
		</Screen>
	);
});
