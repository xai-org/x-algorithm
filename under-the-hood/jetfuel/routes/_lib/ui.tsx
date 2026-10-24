import { jsx, Icon, List2, Nav, Page, Text, View, l, type JetElement, type Localized } from "@jetfuel";
import { type Context, isProd } from "@serving";
import { backgroundPrimary, divider, textPrimary, textSecondary } from "../../../../ui/themes";
import type { ReportInfo } from "./report";

export const OPEN_SOURCE_URL = "https://github.com/xai-org/x-algorithm";

export const REPORT_PATH = "/under_the_hood/report";

export const pilotCopy = l(
	"We're testing this new feature with a small group to collect feedback. When it's more widely available, you'll be able to view your report here.",
	"Shown on the Under the hood pages to users who are not in the pilot group yet.",
);

export const downloadUrl = (ctx: Context, month?: string): string | undefined => {
	if (ctx.clientType !== "web") return undefined;
	const jfHost = ctx.headers.origin?.includes("twitter.com") ? "jf.twitter.com" : "jf.x.com";
	const origin = isProd ? `https://${jfHost}` : `https://${ctx.headers.host ?? "localhost.x.com:3000"}`;
	return reportUrl(`${origin}/under_the_hood/download`, month);
};

export const reportUrl = (path: string, month?: string, mock?: string): string => {
	const query = [month === undefined ? "" : `month=${encodeURIComponent(month)}`, mock ? `mock=${encodeURIComponent(mock)}` : ""].filter(Boolean);
	return query.length > 0 ? `${path}?${query.join("&")}` : path;
};

export const MonthSelector = ({ info, path, month, mock }: { info: ReportInfo; path: string; month?: string; mock?: string }) => {
	if (!info.availableMonths?.length) return null;
	return (
		<View mod="flex-col gap-2">
			<Body t={l("Report month", "Introduces the available historical months for an Under the hood report.")} />
			<View mod="flex-row flex-wrap gap-3">
				<Nav to={reportUrl(path, undefined, mock)} mod="hover:opacity-80">
					<Text t={l("Latest report", "Returns to the latest available Under the hood report.")} mod={`${textPrimary} text-sm ${month === undefined ? "font-bold" : "underline"}`} />
				</Nav>
				{info.availableMonths.map((availableMonth) => {
					const value = String(availableMonth).padStart(6, "0");
					return (
						<Nav to={reportUrl(path, value, mock)} mod="hover:opacity-80">
							<Text t={`${value.slice(0, 4)}-${value.slice(4)}`} mod={`${textPrimary} text-sm ${month === value ? "font-bold" : "underline"}`} />
						</Nav>
					);
				})}
			</View>
		</View>
	);
};

export const Screen = ({ title, page: pageName }: { title: Localized; page: string }, ...children: JetElement[]) => (
	<Page
		mod={`${backgroundPrimary} flex-col`}
		scribe:context={{ page: pageName }}
		headerBehavior="fixed"
		header={
			<View mod={`flex-row items-center px-2 pt-2 pb-2 safe-area-t ${backgroundPrimary}`}>
				<Nav to=":back" mod="w-10 h-10 items-center justify-center rounded-full">
					<Icon key="arrow_left" size={20} mod={textPrimary} />
				</Nav>
				<Text t={title} mod={`${textPrimary} text-lg font-bold ml-2`} numberOfLines={1} truncate="end" />
			</View>
		}
	>
		<List2 mod="flex-col w-full">
			<View mod="flex-col w-full px-4 pt-2 pb-8 safe-area-b gap-4">{children}</View>
		</List2>
	</Page>
);

export const Divider = () => <View mod={`h-px w-full ${divider}`} />;

export const Body = ({ t, mod = "" }: { t: Localized | string; mod?: string }) => (
	<Text t={t} mod={`${textSecondary} text-base leading-snug ${mod}`} />
);

export const ReportStatus = ({ info }: { info: ReportInfo }) => {
	if (!info.servedPeriod || !info.targetPeriod || !info.periodStatus || !info.nextReportNotBefore) return null;
	const t =
		info.periodStatus === "historical"
			? l("Showing historical report for {served}.", "Identifies an explicitly selected historical Under the hood report.", { served: info.servedPeriod })
			: info.periodStatus === "current"
				? l("Next report no earlier than {date} UTC.", "Calendar floor for the next Under the hood report, not a promise of availability.", {
						date: info.nextReportNotBefore,
					})
				: l(
						"{target} is not available yet; showing {served}. Next report no earlier than {date} UTC.",
						"Explains a capped or data-pending Under the hood report and the next report's calendar floor.",
						{ target: info.targetPeriod, served: info.servedPeriod, date: info.nextReportNotBefore },
					);
	return <Body t={t} mod="text-sm" />;
};

export const SectionTitle = ({ t }: { t: Localized }) => <Text t={t} mod={`${textPrimary} text-xl font-bold`} />;
