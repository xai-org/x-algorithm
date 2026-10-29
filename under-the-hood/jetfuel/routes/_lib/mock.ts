import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import type { Report, ReportInfo } from "./report";

export const MOCK_INFO: ReportInfo = {
	reportPeriod: "August 2026",
	eligibilityChecks: [
		{ key: "10 or more posts in the report month", value: true },
		{ key: "Account at least 1 year old", value: true },
	],
};

export const MOCK_INFO_INELIGIBLE: ReportInfo = {
	reportPeriod: "August 2026",
	eligibilityChecks: [
		{ key: "10 or more posts in the report month", value: false },
		{ key: "Account at least 1 year old", value: true },
	],
};

export const MOCK_INFO_CURRENT: ReportInfo = {
	...MOCK_INFO,
	servedPeriod: "August 2026",
	targetPeriod: "August 2026",
	periodStatus: "current",
	nextReportNotBefore: "2026-10-10",
};

export const MOCK_INFO_DATA_PENDING: ReportInfo = {
	...MOCK_INFO_CURRENT,
	targetPeriod: "September 2026",
	periodStatus: "data_pending",
};

export const MOCK_INFO_CAPPED: ReportInfo = {
	...MOCK_INFO_CURRENT,
	targetPeriod: "September 2026",
	periodStatus: "capped",
};

export const MOCK_NOTE: TimelineRichTextFragment = {
	__typename: "TimelineRichText",
	text: "This is a placeholder introductory note, and you can see how links look.",
	entities: [
		{
			__typename: "TimelineRichTextEntity",
			from_index: 61,
			to_index: 71,
			ref: {
				__typename: "TimelineUrl",
				url: "https://github.com/xai-org/x-algorithm",
				url_type: "ExternalUrl",
			},
		},
	],
};

const period = { startDate: "2026-08-01", endDate: "2026-08-31", timezone: "UTC" };
const generatedAt = "2026-09-09T23:59:59Z";

export const MOCK_EMPTY: Report = { period, generatedAt, postCount: 27, postLabels: [], accountLabels: [] };

export const MOCK_LABELED: Report = {
	period,
	generatedAt,
	postCount: 40,
	postLabels: [
		{
			label: "SPAM_HIGH_RECALL",
			about: "Post detected by automated systems as one that may contain spam.",
			effect: "Post hidden from recommendations to non-followers.",
			posts: 3,
			totalPostsInMonth: 40,
			percentageOfPosts: "7.50%",
		},
		{
			label: "NSFW_TEXT",
			about: "Post detected by automated systems as likely to contain adult sexually explicit language.",
			effect:
				"Post hidden from recommendations to non-followers, and hidden from underage users, users without a stated age, and logged out users.",
			posts: 1,
			totalPostsInMonth: 40,
			percentageOfPosts: "2.50%",
		},
	],
	accountLabels: [
		{
			label: "SpamHighRecall",
			about: "Account detected by automated systems as likely to post spam.",
			effect: "The account's posts are hidden from recommendations to non-followers.",
			days: 5,
			daysInPeriod: 31,
			percentageOfDays: "16.12%",
		},
	],
};

export const MOCK_CURRENT: Report = {
	...MOCK_EMPTY,
	periodStatus: { served: "August 2026", target: "August 2026", status: "current", nextReportNotBefore: "2026-10-10" },
};

export const MOCK_DATA_PENDING: Report = {
	...MOCK_CURRENT,
	periodStatus: { served: "August 2026", target: "September 2026", status: "data_pending", nextReportNotBefore: "2026-10-10" },
};

export const MOCK_CAPPED: Report = {
	...MOCK_CURRENT,
	periodStatus: { served: "August 2026", target: "September 2026", status: "capped", nextReportNotBefore: "2026-10-10" },
};
