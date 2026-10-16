import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import type { Report, ReportInfo } from "./report";

export const MOCK_INFO: ReportInfo = {
	reportPeriod: "August 2026",
	eligibilityChecks: [
		{ key: "10 or more posts in the prior month", value: true },
		{ key: "Account at least 1 year old", value: true },
	],
};

export const MOCK_INFO_INELIGIBLE: ReportInfo = {
	reportPeriod: "August 2026",
	eligibilityChecks: [
		{ key: "10 or more posts in the prior month", value: false },
		{ key: "Account at least 1 year old", value: true },
	],
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

export const MOCK_POST_IDS: Report = {
	...MOCK_LABELED,
	postLabels: [
		{
			...MOCK_LABELED.postLabels[0],
			postIds: ["2080000000000000003", "2080000000000000002", "2080000000000000001"],
			postIdsTruncated: false,
			postIdsAvailable: true,
		},
		{
			...MOCK_LABELED.postLabels[1],
			postIds: ["2080000000000000004"],
			postIdsTruncated: false,
			postIdsAvailable: true,
		},
	],
};

export const MOCK_POST_IDS_TRUNCATED: Report = {
	...MOCK_POST_IDS,
	postLabels: [
		{
			...MOCK_POST_IDS.postLabels[0],
			postIds: ["2080000000000000003", "2080000000000000002"],
			postIdsTruncated: true,
		},
		MOCK_POST_IDS.postLabels[1],
	],
};

export const MOCK_POST_IDS_UNAVAILABLE: Report = {
	...MOCK_LABELED,
	postLabels: MOCK_LABELED.postLabels.map((label) => ({ ...label, postIds: [], postIdsTruncated: false, postIdsAvailable: false })),
};
