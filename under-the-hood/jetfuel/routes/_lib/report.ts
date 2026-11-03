import type { TimelineRichTextFragment } from "@bird/gql/sdk";

type Appeal = { howToAppeal?: string };

export type PostLabel = Appeal & {
	label: string;
	about: string;
	effect: string;
	posts: number;
	totalPostsInMonth: number;
	percentageOfPosts: string;
	postIds: string[];
	postIdsNotShownNote?: string;
};

export type AccountLabel = Appeal & {
	label: string;
	about: string;
	effect: string;
	days: number;
	daysInPeriod: number;
	percentageOfDays: string;
};

export type Report = {
	period: { startDate: string; endDate: string; timezone: string };
	generatedAt: string;
	postCount: number;
	maxPostIdsPerLabel: number;
	postLabels: PostLabel[];
	accountLabels: AccountLabel[];
};

const str = (v: unknown): string => (typeof v === "string" ? v : "");

const num = (v: unknown): number => {
	if (typeof v === "number") return Number.isFinite(v) ? v : 0;
	if (typeof v === "string") {
		const n = Number(v);
		return Number.isFinite(n) ? n : 0;
	}
	return 0;
};

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

const isPostId = (v: unknown): v is string => typeof v === "string" && /^[1-9]\d{0,19}$/.test(v);

const byId = (a: string, b: string): number => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);

const postIds = (v: unknown): string[] => (Array.isArray(v) ? [...new Set(v.filter(isPostId))].sort(byId) : []);

export const TWITTER_EPOCH_MS = 1288834974657n;

export const postCreatedAt = (id: string): Date => new Date(Number((BigInt(id) >> 22n) + TWITTER_EPOCH_MS));

export const postUrl = (id: string): string => `https://x.com/i/status/${id}`;

export const showsMostRecent = (label: PostLabel, maxPostIdsPerLabel: number): boolean =>
	maxPostIdsPerLabel > 0 && label.posts > maxPostIdsPerLabel;

const appeal = (v: Record<string, unknown>): Appeal => (str(v.howToAppeal) ? { howToAppeal: str(v.howToAppeal) } : {});

const postLabel = (v: unknown): PostLabel | undefined => {
	if (!isRecord(v) || !str(v.label)) return undefined;
	return {
		...appeal(v),
		label: str(v.label),
		about: str(v.about),
		effect: str(v.effect),
		posts: num(v.posts),
		totalPostsInMonth: num(v.totalPostsInMonth),
		percentageOfPosts: str(v.percentageOfPosts),
		postIds: postIds(v.postIds),
		...(str(v.postIdsNotShownNote) ? { postIdsNotShownNote: str(v.postIdsNotShownNote) } : {}),
	};
};

const accountLabel = (v: unknown): AccountLabel | undefined => {
	if (!isRecord(v) || !str(v.label)) return undefined;
	return {
		...appeal(v),
		label: str(v.label),
		about: str(v.about),
		effect: str(v.effect),
		days: num(v.days),
		daysInPeriod: num(v.daysInPeriod),
		percentageOfDays: str(v.percentageOfDays),
	};
};

const list = <T>(v: unknown, map: (item: unknown) => T | undefined): T[] =>
	Array.isArray(v) ? v.map(map).filter((x): x is T => x !== undefined) : [];

export const parseReport = (json: string | undefined, maxPostIdsPerLabelSent?: number): Report | undefined => {
	if (!json) return undefined;
	let raw: unknown;
	try {
		raw = JSON.parse(json);
	} catch {
		return undefined;
	}
	if (!isRecord(raw)) return undefined;
	const period = isRecord(raw.period) ? raw.period : {};
	const maxPostIdsPerLabel = Math.max(0, Math.floor(num(maxPostIdsPerLabelSent)));
	const postLabels = list(raw.postLabels, postLabel);
	return {
		period: { startDate: str(period.startDate), endDate: str(period.endDate), timezone: str(period.timezone) },
		generatedAt: str(raw.generatedAt),
		postCount: num(raw.postCount),
		maxPostIdsPerLabel,
		postLabels:
			maxPostIdsPerLabel > 0 ? postLabels.map((label) => ({ ...label, postIds: label.postIds.slice(-maxPostIdsPerLabel) })) : postLabels,
		accountLabels: list(raw.accountLabels, accountLabel),
	};
};

export type EligibilityCheck = { key: string; value: boolean };

export type ReportInfo = { reportPeriod: string; eligibilityChecks: EligibilityCheck[] };

export type LabelAppeals = {
	posts: ReadonlyMap<string, TimelineRichTextFragment>;
	accounts: ReadonlyMap<string, TimelineRichTextFragment>;
};

export const NO_APPEALS: LabelAppeals = { posts: new Map(), accounts: new Map() };

export const labelAppeals = (
	posts: { key: string; value: TimelineRichTextFragment }[] | undefined,
	accounts: { key: string; value: TimelineRichTextFragment }[] | undefined,
): LabelAppeals => ({
	posts: new Map((posts ?? []).map(({ key, value }) => [key, value])),
	accounts: new Map((accounts ?? []).map(({ key, value }) => [key, value])),
});
