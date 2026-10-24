export type PostLabel = {
	label: string;
	about: string;
	effect: string;
	posts: number;
	totalPostsInMonth: number;
	percentageOfPosts: string;
};

export type AccountLabel = {
	label: string;
	about: string;
	effect: string;
	days: number;
	daysInPeriod: number;
	percentageOfDays: string;
};

export type PeriodStatus = { served: string; target: string; status: string; nextReportNotBefore: string };

export type Report = {
	period: { startDate: string; endDate: string; timezone: string };
	generatedAt: string;
	postCount: number;
	postLabels: PostLabel[];
	accountLabels: AccountLabel[];
	periodStatus?: PeriodStatus;
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

const postLabel = (v: unknown): PostLabel | undefined => {
	if (!isRecord(v) || !str(v.label)) return undefined;
	return {
		label: str(v.label),
		about: str(v.about),
		effect: str(v.effect),
		posts: num(v.posts),
		totalPostsInMonth: num(v.totalPostsInMonth),
		percentageOfPosts: str(v.percentageOfPosts),
	};
};

const accountLabel = (v: unknown): AccountLabel | undefined => {
	if (!isRecord(v) || !str(v.label)) return undefined;
	return {
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

const periodStatus = (v: unknown): PeriodStatus | undefined => {
	if (!isRecord(v) || !str(v.served) || !str(v.target) || !str(v.status) || !str(v.nextReportNotBefore)) return undefined;
	return { served: str(v.served), target: str(v.target), status: str(v.status), nextReportNotBefore: str(v.nextReportNotBefore) };
};

export const parseReport = (json: string | undefined): Report | undefined => {
	if (!json) return undefined;
	let raw: unknown;
	try {
		raw = JSON.parse(json);
	} catch {
		return undefined;
	}
	if (!isRecord(raw)) return undefined;
	const period = isRecord(raw.period) ? raw.period : {};
	const status = periodStatus(raw.periodStatus);
	return {
		period: { startDate: str(period.startDate), endDate: str(period.endDate), timezone: str(period.timezone) },
		generatedAt: str(raw.generatedAt),
		postCount: num(raw.postCount),
		postLabels: list(raw.postLabels, postLabel),
		accountLabels: list(raw.accountLabels, accountLabel),
		...(status ? { periodStatus: status } : {}),
	};
};

export type EligibilityCheck = { key: string; value: boolean };

export type ReportInfo = {
	reportPeriod: string;
	eligibilityChecks: EligibilityCheck[];
	servedPeriod?: string;
	targetPeriod?: string;
	periodStatus?: string;
	nextReportNotBefore?: string;
	availableMonths?: number[];
};
