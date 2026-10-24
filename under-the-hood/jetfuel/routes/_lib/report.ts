export type PostLabel = {
	label: string;
	about: string;
	effect: string;
	reason?: string;
	restriction?: string;
	scope?: string;
	affectedTarget?: string;
	affectedMedia?: string;
	posts: number;
	totalPostsInMonth: number;
	percentageOfPosts: string;
};

export type AccountLabel = {
	label: string;
	about: string;
	effect: string;
	reason?: string;
	restriction?: string;
	scope?: string;
	affectedTarget?: string;
	days: number;
	daysInPeriod: number;
	percentageOfDays: string;
};

export type Report = {
	period: { startDate: string; endDate: string; timezone: string };
	generatedAt: string;
	postCount: number;
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

const postLabel = (v: unknown): PostLabel | undefined => {
	if (!isRecord(v) || !str(v.label)) return undefined;
	return {
		label: str(v.label),
		about: str(v.about),
		effect: str(v.effect),
		reason: str(v.reason) || undefined,
		restriction: str(v.restriction) || undefined,
		scope: str(v.scope) || undefined,
		affectedTarget: str(v.affectedTarget) || undefined,
		affectedMedia: str(v.affectedMedia) || undefined,
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
		reason: str(v.reason) || undefined,
		restriction: str(v.restriction) || undefined,
		scope: str(v.scope) || undefined,
		affectedTarget: str(v.affectedTarget) || undefined,
		days: num(v.days),
		daysInPeriod: num(v.daysInPeriod),
		percentageOfDays: str(v.percentageOfDays),
	};
};

const list = <T>(v: unknown, map: (item: unknown) => T | undefined): T[] =>
	Array.isArray(v) ? v.map(map).filter((x): x is T => x !== undefined) : [];

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
	return {
		period: { startDate: str(period.startDate), endDate: str(period.endDate), timezone: str(period.timezone) },
		generatedAt: str(raw.generatedAt),
		postCount: num(raw.postCount),
		postLabels: list(raw.postLabels, postLabel),
		accountLabels: list(raw.accountLabels, accountLabel),
	};
};

export type EligibilityCheck = { key: string; value: boolean };

export type ReportInfo = { reportPeriod: string; eligibilityChecks: EligibilityCheck[] };
