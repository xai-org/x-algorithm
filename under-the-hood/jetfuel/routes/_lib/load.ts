import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import type { Context } from "@serving";
import { isProd } from "@serving";
import {
	MOCK_CAPPED,
	MOCK_CURRENT,
	MOCK_DATA_PENDING,
	MOCK_EMPTY,
	MOCK_INFO,
	MOCK_INFO_CAPPED,
	MOCK_INFO_CURRENT,
	MOCK_INFO_DATA_PENDING,
	MOCK_INFO_INELIGIBLE,
	MOCK_LABELED,
	MOCK_NOTE,
} from "./mock";
import { parseReport, type Report, type ReportInfo } from "./report";

export type Mock = "none" | "pilot" | "ineligible" | "empty" | "labels" | "current" | "data_pending" | "capped";

export type ReportState =
	| { status: "unavailable" }
	| { status: "pilot" }
	| { status: "ok"; info: ReportInfo; report: Report | undefined; rawJson: string | undefined; note: TimelineRichTextFragment | undefined };

const MOCKS: Record<Mock, ReportState> = {
	none: { status: "unavailable" },
	pilot: { status: "pilot" },
	ineligible: { status: "ok", info: MOCK_INFO_INELIGIBLE, report: undefined, rawJson: undefined, note: undefined },
	empty: { status: "ok", info: MOCK_INFO, report: MOCK_EMPTY, rawJson: JSON.stringify(MOCK_EMPTY, null, 2), note: MOCK_NOTE },
	labels: { status: "ok", info: MOCK_INFO, report: MOCK_LABELED, rawJson: JSON.stringify(MOCK_LABELED, null, 2), note: MOCK_NOTE },
	current: { status: "ok", info: MOCK_INFO_CURRENT, report: MOCK_CURRENT, rawJson: JSON.stringify(MOCK_CURRENT, null, 2), note: MOCK_NOTE },
	data_pending: {
		status: "ok",
		info: MOCK_INFO_DATA_PENDING,
		report: MOCK_DATA_PENDING,
		rawJson: JSON.stringify(MOCK_DATA_PENDING, null, 2),
		note: MOCK_NOTE,
	},
	capped: { status: "ok", info: MOCK_INFO_CAPPED, report: MOCK_CAPPED, rawJson: JSON.stringify(MOCK_CAPPED, null, 2), note: MOCK_NOTE },
};

export const canMock = (ctx: Context): boolean => !isProd || ctx.isDogfood;

export const loadReport = async (
	ctx: Context,
	mock: Mock | undefined,
	options?: { headers: Record<string, string> },
): Promise<ReportState> => {
	if (mock && canMock(ctx)) return MOCKS[mock];
	try {
		const result = await ctx.bird.$gql.UserUnderTheHoodReport(undefined, options);
		const data = result.user_under_the_hood_report;
		if (!data) return { status: "unavailable" };
		if (!data.report_info) return { status: "pilot" };
		return {
			status: "ok",
			info: {
				reportPeriod: data.report_info.report_period ?? "",
				servedPeriod: data.report_info.served_period ?? undefined,
				targetPeriod: data.report_info.target_period ?? undefined,
				periodStatus: data.report_info.period_status ?? undefined,
				nextReportNotBefore: data.report_info.next_report_not_before ?? undefined,
				eligibilityChecks: (data.report_info.eligibility_checks ?? []).map((c) => ({ key: c.key, value: c.value })),
			},
			report: parseReport(data.report_json),
			rawJson: data.report_json ?? undefined,
			note: data.report_info.note,
		};
	} catch (e) {
		!isProd && console.error("[under_the_hood] report fetch failed", e);
		return { status: "unavailable" };
	}
};
