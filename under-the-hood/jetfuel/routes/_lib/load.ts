import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import type { Context } from "@serving";
import { isProd } from "@serving";
import { MOCK_EMPTY, MOCK_INFO, MOCK_INFO_INELIGIBLE, MOCK_LABELED, MOCK_NOTE, MOCK_POST_IDS, MOCK_POST_IDS_TRUNCATED, MOCK_POST_IDS_UNAVAILABLE } from "./mock";
import { parseReport, type Report, type ReportInfo } from "./report";

export type Mock = "none" | "pilot" | "ineligible" | "empty" | "labels" | "ids" | "ids_truncated" | "ids_unavailable";

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
	ids: { status: "ok", info: MOCK_INFO, report: MOCK_POST_IDS, rawJson: JSON.stringify(MOCK_POST_IDS, null, 2), note: MOCK_NOTE },
	ids_truncated: { status: "ok", info: MOCK_INFO, report: MOCK_POST_IDS_TRUNCATED, rawJson: JSON.stringify(MOCK_POST_IDS_TRUNCATED, null, 2), note: MOCK_NOTE },
	ids_unavailable: { status: "ok", info: MOCK_INFO, report: MOCK_POST_IDS_UNAVAILABLE, rawJson: JSON.stringify(MOCK_POST_IDS_UNAVAILABLE, null, 2), note: MOCK_NOTE },
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
