import type { TimelineRichTextFragment } from "@bird/gql/sdk";
import type { Context } from "@serving";
import { isProd } from "@serving";
import {
	MOCK_APPEALS,
	MOCK_EMPTY,
	MOCK_INFO,
	MOCK_INFO_INELIGIBLE,
	MOCK_LABELED,
	MOCK_LAUNCH,
	MOCK_LAUNCH_INFO,
	MOCK_LAUNCH_NOTE,
	MOCK_NOTE,
} from "./mock";
import { NO_APPEALS, labelAppeals, parseReport, type LabelAppeals, type Report, type ReportInfo } from "./report";

export type Mock = "none" | "pilot" | "ineligible" | "empty" | "labels" | "launch";

export type ReportState =
	| { status: "unavailable" }
	| { status: "pilot" }
	| {
			status: "ok";
			info: ReportInfo;
			report: Report | undefined;
			rawJson: string | undefined;
			note: TimelineRichTextFragment | undefined;
			appeals: LabelAppeals;
	  };

const MOCKS: Record<Mock, ReportState> = {
	none: { status: "unavailable" },
	pilot: { status: "pilot" },
	ineligible: { status: "ok", info: MOCK_INFO_INELIGIBLE, report: undefined, rawJson: undefined, note: undefined, appeals: NO_APPEALS },
	empty: {
		status: "ok",
		info: MOCK_INFO,
		report: MOCK_EMPTY,
		rawJson: JSON.stringify(MOCK_EMPTY, null, 2),
		note: MOCK_NOTE,
		appeals: NO_APPEALS,
	},
	labels: {
		status: "ok",
		info: MOCK_INFO,
		report: MOCK_LABELED,
		rawJson: JSON.stringify(MOCK_LABELED, null, 2),
		note: MOCK_NOTE,
		appeals: MOCK_APPEALS,
	},
	launch: {
		status: "ok",
		info: MOCK_LAUNCH_INFO,
		report: MOCK_LAUNCH,
		rawJson: JSON.stringify(MOCK_LAUNCH, null, 2),
		note: MOCK_LAUNCH_NOTE,
		appeals: MOCK_APPEALS,
	},
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
			report: parseReport(data.report_json, data.report_info.max_post_ids_per_label),
			rawJson: data.report_json ?? undefined,
			note: data.report_info.note,
			appeals: labelAppeals(data.report_info.post_label_appeals, data.report_info.account_label_appeals),
		};
	} catch (e) {
		!isProd && console.error("[under_the_hood] report fetch failed", e);
		return { status: "unavailable" };
	}
};
