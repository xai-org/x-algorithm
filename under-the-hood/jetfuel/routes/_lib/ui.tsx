import { jsx, Icon, List2, Nav, Page, Text, View, l, type JetElement, type Localized } from "@jetfuel";
import { type Context, isProd } from "@serving";
import { backgroundPrimary, divider, textPrimary, textSecondary } from "../../../../ui/themes";

export const OPEN_SOURCE_URL = "https://github.com/xai-org/x-algorithm";

export const REPORT_PATH = "/under_the_hood/report";

export const pilotCopy = l(
	"We're testing this new feature with a small group to collect feedback. When it's more widely available, you'll be able to view your report here.",
	"Shown on the Under the hood pages to users who are not in the pilot group yet.",
);

export const downloadUrl = (ctx: Context): string | undefined => {
	if (ctx.clientType !== "web") return undefined;
	const jfHost = ctx.headers.origin?.includes("twitter.com") ? "jf.twitter.com" : "jf.x.com";
	const origin = isProd ? `https://${jfHost}` : `https://${ctx.headers.host ?? "localhost.x.com:3000"}`;
	return `${origin}/under_the_hood/download`;
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

export const SectionTitle = ({ t }: { t: Localized }) => <Text t={t} mod={`${textPrimary} text-xl font-bold`} />;

export const SwitchIndicator = ({ on }: { on: boolean }) => (
	<View
		mod={`flex-row items-center w-12 h-7 px-0.5 rounded-full shrink-0 ${on ? "justify-end bg-green-500" : "justify-start light:bg-gray-300 dark:bg-zinc-700 dim:bg-gray-600"}`}
	>
		<View mod="w-6 h-6 rounded-full bg-white" />
	</View>
);
