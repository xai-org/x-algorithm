# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import enum

from xai_proto import recsys_pb2

action_type_map_raw = {
    value_descriptor.name: value_descriptor.number
    for value_descriptor in recsys_pb2.ActionName.DESCRIPTOR.values
}

continuous_action_type_map = {
    value_descriptor.name: value_descriptor.number
    for value_descriptor in recsys_pb2.ContinuousActionName.DESCRIPTOR.values
}


def to_pascal_case(s):
    return "".join(word.capitalize() for word in s.lower().split("_"))


action_type_map = {to_pascal_case(k): v for k, v in action_type_map_raw.items()}


primary_engagement_to_action_types = {
    "IsFavorited": [
        "ServerTweetFav",
    ],
    "IsReplied": [
        "ServerTweetReply",
    ],
    "IsRetweeted": [
        "ServerTweetRetweet",
    ],
    "IsQuoted": [
        "ServerTweetQuote",
    ],
    "IsBookmarked": [
        "ClientTweetBookmark",
    ],
    "IsShared": [
        "ClientTweetShare",
    ],
    "IsSharedViaCopyLink": [
        "ClientTweetShareViaCopyLink",
    ],
    "IsSharedViaDirectMessage": [
        "ClientTweetClickSendViaDirectMessage",
    ],
    "IsNotInterestedIn": [
        "ClientTweetNotInterestedIn",
    ],
    "IsMuteAuthor": [
        "ClientTweetMuteAuthor",
    ],
    "IsBlockAuthor": [
        "ClientTweetBlockAuthor",
    ],
    "IsReported": [
        "ClientTweetReport",
    ],
    "IsVideoQualityViewed": [
        "ClientTweetVideoQualityView",
    ],
    "IsVideoOpened": [
        "ClientTweetVideoOpen",
    ],
    "IsProfileClicked": [
        "ClientTweetClickProfile",
    ],
    "IsRecapDwelled": [
        "ClientTweetRecapDwelled",
    ],
    "IsRecapNotDwelled": [
        "ClientTweetRecapNotDwelled",
    ],
    "IsOpenLink": [
        "ClientTweetOpenLink",
    ],
    "IsClicked": [
        "ClientTweetClick",
    ],
    "IsPostUnexplored": [
        "ServerTweetPostUnexplored",
    ],
    "IsPostedVideo": [
        "ServerTweetPostVideo",
    ],
    "IsExternalLinkLongDwelled": [
        "ClientTweetExternalLinkLongDwelled",
    ],
    "IsExternalLinkSessionLessThan3Sec": [
        "ClientExternalLinkSessionLessThan3Sec",
    ],
    "IsExternalLinkSessionLessThan5Sec": [
        "ClientExternalLinkSessionLessThan5Sec",
    ],
    "IsExternalLinkSessionLessThan10Sec": [
        "ClientExternalLinkSessionLessThan10Sec",
    ],
    "IsExternalLinkSessionMoreThan3Sec": [
        "ClientExternalLinkSessionMoreThan3Sec",
    ],
    "IsExternalLinkSessionMoreThan5Sec": [
        "ClientExternalLinkSessionMoreThan5Sec",
    ],
    "IsExternalLinkSessionMoreThan15Sec": [
        "ClientExternalLinkSessionMoreThan15Sec",
    ],
    "IsExternalLinkSessionMoreThan20Sec": [
        "ClientExternalLinkSessionMoreThan20Sec",
    ],
    "IsExternalLinkSessionMoreThan25Sec": [
        "ClientExternalLinkSessionMoreThan25Sec",
    ],
    "IsExternalLinkSessionMoreThan30Sec": [
        "ClientExternalLinkSessionMoreThan30Sec",
    ],
    "IsExternalLinkSessionMoreThan60Sec": [
        "ClientExternalLinkSessionMoreThan60Sec",
    ],
}


engagement_to_action_types = {
    **primary_engagement_to_action_types,
    "IsPhotoExpanded": [
        "ClientTweetPhotoExpand",
    ],
    "IsQuotedTweetPhotoExpanded": [
        "ClientQuotedTweetPhotoExpand",
    ],
    "IsLinkOpened": [
        "ClientTweetOpenLink",
    ],
    "IsHashtagClicked": [
        "ClientTweetClickHashtag",
    ],
    "IsMentionClicked": [
        "ClientTweetClickMentionScreenName",
    ],
    "IsAuthorFollowed": [
        "ClientTweetFollowAuthor",
    ],
    "IsTranslated": [
        "ClientTweetTranslateClick",
    ],
    "IsScreenshotTaken": [
        "ClientTweetTakeScreenshot",
    ],
    "IsShowMoreExpanded": [
        "ClientTweetShowMoreExpand",
    ],
    "IsGrokAnalyzeClicked": [
        "ClientTweetClickGrokAnalyze",
    ],
    "IsAuthorUnmuted": [
        "ClientTweetUnmuteAuthor",
    ],
    "IsUndoSeeFewer": [
        "ClientTweetUndoSeeFewer",
    ],
}

notification_engagement_to_action_types = {
    **engagement_to_action_types,
    "IsNotificationOpened": [
        "ClientNotificationOpen",
    ],
    "IsNotificationClicked": [
        "ClientNotificationClick",
    ],
    "IsNotificationDismissed": [
        "ClientNotificationDismiss",
    ],
    "IsNotificationSeeLessOften": [
        "ClientNotificationSeeLessOften",
    ],
    "IsNotificationImpression": [
        "ClientNotificationImpression",
    ],
    "IsNotificationSent": [
        "ClientNotificationSent",
    ],
    "IsNotificationSettingOptedOut": [
        "ClientNotificationSettingOptOut",
    ],
}

SEARCH_RELEVANCE_ACTION_INDICES = [
    recsys_pb2.ActionName.CLIENT_TWEET_RELEVANT_TO_SEARCH,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_RELEVANT_TO_SEARCH,
]

search_engagement_to_action_types = {
    **primary_engagement_to_action_types,
    "IsRelevantToSearch": [
        "ClientTweetRelevantToSearch",
    ],
    "IsNotRelevantToSearch": [
        "ClientTweetNotRelevantToSearch",
    ],
    "IsSearchQueryReformulated": ["ClientTweetSearchQueryReformulated"],
}

ads_conversion_engagement_to_action_types = {
    **primary_engagement_to_action_types,
    "IsPurchaseConversion": [
        "AdsPurchaseConversion",
    ],
    "IsSignupConversion": [
        "AdsSignUpConversion",
    ],
    "IsAddToCartConversion": [
        "AdsAddToCartConversion",
    ],
    "IsSiteVisitConversion": [
        "AdsSiteVisitConversion",
    ],
    "IsSearchConversion": [
        "AdsSearchConversion",
    ],
    "IsCustomConversion": [
        "AdsCustomConversion",
    ],
    "IsContentViewConversion": [
        "AdsContentViewConversion",
    ],
}

ads_p_conv_click_engagement_to_action_types = {
    **primary_engagement_to_action_types,
    "IsAttributedKeyClickConversion": [
        "AdsAttributedKeyClickConversion",
    ],
    "IsAttributedClickConversion": [
        "AdsAttributedClickConversion",
    ],
    "IsAttributedMactClickInstall": [
        "AdsAttributedMactClickInstall",
    ],
    "IsMmpClick": [
        "AdsMmpClick",
    ],
    "IsAttributedMactPurchase": [
        "AdsAttributedMactPurchase",
    ],
    "IsAttributedMactAddToCart": [
        "AdsAttributedMactAddToCart",
    ],
    "IsAttributedMactLevelAchieved": [
        "AdsAttributedMactLevelAchieved",
    ],
    "IsAttributedMactTutorialComplete": [
        "AdsAttributedMactTutorialComplete",
    ],
    "IsAttributedMactSignUp": [
        "AdsAttributedMactSignUp",
    ],
    "IsAttributedMactCustom": [
        "AdsAttributedMactCustom",
    ],
    "IsAttributedMactViewPurchase": [
        "AdsAttributedMactViewPurchase",
    ],
    "IsAttributedMactViewAddToCart": [
        "AdsAttributedMactViewAddToCart",
    ],
    "IsAttributedMactViewLevelAchieved": [
        "AdsAttributedMactViewLevelAchieved",
    ],
    "IsAttributedMactViewTutorialComplete": [
        "AdsAttributedMactViewTutorialComplete",
    ],
    "IsAttributedMactViewSignUp": [
        "AdsAttributedMactViewSignUp",
    ],
    "IsAttributedMactViewCustom": [
        "AdsAttributedMactViewCustom",
    ],
    "IsAttributedMactMmpClickInstall": [
        "AdsAttributedMactMmpClickInstall",
    ],
    "IsAttributedMactMmpViewInstall": [
        "AdsAttributedMactMmpViewInstall",
    ],
    "IsPurchaseConversion": [
        "AdsPurchaseConversion",
    ],
    "IsAddToCartConversion": [
        "AdsAddToCartConversion",
    ],
    "IsCheckoutInitiatedConversion": [
        "AdsCheckoutInitiatedConversion",
    ],
    "IsSignupConversion": [
        "AdsSignUpConversion",
    ],
    "IsLeadgenConversion": [
        "AdsLeadgenConversion",
    ],
    "IsSiteVisitConversion": [
        "AdsSiteVisitConversion",
    ],
    "IsWebConversion": [
        "AdsWebConversion",
    ],
    "IsSearchConversion": [
        "AdsSearchConversion",
    ],
    "IsCustomConversion": [
        "AdsCustomConversion",
    ],
    "IsContentViewConversion": [
        "AdsContentViewConversion",
    ],
    "IsAttributedKeyClickConversionDelayed": [
        "AdsAttributedKeyClickConversionDelayed",
    ],
    "IsAttributedClickConversionDelayed": [
        "AdsAttributedClickConversionDelayed",
    ],
    "IsPurchaseConversionDelayed": [
        "AdsPurchaseConversionDelayed",
    ],
    "IsMidFunnelConversionDelayed": [
        "AdsMidFunnelConversionDelayed",
    ],
    "IsAddToCartConversionDelayed": [
        "AdsAddToCartConversionDelayed",
    ],
    "IsUpperFunnelConversionDelayed": [
        "AdsUpperFunnelConversionDelayed",
    ],
    "IsWebConversionDelayed": [
        "AdsWebConversionDelayed",
    ],
    "IsSearchConversionDelayed": [
        "AdsSearchConversionDelayed",
    ],
    "IsSignupConversionDelayed": [
        "AdsSignUpConversionDelayed",
    ],
    "IsCheckoutInitiatedConversionDelayed": [
        "AdsCheckoutInitiatedConversionDelayed",
    ],
    "IsAttributedKeyViewConversion": [
        "AdsAttributedKeyViewConversion",
    ],
    "IsAttributedViewConversion": [
        "AdsAttributedViewConversion",
    ],
    "IsPurchaseConversionViewThrough": [
        "AdsPurchaseConversionViewThrough",
    ],
    "IsAddToCartConversionViewThrough": [
        "AdsAddToCartConversionViewThrough",
    ],
    "IsCheckoutInitiatedConversionViewThrough": [
        "AdsCheckoutInitiatedConversionViewThrough",
    ],
    "IsSignUpConversionViewThrough": [
        "AdsSignUpConversionViewThrough",
    ],
    "IsSiteVisitConversionViewThrough": [
        "AdsSiteVisitConversionViewThrough",
    ],
    "IsSessionConversionViewThrough": [
        "AdsSessionConversionViewThrough",
    ],
    "IsLandingPageViewConversionViewThrough": [
        "AdsLandingPageViewConversionViewThrough",
    ],
    "IsUpperFunnelConversionViewThrough": [
        "AdsUpperFunnelConversionViewThrough",
    ],
    "IsAttributedKeyViewConversionDelayed": [
        "AdsAttributedKeyViewConversionDelayed",
    ],
    "IsAttributedViewConversionDelayed": [
        "AdsAttributedViewConversionDelayed",
    ],
    "IsPurchaseConversionViewThroughDelayed": [
        "AdsPurchaseConversionViewThroughDelayed",
    ],
    "IsAddToCartConversionViewThroughDelayed": [
        "AdsAddToCartConversionViewThroughDelayed",
    ],
    "IsCheckoutInitiatedConversionViewThroughDelayed": [
        "AdsCheckoutInitiatedConversionViewThroughDelayed",
    ],
    "IsSignUpConversionViewThroughDelayed": [
        "AdsSignUpConversionViewThroughDelayed",
    ],
}

ads_slim_engagement_to_action_types = {
    "IsOpenLink": [
        "ClientTweetOpenLink",
    ],
    "IsMmpClick": [
        "AdsMmpClick",
    ],
    "IsBrandEngagement": [
        "BrandEngagement",
    ],
    "IsExternalLinkLongDwelled": [
        "ClientTweetExternalLinkLongDwelled",
    ],
    "IsVideoQualityViewed": [
        "ClientTweetVideoQualityView",
    ],
    "IsNotInterestedIn": [
        "ClientTweetNotInterestedIn",
    ],
    "IsBlockAuthor": [
        "ClientTweetBlockAuthor",
    ],
    "IsReported": [
        "ClientTweetReport",
    ],
    "IsMuteAuthor": [
        "ClientTweetMuteAuthor",
    ],
    **{
        eng: actions
        for eng, actions in ads_p_conv_click_engagement_to_action_types.items()
        if eng not in primary_engagement_to_action_types
    },
}

_REPLY_KEEP_ENGAGEMENTS = {
    "IsFavorited",
    "IsClicked",
    "IsRecapDwelled",
    "IsRecapNotDwelled",
    "IsVideoQualityViewed",
    "IsProfileClicked",
    "IsOpenLink",
    "IsReplied",
    "IsRetweeted",
}
reply_engagement_to_action_types = {
    eng: actions
    for eng, actions in primary_engagement_to_action_types.items()
    if eng in _REPLY_KEEP_ENGAGEMENTS
}
reply_engagement_to_action_types["IsNegativeFeedback"] = [
    "ClientTweetBlockAuthor",
    "ClientTweetMuteAuthor",
    "ClientTweetReport",
    "ServerTweetReport",
]


eng_to_action_types = {
    "default": primary_engagement_to_action_types,
    "all": engagement_to_action_types,
    "notifications": notification_engagement_to_action_types,
    "search": search_engagement_to_action_types,
    "ads_conversion": ads_conversion_engagement_to_action_types,
    "ads_p_conv_click": ads_p_conv_click_engagement_to_action_types,
    "ads_slim": ads_slim_engagement_to_action_types,
    "reply": reply_engagement_to_action_types,
    "none": {},
}


def engagement_to_ids(metric_group):
    return {
        eng: [action_type_map[name] for name in names]
        for eng, names in eng_to_action_types[metric_group].items()
    }


CLICK_ACTION_INDEX = recsys_pb2.ActionName.CLIENT_TWEET_OPEN_LINK
MMP_CLICK_ACTION_INDEX = recsys_pb2.ActionName.ADS_MMP_CLICK


VIDEO_CREATIVE_BIT = 1 << 30


class LineItemObjective(enum.IntEnum):
    OBJECTIVE_UNSPECIFIED = 0
    APP_INSTALLS = 1
    FOLLOWERS = 2
    BRAND_ENGAGEMENTS = 3
    VIDEO_VIEWS = 4
    WEBSITE_CLICKS = 5
    APP_ENGAGEMENTS = 8
    PREROLL_VIEWS = 9
    REACH = 11
    MIDROLL_VIEWS = 12
    TAKEOVER = 13


CLICK_CONDITIONED_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_CLICK_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_CLICK_CONVERSION,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION,
    recsys_pb2.ActionName.ADS_MID_FUNNEL_CONVERSION,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION,
    recsys_pb2.ActionName.ADS_UPPER_FUNNEL_CONVERSION,
    recsys_pb2.ActionName.ADS_WEB_CONVERSION,
    recsys_pb2.ActionName.ADS_SEARCH_CONVERSION,
    recsys_pb2.ActionName.ADS_CUSTOM_CONVERSION,
    recsys_pb2.ActionName.ADS_CONTENT_VIEW_CONVERSION,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_CLICK_INSTALL,
]

FRESH_CLICK_CONDITIONED_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_LEADGEN_CONVERSION,
]

MMP_CLICK_CONDITIONED_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_PURCHASE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_ADD_TO_CART,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_LEVEL_ACHIEVED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_TUTORIAL_COMPLETE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_SIGN_UP,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_CUSTOM,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_MMP_CLICK_INSTALL,
]
MMP_CLICK_VIEW_THROUGH_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_PURCHASE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_ADD_TO_CART,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_LEVEL_ACHIEVED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_TUTORIAL_COMPLETE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_SIGN_UP,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_CUSTOM,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_MMP_VIEW_INSTALL,
]

CONVERSION_KEEP_WEB = 1 << 0
CONVERSION_KEEP_APP = 1 << 1

CONVERSION_DELAY_NONE = -1
CONVERSION_DELAY_HEAD_COLUMNS: dict[str, str] = {
    "IsAttributedClickConversion": "conversionDelayMsSeq",
    "IsAttributedKeyClickConversion": "conversionDelayMsSeq_KEY",
    "IsPurchaseConversion": "conversionDelayMsSeq_PURCHASE",
    "IsSignupConversion": "conversionDelayMsSeq_SIGN_UP",
    "IsCustomConversion": "conversionDelayMsSeq_CUSTOM",
    "IsAddToCartConversion": "conversionDelayMsSeq_ADD_TO_CART",
    "IsCheckoutInitiatedConversion": "conversionDelayMsSeq_CHECKOUT_INITIATED",
}
CONVERSION_DELAY_COLUMNS: tuple[str, ...] = tuple(CONVERSION_DELAY_HEAD_COLUMNS.values())

MACT_IN_APP_LOSS_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_PURCHASE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_ADD_TO_CART,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_LEVEL_ACHIEVED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_TUTORIAL_COMPLETE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_SIGN_UP,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_CUSTOM,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_PURCHASE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_ADD_TO_CART,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_LEVEL_ACHIEVED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_TUTORIAL_COMPLETE,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_SIGN_UP,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_MACT_VIEW_CUSTOM,
]

VIEW_LOSS_ACTION_INDICES = [
    recsys_pb2.ActionName.CLIENT_TWEET_RECAP_DWELLED,
    recsys_pb2.ActionName.CLIENT_TWEET_RECAP_NOT_DWELLED,
    recsys_pb2.ActionName.CLIENT_TWEET_VIDEO_QUALITY_VIEW,
]

VIEW_THROUGH_ACTION_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_VIEW_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_VIEW_CONVERSION,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_SITE_VISIT_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_SESSION_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_LANDING_PAGE_VIEW_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_UPPER_FUNNEL_CONVERSION_VIEW_THROUGH,
]

PURCHASE_VALUE_ACTION_INDEX = recsys_pb2.ContinuousActionName.ADS_WEB_CT_PURCHASE_VALUE

NEGATIVE_FEEDBACK_HEAD_INDICES = [
    recsys_pb2.ActionName.CLIENT_TWEET_REPORT,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_INTERESTED_IN,
    recsys_pb2.ActionName.CLIENT_TWEET_SEE_FEWER,
    recsys_pb2.ActionName.CLIENT_TWEET_MUTE_AUTHOR,
    recsys_pb2.ActionName.CLIENT_TWEET_BLOCK_AUTHOR,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_RELEVANT,
    recsys_pb2.ActionName.CLIENT_TWEET_MUTE_CONVERSATION,
    recsys_pb2.ActionName.CLIENT_NOTIFICATION_SEE_LESS_OFTEN,
    recsys_pb2.ActionName.CLIENT_TWEET_SEARCH_QUERY_REFORMULATED,
]
