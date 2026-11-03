# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import enum


class CategoricalFeature(enum.IntEnum):
    productSurfaceSeq = 0
    timezoneSeq = 5
    localHourOfDaySeq = 6
    localDayOfWeekSeq = 7
    postAgeBucketSeq = 8
    favCountBucketSeq = 9
    replyCountBucketSeq = 10
    repostCountBucketSeq = 11
    quoteCountBucketSeq = 12
    viewCountBucketSeq = 13
    authorIsNsfwSeq = 14
    webConvTrackingIntegrationSeq = 15
    exactPhraseSeq = 16
    matchedWordFractionBucketSeq = 17


COMPUTED_CATEGORICAL_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "localHourOfDaySeq",
        "localDayOfWeekSeq",
        "postAgeBucketSeq",
        "favCountBucketSeq",
        "replyCountBucketSeq",
        "repostCountBucketSeq",
        "quoteCountBucketSeq",
        "viewCountBucketSeq",
        "authorIsNsfwSeq",
        "matchedWordFractionBucketSeq",
    }
)

COMPUTED_BOOL_FEATURE_NAMES: frozenset[str] = frozenset({"isStalePost14d"})

AUTHOR_NSFW_BIT = 2


class BoolFeature(enum.IntEnum):
    isStalePost14d = 0
    isAuthorFollowedByViewerSeq = 1
    isAuthorFollowingViewerSeq = 2


WEB_CONV_TRACKING_INTEGRATION_CARDINALITY = 8


class FloatFeature(enum.IntEnum):
    matchedWordFractionSeq = 1


class Int64Feature(enum.IntEnum):
    clientAppIdSeq = 0
    promotedIdSeq = 1
    favCountSeq = 5
    replyCountSeq = 6
    repostCountSeq = 7
    quoteCountSeq = 8
    viewCountSeq = 9
    ipAddressSeq = 10
    firstDpaProductKey = 11
    firstDpaProductKeyHash2 = 12
    webConvTimeOnSiteInferredMsSeq = 13
    webConvTimeOnSiteMeasuredMsSeq = 14
    requestTimeSecSeq = 15
    latestHistoryImpressionTimeSecSeq = 16


COMPUTED_INT64_FEATURE_NAMES: frozenset[str] = frozenset(
    {"firstDpaProductKeyHash2", "requestTimeSecSeq", "latestHistoryImpressionTimeSecSeq"}
)


ADS_PRODUCT_KEY_TABLE_SIZE = 10_000_000
ADS_PRODUCT_KEY_HASH_SCALE = 1_566_083_941
ADS_PRODUCT_KEY_HASH_BIAS = 774_047_449
ADS_PRODUCT_KEY_HASH_SCALE_2 = 1_204_318_477
ADS_PRODUCT_KEY_HASH_BIAS_2 = 393_342_739
ADS_PRODUCT_KEY_HASH_MODULUS = 2_147_483_647

STALE_POST_14D_TTL_SEC = 1_213_200
STALE_POST_30D_TTL_SEC = 2_595_600

ADS_COUNT_FAMILIES: tuple[str, ...] = (
    "View",
    "Click",
    "LongDwellClick",
    "VideoOpen",
    "LongVideoView",
    "Fav",
    "NotInterested",
)
ADS_COUNT_WINDOWS: tuple[str, ...] = ("1h", "3h", "12h", "24h", "3d", "7d", "14d", "28d")
NUM_ADS_COUNT_FEATURES = 168
ADS_COUNT_BUCKETS = 33


CATEGORICAL_FEATURES: list[str] = [f.name for f in CategoricalFeature]
BOOL_FEATURES: list[str] = [f.name for f in BoolFeature]
FLOAT_FEATURES: list[str] = [f.name for f in FloatFeature]
INT64_FEATURES: list[str] = [f.name for f in Int64Feature]

ENGAGEMENT_COUNT_BUCKET_MAP: tuple[tuple[CategoricalFeature, Int64Feature], ...] = (
    (CategoricalFeature.favCountBucketSeq, Int64Feature.favCountSeq),
    (CategoricalFeature.replyCountBucketSeq, Int64Feature.replyCountSeq),
    (CategoricalFeature.repostCountBucketSeq, Int64Feature.repostCountSeq),
    (CategoricalFeature.quoteCountBucketSeq, Int64Feature.quoteCountSeq),
    (CategoricalFeature.viewCountBucketSeq, Int64Feature.viewCountSeq),
)


class UserCategoricalFeature(enum.IntEnum):
    userLanguageCode = 0
    userCountryCode = 1
    userState = 2
    userGender = 3
    userAgeBracket = 4
    userDmaCode = 5
    userInferredGender = 6


class UserBoolFeature(enum.IntEnum):
    pass


class UserFloatFeature(enum.IntEnum):
    userLongitude = 0
    userLatitude = 1
    userInferredGenderScore = 2
    userAgeInYears = 3


class UserInt64Feature(enum.IntEnum):
    pass


USER_CATEGORICAL_FEATURES: list[str] = [f.name for f in UserCategoricalFeature]
USER_BOOL_FEATURES: list[str] = [f.name for f in UserBoolFeature]
USER_FLOAT_FEATURES: list[str] = [f.name for f in UserFloatFeature]
USER_INT64_FEATURES: list[str] = [f.name for f in UserInt64Feature]

BASE_REQUIRED_COLUMNS: list[str] = [
    "length",
    "userId",
    "tweetIdSeq",
    "authorIdSeq",
    "promotedIdSeq",
    "impressedTimeMsSeq",
    "clientAppIdSeq",
    "actionNameMultiHotSeqSeq",
    "continuousActionValuesSeqSeq",
    "paddingMask",
    "newEventMask",
    "installedAppsMultiHot",
]


OPTIONAL_COLUMNS: list[str] = [
    "searchQueryEmbeddingSeq",
    "lineItemObjectiveSeq",
    "safetyLabelMaskSeq",
    "semanticIdSeq",
    "sampleWeight",
    "firstDpaProductKey",
    "authorFollowerCountSeq",
    "inReplyToPostIdSeq",
    "sample_source",
    "valueLabelValidSeq",
    "valueBaselineMeanUsdSeq",
    "lineItemIdSeq",
    "auxValueContentViewRatioSeq",
    "auxValueContentViewValidSeq",
    "auxValueContentViewBaselineMeanUsdSeq",
    "auxValueAddToCartRatioSeq",
    "auxValueAddToCartValidSeq",
    "auxValueAddToCartBaselineMeanUsdSeq",
    "conversionKeepBits",
    "earlyActionNameMultiHotSeqSeq",
    "conversionEligibleTypesSeq",
    "conversionDelayMsSeq",
    "conversionDelayMsSeq_KEY",
    "conversionDelayMsSeq_PURCHASE",
    "conversionDelayMsSeq_SIGN_UP",
    "conversionDelayMsSeq_CUSTOM",
    "conversionDelayMsSeq_ADD_TO_CART",
    "conversionDelayMsSeq_CHECKOUT_INITIATED",
    "webConvTimeOnSiteInferredMsSeq",
    "webConvTimeOnSiteMeasuredMsSeq",
    "webConvTrackingIntegrationSeq",
    "exactPhraseSeq",
    "matchedWordFractionSeq",
    "lexicalTextSeq",
    "lexicalQuerySeq",
    "lexicalAuthorSeq",
    "validPageRecord",
    "servedSlotSeq",
    "weightedScoreSeq",
    "finalScoreSeq",
    "rewardRerankSlotProbSeq",
    "pageDecodeSlotProbSeq",
    "servedTimeMsSeq",
]


def _build_required_columns() -> list[str]:
    excluded_names = (
        COMPUTED_CATEGORICAL_FEATURE_NAMES
        | COMPUTED_INT64_FEATURE_NAMES
        | COMPUTED_BOOL_FEATURE_NAMES
        | set(OPTIONAL_COLUMNS)
    )
    seen: set[str] = set()
    result: list[str] = []
    for col in BASE_REQUIRED_COLUMNS:
        if col not in seen:
            result.append(col)
            seen.add(col)
    for feature_list in (
        CATEGORICAL_FEATURES,
        BOOL_FEATURES,
        FLOAT_FEATURES,
        INT64_FEATURES,
        USER_CATEGORICAL_FEATURES,
        USER_BOOL_FEATURES,
        USER_FLOAT_FEATURES,
        USER_INT64_FEATURES,
    ):
        for col in feature_list:
            if col not in seen and col not in excluded_names:
                result.append(col)
                seen.add(col)
    return result


REQUIRED_COLUMNS: list[str] = _build_required_columns()
