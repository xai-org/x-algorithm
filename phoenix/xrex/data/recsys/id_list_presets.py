# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

from .id_list_features import feature_name_for, identity_for


@dataclasses.dataclass(frozen=True)
class IdListEntry:
    source: str
    k: int

    @property
    def identity(self) -> str:
        return identity_for(self.source)

    @property
    def feature_name(self) -> str:
        return feature_name_for(self.source, self.k)

    @property
    def spec(self) -> str:
        return f"{self.source}:{self.k}"


def _entries(*items: tuple[str, int]) -> tuple[IdListEntry, ...]:
    return tuple(IdListEntry(source, k) for source, k in items)


HYDRATOR_ENGAGEMENT_LISTS: tuple[IdListEntry, ...] = _entries(
    ("top_p_50_2d_VIEW", 50),
    ("top_p_50_7d_LINGER_VIEW", 50),
    ("top_p_50_28d_LINGER_VIEW", 50),
    ("top_au_25_28d_POST_CLICK", 25),
    ("latest_p_30_1d_CLIENT_TWEET_RECAP_DWELLED", 30),
    ("latest_p_65_7d_CLIENT_TWEET_RECAP_DWELLED", 65),
    ("latest_p_80_28d_CLIENT_TWEET_RECAP_DWELLED", 80),
    ("latest_p_5_1d_CLIENT_TWEET_CLICK", 5),
    ("latest_p_5_2d_CLIENT_TWEET_CLICK", 5),
    ("latest_p_10_7d_CLIENT_TWEET_CLICK", 10),
    ("latest_p_25_28d_CLIENT_TWEET_CLICK", 25),
    ("latest_p_5_4hr_CLIENT_TWEET_OPEN_LINK", 5),
    ("latest_p_5_1d_CLIENT_TWEET_OPEN_LINK", 5),
    ("latest_p_5_7d_CLIENT_TWEET_OPEN_LINK", 5),
    ("latest_p_10_90d_CLIENT_TWEET_OPEN_LINK", 10),
    ("latest_p_15_7d_ENGAGE", 15),
    ("latest_p_30_28d_ENGAGE", 30),
    ("latest_p_50_180d_ENGAGE", 50),
    ("latest_p_15_7d_CLIENT_TWEET_VIDEO_QUALITY_VIEW", 15),
    ("latest_p_30_28d_CLIENT_TWEET_VIDEO_QUALITY_VIEW", 30),
    ("latest_p_5_7d_CLIENT_EXTERNAL_LINK_SESSION", 5),
    ("latest_p_10_90d_CLIENT_EXTERNAL_LINK_SESSION", 10),
    ("latest_p_5_28d_DPA_CLICK", 5),
    ("latest_au_5_4hr_CLIENT_TWEET_OPEN_LINK", 5),
    ("latest_au_65_180d_CLIENT_TWEET_CLICK", 65),
    ("latest_au_30_7d_ENGAGE", 30),
    ("latest_au_50_28d_ENGAGE", 50),
    ("latest_au_10_7d_BRAND_ENGAGE", 10),
    ("latest_au_15_28d_BRAND_ENGAGE", 15),
    ("latest_au_5_28d_DPA_CLICK", 5),
    ("latest_au_5_90d_CLIENT_TWEET_TAKE_SCREENSHOT", 5),
)
HYDRATOR_LINK_DWELL_LISTS: tuple[IdListEntry, ...] = _entries(
    ("latest_p_5_7d_CLIENT_EXTERNAL_LINK_SESSION_MORE_THAN_5_SEC", 5),
    ("latest_p_5_90d_CLIENT_EXTERNAL_LINK_SESSION_MORE_THAN_5_SEC", 5),
    ("latest_au_5_7d_CLIENT_EXTERNAL_LINK_SESSION_MORE_THAN_5_SEC", 5),
    ("latest_au_10_90d_CLIENT_EXTERNAL_LINK_SESSION_MORE_THAN_5_SEC", 10),
)
HYDRATOR_CONVERSION_LISTS: tuple[IdListEntry, ...] = _entries(
    ("top_cp_10_28d_CONVERSION_ALL_CLICK_THROUGH", 10),
    ("top_cp_25_180d_CONVERSION_ALL_CLICK_THROUGH", 25),
    ("top_cp_5_180d_CONVERSION_PURCHASE_CLICK_THROUGH", 5),
    ("top_cau_15_28d_CONVERSION_SESSION_CLICK_THROUGH", 15),
    ("top_cau_65_180d_CONVERSION_SESSION_CLICK_THROUGH", 65),
    ("top_cau_10_28d_CONVERSION_LANDING_PAGE_CLICK_THROUGH", 10),
    ("top_cau_25_180d_CONVERSION_LANDING_PAGE_CLICK_THROUGH", 25),
    ("top_cau_10_28d_CONVERSION_LANDING_PAGE_VIEW_CLICK_THROUGH", 10),
    ("top_cau_25_180d_CONVERSION_LANDING_PAGE_VIEW_CLICK_THROUGH", 25),
    ("top_cau_5_28d_CONVERSION_PURCHASE_CLICK_THROUGH", 5),
    ("top_cau_10_180d_CONVERSION_PURCHASE_CLICK_THROUGH", 10),
    ("top_cau_10_180d_CONVERSION_CONTENT_VIEW_CLICK_THROUGH", 10),
    ("top_cau_5_180d_CONVERSION_CUSTOM_CLICK_THROUGH", 5),
    ("top_cau_5_180d_CONVERSION_SIGN_UP_CLICK_THROUGH", 5),
    ("top_cau_5_180d_CONVERSION_ADD_TO_CART_CLICK_THROUGH", 5),
    ("top_cau_15_7d_CONVERSION_SESSION_CLICK_THROUGH", 15),
    ("top_cau_5_180d_CONVERSION_OTHER_CLICK_THROUGH", 5),
)
HYDRATOR_LISTS: tuple[IdListEntry, ...] = (
    HYDRATOR_ENGAGEMENT_LISTS + HYDRATOR_LINK_DWELL_LISTS + HYDRATOR_CONVERSION_LISTS
)

HISTORY_AUTHOR_LISTS: tuple[IdListEntry, ...] = _entries(
    ("hist_top_au_30_1d_RECAP_DWELLED", 30),
    ("hist_top_au_30_28d_RECAP_DWELLED", 30),
    ("hist_top_au_50_1d_VIDEO_QUALITY_VIEW", 50),
    ("hist_top_au_50_28d_VIDEO_QUALITY_VIEW", 50),
    ("hist_top_au_40_1d_CLICK", 40),
    ("hist_top_au_60_28d_CLICK", 60),
    ("hist_top_au_25_1d_PHOTO_EXPAND", 25),
    ("hist_top_au_35_28d_PHOTO_EXPAND", 35),
    ("hist_top_au_15_1d_SHOW_MORE_EXPAND", 15),
    ("hist_top_au_20_28d_SHOW_MORE_EXPAND", 20),
    ("hist_top_au_10_7d_CLICK_PROFILE", 10),
    ("hist_top_au_15_28d_CLICK_PROFILE", 15),
    ("hist_top_au_40_7d_FAV", 40),
    ("hist_top_au_20_7d_OPEN_LINK", 20),
    ("hist_top_au_55_90d_OPEN_LINK", 55),
    ("hist_top_au_15_7d_EXTERNAL_LINK_LONG_DWELLED", 15),
    ("hist_top_au_30_90d_EXTERNAL_LINK_LONG_DWELLED", 30),
    ("hist_top_au_5_7d_TAKE_SCREENSHOT", 5),
    ("hist_top_au_10_90d_TAKE_SCREENSHOT", 10),
    ("hist_top_au_30_90d_BOOKMARK", 30),
    ("hist_top_au_30_90d_RETWEET", 30),
    ("hist_top_au_5_90d_REPLY", 5),
    ("hist_top_au_5_90d_SHARE", 5),
    ("hist_top_au_5_180d_FOLLOW_AUTHOR", 5),
)

HISTORY_CONVERSION_LISTS: tuple[IdListEntry, ...] = _entries(
    ("hist_top_cau_25_28d_ADS_SESSION_CONVERSION", 25),
    ("hist_top_cau_65_180d_ADS_SESSION_CONVERSION", 65),
    ("hist_top_cau_15_28d_ADS_SITE_VISIT_CONVERSION", 15),
    ("hist_top_cau_25_180d_ADS_SITE_VISIT_CONVERSION", 25),
    ("hist_top_cau_15_28d_ADS_LONG_SITE_DWELL_CONVERSION", 15),
    ("hist_top_cau_25_90d_ADS_LONG_SITE_DWELL_CONVERSION", 25),
    ("hist_top_cau_10_28d_ADS_LANDING_PAGE_VIEW_CONVERSION", 10),
    ("hist_top_cau_25_180d_ADS_LANDING_PAGE_VIEW_CONVERSION", 25),
    ("hist_top_cau_5_28d_ADS_PURCHASE_CONVERSION", 5),
    ("hist_top_cau_10_180d_ADS_PURCHASE_CONVERSION", 10),
    ("hist_top_cau_10_180d_ADS_CONTENT_VIEW_CONVERSION", 10),
    ("hist_top_cau_10_180d_CUSTOM_SIGNUP_ADDTOCART", 10),
    ("hist_top_cau_5_180d_OTHER_CONVERSION", 5),
    ("hist_latest_cau_15_ADS_SESSION_CONVERSION", 15),
    ("hist_top_cp_20_28d_ANY_CONVERSION", 20),
    ("hist_top_cp_30_180d_ANY_CONVERSION", 30),
    ("hist_top_cp_5_180d_ADS_PURCHASE_CONVERSION", 5),
)

HISTORY_ORGANIC_POST_LISTS: tuple[IdListEntry, ...] = _entries(
    ("hist_top_op_80_1d_ENGAGE", 80),
    ("hist_latest_op_50_ENGAGE", 50),
    ("hist_top_op_40_1d_FAV", 40),
    ("hist_latest_op_30_FAV", 30),
    ("hist_latest_op_50_RESHARE", 50),
)

HISTORY_LISTS: tuple[IdListEntry, ...] = (
    HISTORY_AUTHOR_LISTS + HISTORY_CONVERSION_LISTS + HISTORY_ORGANIC_POST_LISTS
)
OFFLINE_LISTS: tuple[IdListEntry, ...] = HYDRATOR_ENGAGEMENT_LISTS + HISTORY_LISTS
ALL_LISTS: tuple[IdListEntry, ...] = HYDRATOR_LISTS + HISTORY_LISTS


def with_k(entries: Iterable[IdListEntry], overrides: Mapping[str, int]) -> tuple[IdListEntry, ...]:
    entries = tuple(entries)
    known = {e.identity for e in entries}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise KeyError(f"no id list with identity {unknown}; known: {sorted(known)}")
    return tuple(dataclasses.replace(e, k=overrides.get(e.identity, e.k)) for e in entries)


def spec_string(entries: Iterable[IdListEntry]) -> str:
    return ",".join(e.spec for e in entries)


ADS_ID_LIST_FEATURES_HYD = spec_string(HYDRATOR_LISTS)
ADS_ID_LIST_FEATURES_HIST = spec_string(HISTORY_LISTS)
ADS_ID_LIST_FEATURES_ALL = spec_string(OFFLINE_LISTS)
ADS_ID_LIST_FEATURES_ALL_LIVE = spec_string(ALL_LISTS)
