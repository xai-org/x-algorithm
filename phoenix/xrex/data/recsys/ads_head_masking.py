# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from xrex.data.recsys.ads_late_window import (
    ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART,
    ADS_LATE_ENGAGEMENT_TWIN_HEAD_INDICES,
    ADS_LATE_WINDOW_TWIN_HEAD_INDICES,
    ADS_LATE_WINDOW_TWIN_SUPPRESSOR,
    ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES,
    LATE_WINDOW_STREAM_ID,
)
from xrex.data.recsys.constants import (
    CLICK_ACTION_INDEX,
    CLICK_CONDITIONED_ACTION_INDICES,
    CONVERSION_KEEP_APP,
    CONVERSION_KEEP_WEB,
    FRESH_CLICK_CONDITIONED_ACTION_INDICES,
    MMP_CLICK_ACTION_INDEX,
    MMP_CLICK_CONDITIONED_ACTION_INDICES,
    MMP_CLICK_VIEW_THROUGH_ACTION_INDICES,
    VIEW_THROUGH_ACTION_INDICES,
)

FRESH_STREAM_ID = 0
EARLY_RELABEL_STREAM_ID = 1
LATE_SPLIT_STREAM_ID = 2

ADS_FAMILY_SLICE_HEADS: dict[str, int] = {
    "fresh_website_clicked": FRESH_CLICK_CONDITIONED_ACTION_INDICES[0],
    "delayed_website_clicked": CLICK_CONDITIONED_ACTION_INDICES[0],
    "delayed_website_non_clicked": VIEW_THROUGH_ACTION_INDICES[0],
    "delayed_app_clicked": MMP_CLICK_CONDITIONED_ACTION_INDICES[0],
    "delayed_app_non_clicked": MMP_CLICK_VIEW_THROUGH_ACTION_INDICES[0],
}

ADS_FAMILY_METRIC_SLICES: tuple[tuple[Sequence[int], tuple[str, ...]], ...] = (
    (FRESH_CLICK_CONDITIONED_ACTION_INDICES, ("fresh_website_clicked",)),
    (CLICK_CONDITIONED_ACTION_INDICES, ("delayed_website_clicked",)),
    (VIEW_THROUGH_ACTION_INDICES, ("delayed_website_non_clicked",)),
    (MMP_CLICK_CONDITIONED_ACTION_INDICES, ("delayed_app_clicked",)),
    (MMP_CLICK_VIEW_THROUGH_ACTION_INDICES, ("delayed_app_non_clicked",)),
    (ADS_LATE_WINDOW_TWIN_HEAD_INDICES, ("late_split_no_early",)),
    (ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES, ("late_split_non_clicked_no_early",)),
)
ADS_ENGAGEMENT_METRIC_SLICES = ("fresh", "fresh_home_website_clicks", "dpa")


def head_metric_slices(action_ids: Sequence[int]) -> tuple[str, ...] | None:
    ids = set(action_ids)
    for family, slices in ADS_FAMILY_METRIC_SLICES:
        if ids <= set(family):
            return slices
    if all(ids.isdisjoint(family) for family, _ in ADS_FAMILY_METRIC_SLICES):
        return ADS_ENGAGEMENT_METRIC_SLICES
    return None


def metric_head_mask_keys(
    eng_to_ids: dict[str, list[int]], mask_keys: Sequence[str], *, own_slice: bool
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for eng_name, ids in eng_to_ids.items():
        own = head_metric_slices(ids) if own_slice else None
        out[eng_name] = [m for m in mask_keys if own is None or m in own]
    return out


def _head_indicator(indices: Sequence[int], num_actions: int) -> npt.NDArray[np.bool_]:
    indicator = np.zeros(num_actions, dtype=np.bool_)
    indicator[list(indices)] = True
    return indicator


def build_trained_candidate_mask(
    actions: npt.NDArray[np.bool_],
    sample_source: npt.NDArray[np.integer] | None,
    trained_actions_bits: npt.NDArray[np.uint8],
    *,
    ads_head_masking: bool,
    ads_late_window_heads: bool = False,
) -> npt.NDArray[np.bool_]:
    rows, _, num_actions = actions.shape
    keep_web = ((trained_actions_bits & CONVERSION_KEEP_WEB) != 0)[:, :, None]
    keep_app = ((trained_actions_bits & CONVERSION_KEEP_APP) != 0)[:, :, None]

    if not ads_head_masking:
        return np.broadcast_to(keep_web | keep_app, actions.shape).copy()

    if sample_source is None:
        source = np.full((rows, 1, 1), FRESH_STREAM_ID, dtype=np.int64)
    else:
        source = np.asarray(sample_source).reshape(rows, 1, 1)
    fresh = source == FRESH_STREAM_ID
    late = source == LATE_SPLIT_STREAM_ID
    late_window = source == LATE_WINDOW_STREAM_ID
    early = ~fresh & ~late & ~late_window

    actions = actions.astype(np.bool_, copy=False)
    click = actions[:, :, CLICK_ACTION_INDEX][:, :, None]
    app_click = actions[:, :, MMP_CLICK_ACTION_INDEX][:, :, None]

    fresh_ct = _head_indicator(FRESH_CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    web_ct = _head_indicator(CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    web_vt = _head_indicator(VIEW_THROUGH_ACTION_INDICES, num_actions)
    app_ct = _head_indicator(MMP_CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    app_vt = _head_indicator(MMP_CLICK_VIEW_THROUGH_ACTION_INDICES, num_actions)
    twins = _head_indicator(ADS_LATE_WINDOW_TWIN_HEAD_INDICES, num_actions)
    vt_twins = _head_indicator(ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES, num_actions)
    late_eng = _head_indicator(
        ADS_LATE_ENGAGEMENT_TWIN_HEAD_INDICES if ads_late_window_heads else [], num_actions
    )
    engagement = ~(fresh_ct | web_ct | web_vt | app_ct | app_vt | twins | vt_twins | late_eng)

    suppressor_index = list(range(num_actions))
    for twin, suppressor in ADS_LATE_WINDOW_TWIN_SUPPRESSOR.items():
        suppressor_index[twin] = suppressor
    if ads_late_window_heads:
        for twin, counterpart in ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART.items():
            suppressor_index[twin] = counterpart
    no_early_counterpart = ~actions[:, :, suppressor_index]

    kept = keep_web | keep_app
    return (
        (engagement & fresh & kept)
        | (fresh_ct & fresh & kept & click)
        | (web_ct & early & keep_web & click)
        | (web_vt & early & keep_web & ~click)
        | (app_ct & early & keep_app & app_click)
        | (app_vt & early & keep_app & ~app_click)
        | (twins & late & click & no_early_counterpart & kept)
        | (vt_twins & late & keep_web & ~click & no_early_counterpart)
        | (late_eng & fresh & kept & no_early_counterpart)
        | (late_eng & late_window & actions)
    )
