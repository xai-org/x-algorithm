# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import numpy as np
import numpy.typing as npt

from xai_proto import recsys_pb2

ADS_LATE_WINDOW_TWIN_HEAD_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_CLICK_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_CLICK_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_MID_FUNNEL_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_UPPER_FUNNEL_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_WEB_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_SEARCH_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_DELAYED,
]

ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES = [
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_VIEW_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_VIEW_CONVERSION_DELAYED,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_VIEW_THROUGH_DELAYED,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_VIEW_THROUGH_DELAYED,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_VIEW_THROUGH_DELAYED,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_VIEW_THROUGH_DELAYED,
]

ADS_LATE_WINDOW_TWIN_SUPPRESSOR = {
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_CLICK_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_CLICK_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_CLICK_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ATTRIBUTED_CLICK_CONVERSION,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION,
    recsys_pb2.ActionName.ADS_MID_FUNNEL_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_MID_FUNNEL_CONVERSION,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION,
    recsys_pb2.ActionName.ADS_UPPER_FUNNEL_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_UPPER_FUNNEL_CONVERSION,
    recsys_pb2.ActionName.ADS_WEB_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ATTRIBUTED_CLICK_CONVERSION,
    recsys_pb2.ActionName.ADS_SEARCH_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_SEARCH_CONVERSION,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_VIEW_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ATTRIBUTED_KEY_VIEW_CONVERSION,
    recsys_pb2.ActionName.ADS_ATTRIBUTED_VIEW_CONVERSION_DELAYED: recsys_pb2.ActionName.ADS_ATTRIBUTED_VIEW_CONVERSION,
    recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_VIEW_THROUGH_DELAYED: recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_VIEW_THROUGH_DELAYED: recsys_pb2.ActionName.ADS_ADD_TO_CART_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_VIEW_THROUGH_DELAYED: recsys_pb2.ActionName.ADS_CHECKOUT_INITIATED_CONVERSION_VIEW_THROUGH,
    recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_VIEW_THROUGH_DELAYED: recsys_pb2.ActionName.ADS_SIGN_UP_CONVERSION_VIEW_THROUGH,
}

LATE_WINDOW_STREAM_ID = 3

ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART = {
    recsys_pb2.ActionName.ADS_OPEN_LINK_LATE_WINDOW: recsys_pb2.ActionName.CLIENT_TWEET_OPEN_LINK,
    recsys_pb2.ActionName.ADS_LONG_DWELL_LATE_WINDOW: recsys_pb2.ActionName.CLIENT_TWEET_EXTERNAL_LINK_LONG_DWELLED,
}
ADS_LATE_ENGAGEMENT_TWIN_HEAD_INDICES = list(ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART)
ADS_LATE_ENGAGEMENT_COUNTERPART_INDICES = list(ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART.values())

EARLY_ACTIONS_COLUMN = "earlyActionNameMultiHotSeqSeq"


def apply_late_engagement_labels(
    actions: npt.NDArray[np.bool_],
    early_counterparts: npt.NDArray[np.bool_] | None,
    sample_source: npt.NDArray[np.integer],
    new_event_mask: npt.NDArray[np.bool_],
) -> None:
    rows = actions.shape[0]
    late = (np.asarray(sample_source).reshape(rows) == LATE_WINDOW_STREAM_ID)[:, None]
    late = late & new_event_mask.astype(np.bool_, copy=False)
    twins = ADS_LATE_ENGAGEMENT_TWIN_HEAD_INDICES
    if early_counterparts is None or not late.any():
        actions[:, :, twins] = False
        return
    for k, (twin, counterpart) in enumerate(ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART.items()):
        actions[:, :, twin] = late & actions[:, :, counterpart] & ~early_counterparts[:, :, k]


def late_engagement_probability(raw):
    b = raw.clip(0.0, 0.5)
    return b / (1.0 - b)


def compose_late_engagement(p_early, raw_late):
    return p_early + (1.0 - p_early) * late_engagement_probability(raw_late)


def compose_late_engagement_log_probs(log_probs):
    import jax.numpy as jnp

    for twin, counterpart in ADS_LATE_ENGAGEMENT_TWIN_COUNTERPART.items():
        composed = compose_late_engagement(
            jnp.exp(log_probs[..., counterpart]), jnp.exp(log_probs[..., twin])
        )
        log_probs = log_probs.at[..., counterpart].set(
            jnp.log(jnp.clip(composed, 0.0, 1.0 - 1e-6)).astype(log_probs.dtype)
        )
    return log_probs
