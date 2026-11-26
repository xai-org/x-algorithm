# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence

import numba
import numpy as np

from .id_list_features import (
    IdListFeatureSpec,
    is_history_list,
    parse_history_list,
    slot_offsets,
)

EVENT_ACTIONS: dict[str, tuple[str, ...]] = {
    "FAV": ("SERVER_TWEET_FAV",),
    "REPLY": ("SERVER_TWEET_REPLY",),
    "RETWEET": ("SERVER_TWEET_RETWEET",),
    "QUOTE": ("SERVER_TWEET_QUOTE",),
    "FOLLOW_AUTHOR": ("CLIENT_TWEET_FOLLOW_AUTHOR",),
    "BOOKMARK": ("CLIENT_TWEET_BOOKMARK",),
    "TAKE_SCREENSHOT": ("CLIENT_TWEET_TAKE_SCREENSHOT",),
    "CLICK": ("CLIENT_TWEET_CLICK",),
    "CLICK_PROFILE": ("CLIENT_TWEET_CLICK_PROFILE",),
    "OPEN_LINK": ("CLIENT_TWEET_OPEN_LINK",),
    "PHOTO_EXPAND": ("CLIENT_TWEET_PHOTO_EXPAND",),
    "SHOW_MORE_EXPAND": ("CLIENT_TWEET_SHOW_MORE_EXPAND",),
    "CLICK_HASHTAG": ("CLIENT_TWEET_CLICK_HASHTAG",),
    "CLICK_MENTION": ("CLIENT_TWEET_CLICK_MENTION_SCREEN_NAME",),
    "QUOTED_TWEET_CLICK": ("CLIENT_QUOTED_TWEET_CLICK",),
    "TRANSLATE_CLICK": ("CLIENT_TWEET_TRANSLATE_CLICK",),
    "CLICK_GROK_ANALYZE": ("CLIENT_TWEET_CLICK_GROK_ANALYZE",),
    "EXTERNAL_LINK_SESSION": ("CLIENT_EXTERNAL_LINK_SESSION",),
    "EXTERNAL_LINK_LONG_DWELLED": ("CLIENT_TWEET_EXTERNAL_LINK_LONG_DWELLED",),
    "VIDEO_QUALITY_VIEW": ("CLIENT_TWEET_VIDEO_QUALITY_VIEW",),
    "RECAP_DWELLED": ("CLIENT_TWEET_RECAP_DWELLED",),
    "SHARE": (
        "CLIENT_TWEET_CLICK_SHARE",
        "CLIENT_TWEET_SHARE_VIA_COPY_LINK",
        "CLIENT_TWEET_CLICK_SEND_VIA_DIRECT_MESSAGE",
        "CLIENT_TWEET_SHARE",
    ),
    "ADS_SESSION_CONVERSION": ("ADS_SESSION_CONVERSION",),
    "ADS_SITE_VISIT_CONVERSION": ("ADS_SITE_VISIT_CONVERSION",),
    "ADS_LONG_SITE_DWELL_CONVERSION": ("ADS_LONG_SITE_DWELL_CONVERSION",),
    "ADS_LANDING_PAGE_VIEW_CONVERSION": ("ADS_LANDING_PAGE_VIEW_CONVERSION",),
    "ADS_PURCHASE_CONVERSION": ("ADS_PURCHASE_CONVERSION",),
    "ADS_CONTENT_VIEW_CONVERSION": ("ADS_CONTENT_VIEW_CONVERSION",),
    "ADS_SIGN_UP_CONVERSION": ("ADS_SIGN_UP_CONVERSION",),
    "ADS_CUSTOM_CONVERSION": ("ADS_CUSTOM_CONVERSION",),
    "ADS_ADD_TO_CART_CONVERSION": ("ADS_ADD_TO_CART_CONVERSION",),
    "ANY_CONVERSION": ("ADS_WEB_CONVERSION",),
}
EVENT_ACTIONS["CUSTOM_SIGNUP_ADDTOCART"] = (
    EVENT_ACTIONS["ADS_CUSTOM_CONVERSION"]
    + EVENT_ACTIONS["ADS_SIGN_UP_CONVERSION"]
    + EVENT_ACTIONS["ADS_ADD_TO_CART_CONVERSION"]
)
EVENT_ACTIONS["RESHARE"] = (
    EVENT_ACTIONS["RETWEET"] + EVENT_ACTIONS["REPLY"] + EVENT_ACTIONS["BOOKMARK"]
)
EVENT_ACTIONS["OTHER_CONVERSION"] = (
    "ADS_DOWNLOAD_CONVERSION",
    "ADS_ADD_TO_WISHLIST_CONVERSION",
    "ADS_SEARCH_CONVERSION",
    "ADS_ADDED_PAYMENT_INFO_CONVERSION",
    "ADS_SUBSCRIBE_CONVERSION",
    "ADS_START_TRIAL_CONVERSION",
    "ADS_PRODUCT_CUSTOMIZATION_CONVERSION",
)
ENGAGE_EVENTS = (
    "FAV",
    "REPLY",
    "RETWEET",
    "QUOTE",
    "FOLLOW_AUTHOR",
    "BOOKMARK",
    "SHARE",
    "TAKE_SCREENSHOT",
    "CLICK",
    "CLICK_PROFILE",
    "OPEN_LINK",
    "PHOTO_EXPAND",
    "SHOW_MORE_EXPAND",
    "CLICK_HASHTAG",
    "CLICK_MENTION",
    "QUOTED_TWEET_CLICK",
    "TRANSLATE_CLICK",
    "CLICK_GROK_ANALYZE",
    "EXTERNAL_LINK_SESSION",
    "EXTERNAL_LINK_LONG_DWELLED",
    "VIDEO_QUALITY_VIEW",
)
EVENT_ACTIONS["ENGAGE"] = tuple(a for e in ENGAGE_EVENTS for a in EVENT_ACTIONS[e])
EVENT_ACTIONS["CAROUSEL_SWIPE"] = (
    "CLIENT_TWEET_CAROUSEL_SWIPE_NEXT",
    "CLIENT_TWEET_CAROUSEL_SWIPE_PREVIOUS",
)
BRAND_ENGAGE_EVENTS = (
    "FAV",
    "REPLY",
    "RETWEET",
    "FOLLOW_AUTHOR",
    "CLICK",
    "OPEN_LINK",
    "CLICK_PROFILE",
    "PHOTO_EXPAND",
    "CAROUSEL_SWIPE",
)
EVENT_ACTIONS["BRAND_ENGAGE"] = tuple(a for e in BRAND_ENGAGE_EVENTS for a in EVENT_ACTIONS[e])

NEGATIVE_ACTIONS: tuple[str, ...] = (
    "SERVER_TWEET_REPORT",
    "CLIENT_TWEET_REPORT",
    "CLIENT_TWEET_NOT_INTERESTED_IN",
    "CLIENT_TWEET_UNDO_NOT_INTERESTED_IN",
    "CLIENT_TWEET_SEE_FEWER",
    "CLIENT_TWEET_UNDO_SEE_FEWER",
    "CLIENT_TWEET_NOT_RELEVANT",
    "CLIENT_TWEET_UNDO_NOT_RELEVANT",
    "CLIENT_TWEET_BLOCK_AUTHOR",
    "CLIENT_TWEET_UNBLOCK_AUTHOR",
    "CLIENT_TWEET_MUTE_AUTHOR",
    "CLIENT_TWEET_UNMUTE_AUTHOR",
    "CLIENT_TWEET_MUTE_CONVERSATION",
    "CLIENT_TWEET_UNMUTE_CONVERSATION",
    "CLIENT_TWEET_UNFOLLOW_AUTHOR",
    "SERVER_TWEET_UNFAV",
    "SERVER_TWEET_UNREPLY",
    "SERVER_TWEET_UNQUOTE",
    "SERVER_TWEET_UNRETWEET",
    "CLIENT_TWEET_UNBOOKMARK",
    "CLIENT_TWEET_REPLY_DOWNVOTE",
)

WEB_CONV_FAKE_TWEET_ID = 4

MS_PER_DAY = 86_400_000
DWELL_TIME_SLOT = 1
NEGATIVE_BIT = 62

_INTENSITY_EVENTS = ENGAGE_EVENTS + (
    "ANY_CONVERSION",
    "ADS_SESSION_CONVERSION",
    "ADS_SITE_VISIT_CONVERSION",
    "ADS_LONG_SITE_DWELL_CONVERSION",
    "ADS_LANDING_PAGE_VIEW_CONVERSION",
    "ADS_PURCHASE_CONVERSION",
    "ADS_CONTENT_VIEW_CONVERSION",
    "ADS_SIGN_UP_CONVERSION",
    "ADS_CUSTOM_CONVERSION",
    "ADS_ADD_TO_CART_CONVERSION",
    "OTHER_CONVERSION",
    "RECAP_DWELLED",
)


class ActionBits:
    _cache: dict[tuple[int, int], ActionBits] = {}

    @classmethod
    def for_vocab(cls, action_index: Mapping[str, int], vocab_size: int) -> ActionBits:
        key = (id(action_index), vocab_size)
        inst = cls._cache.get(key)
        if inst is None:
            inst = cls._cache[key] = cls(action_index, vocab_size)
        return inst

    def __init__(self, action_index: Mapping[str, int], vocab_size: int):
        def bits(names: Sequence[str]) -> np.ndarray:
            missing = [n for n in names if n not in action_index]
            if missing:
                raise KeyError(f"actions missing from the vocabulary: {missing}")
            idx = sorted({action_index[n] for n in names if action_index[n] < vocab_size})
            return np.asarray(idx, dtype=np.int64)

        self.event: dict[str, np.ndarray] = {e: bits(a) for e, a in EVENT_ACTIONS.items()}
        self.negative: np.ndarray = bits(NEGATIVE_ACTIONS)
        intensity = sorted({int(i) for e in _INTENSITY_EVENTS for i in self.event[e]})
        self.intensity: np.ndarray = np.asarray(intensity, dtype=np.int64)
        self.vocab_size: int = vocab_size


@dataclasses.dataclass(frozen=True)
class HistoryArrays:
    actions: np.ndarray
    author_ids: np.ndarray
    tweet_ids: np.ndarray
    promoted_ids: np.ndarray
    impressed_ms: np.ndarray
    request_ms: np.ndarray
    history_mask: np.ndarray
    dwell_secs: np.ndarray | None = None


def request_time_ms(
    impressed_ms: np.ndarray,
    served_ms: np.ndarray | None,
    candidate_mask: np.ndarray,
    history_mask: np.ndarray,
) -> np.ndarray:
    cand_served = (
        np.where(candidate_mask, served_ms, 0).max(axis=1) if served_ms is not None else None
    )
    cand_imp = np.where(candidate_mask, impressed_ms, 0).max(axis=1)
    hist_imp = np.where(history_mask, impressed_ms, 0).max(axis=1)
    out = np.where(cand_imp > 0, cand_imp, hist_imp)
    if cand_served is not None:
        out = np.where(cand_served > 0, cand_served, out)
    return out.astype(np.int64)


def promoted_history_share(arrays: HistoryArrays) -> float:
    n = int(arrays.history_mask.sum())
    if n == 0:
        return 0.0
    return float(((arrays.promoted_ids != 0) & arrays.history_mask).sum()) / n


def fill(
    ids_out: np.ndarray,
    mask_out: np.ndarray,
    specs: Sequence[IdListFeatureSpec],
    arrays: HistoryArrays,
    bits: ActionBits,
    use_kernel: bool = True,
    num_threads: int | None = None,
) -> None:
    hist_specs = [(i, s) for i, s in enumerate(specs) if is_history_list(s.name)]
    if not hist_specs:
        return
    offsets = slot_offsets(specs)
    B, S = arrays.history_mask.shape
    if B == 0 or S == 0:
        return
    parsed = [(i, s, parse_history_list(s.name)) for i, s in hist_specs]
    dwell = (
        np.ascontiguousarray(arrays.dwell_secs, dtype=np.float32)
        if arrays.dwell_secs is not None
        else np.zeros(arrays.history_mask.shape, dtype=np.float32)
    )
    if use_kernel:
        _fill_with_kernel(ids_out, mask_out, parsed, offsets, arrays, dwell, bits, num_threads)
    else:
        _fill_reference(ids_out, mask_out, parsed, offsets, arrays, dwell, bits)


def _fill_reference(ids_out, mask_out, parsed, offsets, arrays: HistoryArrays, dwell, bits) -> None:
    pos, row, event_mask, intensity = _usable_positions(parsed, arrays, bits)
    if pos.size == 0:
        return
    times = arrays.impressed_ms.reshape(-1)[pos]
    age_ms = arrays.request_ms[row] - times
    promoted = arrays.promoted_ids.reshape(-1)[pos] > 0
    windows = sorted({hl.window_days for _, _, hl in parsed if hl.window_days is not None})
    window_mask = {w: (age_ms >= 0) & (age_ms <= _window_ms(w)) for w in windows}
    dwell_ms = _dwell_ms(dwell.reshape(-1)[pos])
    ents_by_kind = {
        is_author: _entity_codes((arrays.author_ids if is_author else arrays.tweet_ids), pos)
        for is_author in {hl.is_author for _, _, hl in parsed}
    }
    for i, spec, hl in parsed:
        m = event_mask[hl.event]
        if hl.promoted_only:
            m = m & promoted
        elif hl.organic_only:
            m = m & ~promoted
        if hl.window_days is not None:
            m = m & window_mask[hl.window_days]
        raw, codes, n_codes = ents_by_kind[hl.is_author]
        m = m & (raw != 0)
        if not hl.is_author:
            m = m & (raw != WEB_CONV_FAKE_TWEET_ID)
        sel = np.flatnonzero(m)
        if sel.size == 0:
            continue
        rows, ents, t = row[sel], raw[sel], times[sel]
        key = rows * n_codes + codes[sel]
        if hl.kind == "latest":
            rows, ents = _rank_latest(rows, ents, t, key)
        elif hl.is_author:
            rows, ents = _rank_authors(rows, ents, t, key)
        else:
            assert intensity is not None
            rows, ents = _rank_posts(rows, ents, t, key, intensity[sel], dwell_ms[sel])
        _scatter(ids_out, mask_out, rows, ents, offsets[i], spec.k)


def _window_ms(window_days: float) -> int:
    return int(round(window_days * MS_PER_DAY))


def _usable_positions(parsed, arrays: HistoryArrays, bits: ActionBits):
    B, S = arrays.history_mask.shape
    events = sorted({hl.event for _, _, hl in parsed})
    need_intensity = any(not hl.is_author and hl.kind == "top" for _, _, hl in parsed)
    used = sorted(
        set(bits.negative.tolist())
        | {int(i) for ev in events for i in bits.event[ev]}
        | (set(bits.intensity.tolist()) if need_intensity else set())
    )
    col_of = {c: j for j, c in enumerate(used)}
    planes = arrays.actions.reshape(B * S, -1)[:, used].T
    rows_of = lambda idx: [col_of[int(i)] for i in idx]
    base_ok = arrays.history_mask.reshape(-1) & ~planes[rows_of(bits.negative)].any(axis=0)
    pos = np.flatnonzero(base_ok)
    if pos.size == 0:
        return pos, pos, {}, None
    row = (pos // S).astype(np.int64)
    planes = planes[:, pos]
    event_mask = {ev: planes[rows_of(bits.event[ev])].any(axis=0) for ev in events}
    intensity = (
        planes[rows_of(bits.intensity)].sum(axis=0, dtype=np.int16).astype(np.int64)
        if need_intensity
        else None
    )
    return pos, row, event_mask, intensity


def _entity_codes(ids: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    raw = ids.reshape(-1)[pos]
    _, codes = np.unique(raw, return_inverse=True)
    codes = codes.ravel().astype(np.int64)
    return raw, codes, int(codes.max()) + 1


def _first_per_key(key: np.ndarray, order: np.ndarray) -> np.ndarray:
    _, idx = np.unique(key[order], return_index=True)
    idx.sort()
    return order[idx]


def _rank_latest(rows, ents, t, key):
    keep = _first_per_key(key, np.lexsort((ents, -t, rows)))
    return rows[keep], ents[keep]


def _rank_authors(rows, ents, t, key):
    _, first_idx, inv, counts = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    newest = np.full(len(first_idx), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(newest, inv.ravel(), t)
    u_rows, u_ents = rows[first_idx], ents[first_idx]
    order = np.lexsort((u_ents, -newest, -counts, u_rows))
    return u_rows[order], u_ents[order]


def _rank_posts(rows, ents, t, key, intensity, dwell_ms):
    keep = _first_per_key(key, np.lexsort((ents, -t, -dwell_ms, -intensity, rows)))
    return rows[keep], ents[keep]


def _dwell_ms(dwell: np.ndarray) -> np.ndarray:
    return np.rint(dwell.astype(np.float64) * 1000.0).astype(np.int64)


def _scatter(
    ids_out: np.ndarray, mask_out: np.ndarray, rows: np.ndarray, ents: np.ndarray, off: int, k: int
) -> None:
    if len(rows) == 0:
        return
    starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
    pos = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
    sel = pos < k
    ids_out[rows[sel], off + pos[sel]] = ents[sel]
    mask_out[rows[sel], off + pos[sel]] = True


def default_kernel_threads() -> int:
    if "NUMBA_NUM_THREADS" in os.environ:
        return numba.get_num_threads()
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    procs = int(os.environ.get("XREX_LOCAL_PROCESSES", "0")) or _local_device_count()
    return max(1, (cores or 1) // max(1, procs))


def _local_device_count() -> int:
    try:
        import jax

        return jax.local_device_count()
    except Exception:
        return 1


def _fill_with_kernel(
    ids_out, mask_out, parsed, offsets, arrays: HistoryArrays, dwell, bits: ActionBits, num_threads
) -> None:
    numba.set_num_threads(num_threads or default_kernel_threads())
    V = arrays.actions.shape[2]
    events = sorted({hl.event for _, _, hl in parsed})
    if len(events) > NEGATIVE_BIT:
        raise ValueError(f"too many distinct events for one kernel pass: {len(events)}")
    evtable = np.zeros(V, dtype=np.int64)
    for j, ev in enumerate(events):
        evtable[bits.event[ev]] |= np.int64(1) << j
    evtable[bits.negative] |= np.int64(1) << NEGATIVE_BIT
    inten_table = np.zeros(V, dtype=np.bool_)
    inten_table[bits.intensity] = True
    event_index = {ev: j for j, ev in enumerate(events)}
    l_event = np.asarray([event_index[hl.event] for _, _, hl in parsed], dtype=np.int64)
    l_window_ms = np.asarray(
        [-1 if hl.window_days is None else _window_ms(hl.window_days) for _, _, hl in parsed],
        dtype=np.int64,
    )
    l_prom = np.asarray(
        [1 if hl.promoted_only else (2 if hl.organic_only else 0) for _, _, hl in parsed],
        dtype=np.int64,
    )
    l_author = np.asarray([hl.is_author for _, _, hl in parsed], dtype=np.bool_)
    l_latest = np.asarray([hl.kind == "latest" for _, _, hl in parsed], dtype=np.bool_)
    l_k = np.asarray([spec.k for _, spec, _ in parsed], dtype=np.int64)
    l_off = np.asarray([offsets[i] for i, _, _ in parsed], dtype=np.int64)
    _fill_rows(
        ids_out,
        mask_out,
        np.ascontiguousarray(arrays.actions),
        np.ascontiguousarray(arrays.history_mask),
        evtable,
        inten_table,
        np.ascontiguousarray(arrays.impressed_ms, dtype=np.int64),
        np.ascontiguousarray(arrays.promoted_ids, dtype=np.int64),
        np.ascontiguousarray(arrays.author_ids, dtype=np.int64),
        np.ascontiguousarray(arrays.tweet_ids, dtype=np.int64),
        dwell,
        np.ascontiguousarray(arrays.request_ms, dtype=np.int64),
        l_event,
        l_window_ms,
        l_prom,
        l_author,
        l_latest,
        l_k,
        l_off,
    )


@numba.njit(cache=True)
def _rank_and_write(ids_out, mask_out, r, off, k, ent_ids, key1, key2, key3):
    n = ent_ids.shape[0]
    order = np.argsort(ent_ids, kind="mergesort")
    tmp = np.empty(n, dtype=np.int64)
    for key in (key3, key2, key1):
        sub = np.argsort(-key[order], kind="mergesort")
        for i in range(n):
            tmp[i] = order[sub[i]]
        for i in range(n):
            order[i] = tmp[i]
    m = k if k < n else n
    for i in range(m):
        ids_out[r, off + i] = ent_ids[order[i]]
        mask_out[r, off + i] = True


@numba.njit(cache=True, parallel=True)
def _fill_rows(
    ids_out,
    mask_out,
    actions,
    hist_mask,
    evtable,
    inten_table,
    times,
    promoted_ids,
    au_raw,
    tw_raw,
    dw,
    request_ms,
    l_event,
    l_window_ms,
    l_prom,
    l_author,
    l_latest,
    l_k,
    l_off,
):
    B, S, V = actions.shape
    L = l_event.shape[0]
    NEG = np.int64(1) << NEGATIVE_BIT
    for r in numba.prange(B):
        evbits = np.zeros(S, dtype=np.int64)
        inten = np.zeros(S, dtype=np.int64)
        ok = np.zeros(S, dtype=np.bool_)
        for p in range(S):
            if not hist_mask[r, p]:
                continue
            bitsum = np.int64(0)
            cnt = 0
            for v in range(V):
                if actions[r, p, v]:
                    bitsum |= evtable[v]
                    if inten_table[v]:
                        cnt += 1
            if bitsum & NEG:
                continue
            evbits[p] = bitsum
            inten[p] = cnt
            ok[p] = True
        req = request_ms[r]
        sel = np.empty(S, dtype=np.int64)
        for li in range(L):
            ev = l_event[li]
            win = l_window_ms[li]
            pm = l_prom[li]
            is_au = l_author[li]
            cnt = 0
            for p in range(S):
                if not ok[p] or (evbits[p] >> ev) & 1 == 0:
                    continue
                prom = promoted_ids[r, p] > 0
                if pm == 1 and not prom:
                    continue
                if pm == 2 and prom:
                    continue
                if win >= 0:
                    age = req - times[r, p]
                    if age > win or age < 0:
                        continue
                ent = au_raw[r, p] if is_au else tw_raw[r, p]
                if ent == 0 or (not is_au and ent == WEB_CONV_FAKE_TWEET_ID):
                    continue
                sel[cnt] = p
                cnt += 1
            if cnt == 0:
                continue
            ids_sel = np.empty(cnt, dtype=np.int64)
            for i in range(cnt):
                p = sel[i]
                ids_sel[i] = au_raw[r, p] if is_au else tw_raw[r, p]
            order = np.argsort(ids_sel, kind="mergesort")
            g_id = np.empty(cnt, dtype=np.int64)
            g_count = np.zeros(cnt, dtype=np.int64)
            g_newest = np.empty(cnt, dtype=np.int64)
            g_inten = np.empty(cnt, dtype=np.int64)
            g_dwell = np.empty(cnt, dtype=np.int64)
            g_best_t = np.empty(cnt, dtype=np.int64)
            ng = 0
            prev = np.int64(-1)
            for j in range(cnt):
                p = sel[order[j]]
                e = ids_sel[order[j]]
                dq = np.int64(np.rint(dw[r, p] * 1000.0))
                if e != prev:
                    ng += 1
                    prev = e
                    g_id[ng - 1] = e
                    g_count[ng - 1] = 1
                    g_newest[ng - 1] = times[r, p]
                    g_inten[ng - 1] = inten[p]
                    g_dwell[ng - 1] = dq
                    g_best_t[ng - 1] = times[r, p]
                else:
                    g_count[ng - 1] += 1
                    if times[r, p] > g_newest[ng - 1]:
                        g_newest[ng - 1] = times[r, p]
                    if (
                        inten[p] > g_inten[ng - 1]
                        or (inten[p] == g_inten[ng - 1] and dq > g_dwell[ng - 1])
                        or (
                            inten[p] == g_inten[ng - 1]
                            and dq == g_dwell[ng - 1]
                            and times[r, p] > g_best_t[ng - 1]
                        )
                    ):
                        g_inten[ng - 1] = inten[p]
                        g_dwell[ng - 1] = dq
                        g_best_t[ng - 1] = times[r, p]
            ids = g_id[:ng]
            zero = np.zeros(ng, dtype=np.int64)
            if l_latest[li]:
                _rank_and_write(
                    ids_out, mask_out, r, l_off[li], l_k[li], ids, g_newest[:ng], zero, zero
                )
            elif is_au:
                _rank_and_write(
                    ids_out, mask_out, r, l_off[li], l_k[li], ids, g_count[:ng], g_newest[:ng], zero
                )
            else:
                _rank_and_write(
                    ids_out,
                    mask_out,
                    r,
                    l_off[li],
                    l_k[li],
                    ids,
                    g_inten[:ng],
                    g_dwell[:ng],
                    g_best_t[:ng],
                )
