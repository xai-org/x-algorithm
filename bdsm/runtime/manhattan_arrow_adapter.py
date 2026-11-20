import logging

import numpy as np
import pyarrow as pa

try:
    import zstd
except ImportError:
    import zstandard

    class _ZstdCompat:
        @staticmethod
        def decompress(data: bytes) -> bytes:
            return zstandard.ZstdDecompressor().decompress(data, max_output_size=64 * 1024 * 1024)

    zstd = _ZstdCompat()

import heads

log = logging.getLogger("manhattan_arrow_adapter")

N_ACTION_TYPES = 256
N_LABELS = heads.N_HEADS
PROTO_RENDER_ACTIONS = frozenset({11, 12})
PROTO_SERVER_ACTIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 61, 62, 70})


def _decompress_arrow(raw_bytes: bytes) -> pa.Table:
    decompressed = zstd.decompress(raw_bytes)
    reader = pa.ipc.RecordBatchFileReader(pa.py_buffer(decompressed))
    return reader.read_all()


def _extract_columns_from_arrow(
    table: pa.Table, max_actions: int = 512
) -> dict[str, np.ndarray] | None:
    if table.num_rows == 0:
        return None

    actions_list = table.column("user_actions")
    if actions_list.null_count == table.num_rows:
        return None

    struct_array = actions_list.combine_chunks().values

    total_actions = len(struct_array)
    if total_actions == 0:
        return None

    if total_actions > max_actions:
        struct_array = struct_array.slice(total_actions - max_actions)

    def _field_np(name, dtype=np.int64, fill=0):
        try:
            # Filling while still in Arrow preserves integer precision. Nullable
            # integer arrays otherwise become float64, rounding IDs above 2**53
            # and turning nulls into the minimum signed integer on conversion.
            field = struct_array.field(name).fill_null(fill)
            arr = field.to_numpy(zero_copy_only=False).astype(dtype)
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=fill)
            return arr
        except (KeyError, ValueError):
            return np.full(len(struct_array), fill, dtype=dtype)

    return {
        "action_name": _field_np("action_name", np.int32, 0),
        "tweet_id": _field_np("tweet_id", np.int64, 0),
        "author_id": _field_np("author_id", np.int64, 0),
        "engagement_time_ms": _field_np("engagement_time_ms", np.int64, 0),
        "dwell_time": _field_np("dwell_time", np.int32, 0),
        "product_surface": _field_np("product_surface", np.int32, 0),
        "client_app_id": _field_np("client_app_id", np.int32, 0),
    }


def arrow_bytes_to_batch(
    user_raw_bytes: list[tuple[int, bytes]],
    seq_len: int = 2048,
    n_action_types: int | None = None,
) -> tuple[dict, np.ndarray]:
    if n_action_types is None:
        n_action_types = N_ACTION_TYPES
    B = len(user_raw_bytes)
    S = seq_len

    user_ids = np.zeros(B, dtype=np.int64)
    action_idx = np.zeros((B, S), dtype=np.uint8)
    tweet_ids = np.zeros((B, S), dtype=np.int64)
    author_ids = np.zeros((B, S), dtype=np.int64)
    engagement_times = np.zeros((B, S), dtype=np.int64)
    dwell_times = np.zeros((B, S), dtype=np.int32)
    product_surface = np.zeros((B, S), dtype=np.int32)
    client_app_id = np.zeros((B, S), dtype=np.int32)
    padding_mask = np.zeros((B, S), dtype=np.bool_)

    for b, (uid, raw_bytes) in enumerate(user_raw_bytes):
        user_ids[b] = uid

        try:
            table = _decompress_arrow(raw_bytes)
            cols = _extract_columns_from_arrow(table, max_actions=S)
        except Exception as e:
            log.warning(f"Failed to parse Arrow for user {uid}: {e}")
            continue

        if cols is None:
            continue

        N = len(cols["action_name"])
        offset = S - N

        padding_mask[b, offset:] = True

        action_idx[b, offset:] = np.clip(cols["action_name"], 0, n_action_types - 1).astype(
            np.uint8
        )

        tweet_ids[b, offset:] = cols["tweet_id"]
        author_ids[b, offset:] = cols["author_id"]
        engagement_times[b, offset:] = cols["engagement_time_ms"]
        dwell_times[b, offset:] = np.minimum(cols["dwell_time"], 300_000)
        product_surface[b, offset:] = cols["product_surface"]
        client_app_id[b, offset:] = cols["client_app_id"]

    time_delta_norm = np.zeros((B, S), dtype=np.float32)
    shifted_times = np.roll(engagement_times, 1, axis=1)
    shifted_times[:, 0] = 0
    both_valid = padding_mask & np.roll(padding_mask, 1, axis=1)
    both_valid[:, 0] = False
    both_positive = (engagement_times > 0) & (shifted_times > 0) & both_valid
    delta_ms = np.where(both_positive, engagement_times - shifted_times, 0).astype(np.float64)
    delta_min = np.maximum(delta_ms / 60_000.0, 0.0)
    time_delta_norm = np.where(
        both_positive,
        np.minimum(np.log1p(delta_min), 10.0) / 10.0,
        0.0,
    ).astype(np.float32)

    dwell_norm = np.zeros((B, S), dtype=np.float32)
    valid_dwell = dwell_times > 0
    dwell_norm[valid_dwell] = np.log1p(
        dwell_times[valid_dwell].astype(np.float32) / 1000.0
    ) / np.log1p(300.0)

    hour_of_day = np.zeros((B, S), dtype=np.int32)
    valid_times = engagement_times > 0
    hour_of_day[valid_times] = ((engagement_times[valid_times] // 3_600_000) % 24).astype(np.int32)

    had_prior_render = np.zeros((B, S), dtype=np.bool_)
    is_render = np.isin(action_idx, list(PROTO_RENDER_ACTIONS)) & padding_mask & (tweet_ids > 0)

    for b in range(B):
        render_positions = np.where(is_render[b])[0]
        if len(render_positions) == 0:
            continue
        rendered_tweets: dict[int, int] = {}
        for rp in render_positions:
            tid = tweet_ids[b, rp]
            if tid not in rendered_tweets:
                rendered_tweets[tid] = rp
        if rendered_tweets:
            valid = padding_mask[b] & (tweet_ids[b] > 0)
            valid_positions = np.where(valid)[0]
            for pos in valid_positions:
                tid = tweet_ids[b, pos]
                first_render = rendered_tweets.get(tid, S + 1)
                if pos > first_render:
                    had_prior_render[b, pos] = True

    action_multihot = np.zeros((B, S, n_action_types), dtype=np.bool_)
    valid_action = (action_idx > 0) & (action_idx < n_action_types)
    rows, cols = np.where(valid_action)
    action_multihot[rows, cols, action_idx[rows, cols]] = True

    continuous_actions = np.stack([time_delta_norm, dwell_norm], axis=-1)

    impr_ts = np.zeros((B, S), dtype=np.int32)
    impr_ts[valid_times] = (engagement_times[valid_times] // 1000).astype(np.int32)

    summary_counters = np.zeros((B, 15), dtype=np.uint32)
    for b in range(B):
        valid = padding_mask[b]
        acts = action_idx[b][valid]
        total = int(valid.sum())
        if total == 0:
            continue

        from collections import Counter

        counts = Counter(acts.tolist())

        server_actions = sum(counts.get(i, 0) for i in PROTO_SERVER_ACTIONS)
        renders = counts.get(12, 0)
        dwells = counts.get(11, 0)
        clicks = counts.get(29, 0)
        follows = counts.get(21, 0) + counts.get(61, 0)
        replies = counts.get(4, 0)
        favs = counts.get(1, 0)
        retweets = counts.get(6, 0)
        photo = counts.get(14, 0)
        distinct_types = len(counts)
        distinct_targets = len(
            set(author_ids[b, s] for s in range(S) if valid[s] and author_ids[b, s] > 0)
        )
        distinct_posts = len(
            set(tweet_ids[b, s] for s in range(S) if valid[s] and tweet_ids[b, s] > 0)
        )
        dwell_vals = dwell_times[b][valid & (dwell_times[b] > 0)]
        avg_dwell = int(dwell_vals.mean()) if len(dwell_vals) > 0 else 0
        server_pct = round(100 * server_actions / total) if total > 0 else 0
        client_follows = counts.get(21, 0)

        summary_counters[b] = [
            total,
            server_pct,
            renders,
            dwells,
            avg_dwell,
            distinct_types,
            distinct_targets,
            distinct_posts,
            follows,
            replies,
            favs,
            retweets,
            clicks,
            photo,
            client_follows,
        ]

    batch = {
        "user_ids_raw": user_ids,
        "tweet_ids_raw": tweet_ids,
        "author_ids_raw": author_ids,
        "summary_counters": summary_counters,
        "action_seq": {
            "action_types": action_multihot,
            "continuous_actions": continuous_actions,
            "product_surface": product_surface,
            "client_app_id": client_app_id,
            "hour_of_day": hour_of_day,
            "had_prior_render": had_prior_render,
            "impr_ts": impr_ts,
            "padding_mask": padding_mask,
        },
        "user_features": {
            "account_age_norm": np.zeros(B, dtype=np.float32),
            "followers_log": np.zeros(B, dtype=np.float32),
            "is_verified": np.zeros(B, dtype=np.float32),
            "user_cred_norm": np.zeros(B, dtype=np.float32),
            "app_reputation": np.zeros(B, dtype=np.float32),
        },
        "label_vector": np.zeros((B, N_LABELS), dtype=np.bool_),
        "label_mask": np.zeros((B, N_LABELS), dtype=np.bool_),
    }

    return batch, user_ids
