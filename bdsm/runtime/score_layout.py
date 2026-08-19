from __future__ import annotations

import heads
import numpy as np

HEAD_ORDER: tuple[str, ...] = heads.HEAD_NAMES


def to_score_row(probs8: np.ndarray, head_names8: list[str]) -> np.ndarray:
    out = np.zeros(len(HEAD_ORDER), dtype=np.float32)
    for j, name in enumerate(head_names8):
        out[HEAD_ORDER.index(name)] = probs8[j]
    return out


def action_histogram(
    action_types_row: np.ndarray, padding_mask_row: np.ndarray, idx_to_name: dict[int, str]
) -> list[tuple[str, int]]:
    mask = padding_mask_row.astype(bool)
    col_sums = action_types_row[mask].sum(axis=0)
    counts = []
    for proto_idx in range(len(col_sums)):
        if col_sums[proto_idx] > 0:
            name = idx_to_name.get(proto_idx)
            if name:
                counts.append((name, int(col_sums[proto_idx])))
    return counts


def behavioral_summary(
    action_types_row: np.ndarray,
    padding_mask_row: np.ndarray,
    app_id_row: np.ndarray | None,
    idx_to_name: dict[int, str],
    name_to_idx: dict[str, int],
) -> dict:
    mask = padding_mask_row.astype(bool)
    n_valid = int(mask.sum())
    timeline_parts = []
    col_sums = action_types_row[mask].sum(axis=0)
    for proto_idx in range(len(col_sums)):
        if col_sums[proto_idx] > 0:
            name = idx_to_name.get(proto_idx, f"a{proto_idx}")
            timeline_parts.append(f"{name}x{int(col_sums[proto_idx])}")

    def _dom(action_name: str) -> tuple[int, int, float]:
        idx = name_to_idx.get(action_name)
        if idx is None or app_id_row is None:
            return 0, 0, 0.0
        hits = mask & action_types_row[:, idx]
        n = int(hits.sum())
        if n == 0:
            return 0, 0, 0.0
        apps, counts = np.unique(app_id_row[hits], return_counts=True)
        i = int(np.argmax(counts))
        return n, int(apps[i]), float(counts[i] / n)

    n_create, dom_create_id, dom_create_frac = _dom("SERVER_TWEET_CREATE")
    n_reply, dom_reply_id, dom_reply_frac = _dom("SERVER_TWEET_REPLY")
    return {
        "n_valid": n_valid,
        "decoded_timeline": " ".join(timeline_parts),
        "n_create": n_create,
        "dom_create_app_id": dom_create_id,
        "dom_create_app_frac": dom_create_frac,
        "n_reply": n_reply,
        "dom_reply_app_id": dom_reply_id,
        "dom_reply_app_frac": dom_reply_frac,
    }
