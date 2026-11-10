from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    y = y_true[order].astype(np.float64)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    # A decision threshold retrieves every row sharing its score. Evaluate
    # precision and recall only after the last row of each tied-score group.
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    new_positives = np.diff(np.r_[0.0, tp])
    return float((precision * new_positives).sum() / n_pos)
