from __future__ import annotations

import heads
import numpy as np


def labels_to_vectors(
    label_strings: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    vec = np.zeros(heads.N_HEADS, dtype=np.bool_)
    mask = np.ones(heads.N_HEADS, dtype=np.bool_)

    if label_strings:
        for label in label_strings:
            head_idx = heads.FLYWHEEL_LABEL_TO_HEAD.get(label)
            if head_idx is None:
                head_idx = heads.HEAD_INDEX.get(label)
            if head_idx is not None and 0 <= head_idx < heads.N_HEADS:
                vec[head_idx] = True
    else:
        vec[heads.LEGITIMATE_USER_INDEX] = True

    return vec, mask
