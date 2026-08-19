# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import threading
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xrex.models.model_utils import Parameter
from xrex.utils.utils import flatten_dict

_metrics = {}
_lock = threading.Lock()


def pop():
    with _lock:
        result = _metrics.copy()
        _metrics.clear()
    return result


def addto(key, value, default):
    with _lock:
        _metrics[key] = _metrics.setdefault(key, default) + value


def _is_embedding_param(key: str) -> bool:
    return "in_out_embed" in key and (key.endswith("/embeddings") or key.endswith("/unembeddings"))


def _embedding_per_row_axis(key: str) -> int:
    return 0 if key.endswith("/embeddings") else -1


def _embedding_row_norm_ratios(
    per_row_norms: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    med = jnp.median(per_row_norms)
    eps = 1e-10
    return (
        jnp.min(per_row_norms) / (med + eps),
        jnp.max(per_row_norms) / (med + eps),
        jnp.argmin(per_row_norms),
        jnp.argmax(per_row_norms),
    )


def _compute_norm(p, axes=None):
    return jnp.sqrt(jnp.sum(p**2, axis=axes)).astype(p.dtype)


def split_metrics(
    metric_dict: dict[str, Parameter],
    suffix: str,
    reducer: Callable[[jax.Array, tuple[int, ...] | None], jax.Array] = _compute_norm,
) -> dict:
    def layer_stack_base(key: str, layer_idx: int) -> str:
        key = key.replace("transformer/layer_stack/decoder_layer_", "decoder_layer_")
        return "/".join(
            [
                part if not part.startswith("decoder_layer_") else part + f"/layer_{layer_idx}"
                for part in key.split("/")
            ]
        )

    splitted_dict = {}
    for k, v in metric_dict.items():
        if "layer_stack" in k:
            values = reducer(v.x, tuple(range(1, v.x.ndim)))
            for i in range(v.shape[0]):
                splitted_dict[layer_stack_base(k, i) + suffix] = values[i]
        else:
            splitted_dict[k + suffix] = reducer(v.x, None)
            if _is_embedding_param(k) and v.x.ndim == 2:
                per_row_axis = _embedding_per_row_axis(k) % v.x.ndim
                reduce_axes = [a for a in range(v.x.ndim) if a != per_row_axis]
                per_row = reducer(v.x, axes=reduce_axes)
                min_ratio, max_ratio, argmin_row, argmax_row = _embedding_row_norm_ratios(per_row)
                splitted_dict[k + suffix + "/row_min_to_median"] = min_ratio
                splitted_dict[k + suffix + "/row_max_to_median"] = max_ratio
                splitted_dict[k + suffix + "/row_argmin"] = argmin_row
                splitted_dict[k + suffix + "/row_argmax"] = argmax_row

    return splitted_dict


def _find_moment_state(state):
    if hasattr(state, "mu") and hasattr(state, "nu"):
        return state
    if isinstance(state, (tuple, list)):
        for sub in state:
            found = _find_moment_state(sub)
            if found is not None:
                return found
    return None


def norm_metrics(params, opt_state, gradients, updates):
    from xrex.optimizers.optim import InjectHyperparamsState

    assert isinstance(opt_state, InjectHyperparamsState)
    metrics = {}

    if not isinstance(opt_state.inner_state, optax._src.combine.MultiTransformState):
        _adam_state = _find_moment_state(opt_state.inner_state)
        if _adam_state is not None:
            mu_metric = split_metrics(flatten_dict(_adam_state.mu), "_mu")
            metrics.update(mu_metric)
            nu_metric = split_metrics(flatten_dict(_adam_state.nu), "_nu")
            metrics.update(nu_metric)

    weight_norm_metric = split_metrics(flatten_dict(params), "_norm")
    metrics.update(weight_norm_metric)
    grad_norm_metric = split_metrics(flatten_dict(gradients), "_grad_norm")
    metrics.update(grad_norm_metric)
    update_norm_metric = split_metrics(flatten_dict(updates), "_update_norm")
    metrics.update(update_norm_metric)
    return metrics


def extract_stacked_metrics(metrics):
    assert len(metrics) == 1, f"Require single entry dict but got {list(metrics.keys())}"
    metric_keys, metrics_array = next(iter(metrics.items()))
    host_sharding = metrics_array.sharding.with_memory_kind("pinned_host")
    metrics_array = np.array(jax.device_put(metrics_array, host_sharding))
    result = dict(zip(metric_keys, metrics_array))
    return result
