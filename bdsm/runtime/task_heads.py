from __future__ import annotations

import os

import heads
import jax
import jax.numpy as jnp
import numpy as np

SEED = 42


def init_head_params(emb_dim: int, hidden_dims: tuple[int, ...]) -> dict:
    dims = (emb_dim, *hidden_dims, heads.N_HEADS)
    params: dict = {}
    key = jax.random.PRNGKey(SEED)
    for i in range(len(dims) - 1):
        key, sub = jax.random.split(key)
        params[f"w{i}"] = 0.02 * jax.random.truncated_normal(
            sub, -2.0, 2.0, (dims[i], dims[i + 1]), jnp.float32
        )
        params[f"b{i}"] = jnp.zeros(dims[i + 1], jnp.float32)
    return params


def head_logits(params: dict, x: jax.Array) -> jax.Array:
    n_layers = len(params) // 2
    for i in range(n_layers):
        x = x @ params[f"w{i}"] + params[f"b{i}"]
        if i < n_layers - 1:
            x = jax.nn.gelu(x)
    return x


def load_head_params(head_checkpoint: str) -> dict:
    arrays = np.load(os.path.join(head_checkpoint, "head_params.npz"))
    params = {}
    for key in arrays.files:
        params[key.split("/")[-1]] = arrays[key]
    expected = {"w0", "b0", "w1", "b1", "w2", "b2"}
    if set(params) != expected:
        raise RuntimeError(
            f"unexpected head param layout {sorted(params)} (expected {sorted(expected)})"
        )
    return params
