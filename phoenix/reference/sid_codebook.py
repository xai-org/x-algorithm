#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

import inspect as _inspect

_SM_PARAMS = _inspect.signature(shard_map).parameters
if "check_vma" in _SM_PARAMS:
    _CHECK_KW = {"check_vma": False}
elif "check_rep" in _SM_PARAMS:
    _CHECK_KW = {"check_rep": False}
else:
    _CHECK_KW = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_MESH: Mesh | None = None
_DATA_SPEC = P("data", None)
_REP_SPEC = P()
_ASSIGN_SPEC = P("data")


def get_mesh() -> Mesh:
    global _MESH
    if _MESH is None:
        devices = jax.devices()
        _MESH = Mesh(devices, ("data",))
        log.info("JAX mesh: %d device(s) on %s", len(devices), devices[0].platform)
        for d in devices:
            log.info("  %s", d)
    return _MESH


@partial(jax.jit, static_argnames=("K",))
def _kmeans_iter(residual, centroids, K: int):
    mesh = get_mesh()

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(_DATA_SPEC, _REP_SPEC),
        out_specs=(_REP_SPEC, _REP_SPEC, _ASSIGN_SPEC),
        **_CHECK_KW,
    )
    def step(res, cent):
        c_sq = jnp.sum(cent * cent, axis=1)
        x_sq = jnp.sum(res * res, axis=1, keepdims=True)
        dists = x_sq + c_sq[None, :] - 2.0 * res @ cent.T
        assigns = jnp.argmin(dists, axis=1).astype(jnp.int32)

        partial_sums = jax.ops.segment_sum(res, assigns, num_segments=K)
        partial_counts = jax.ops.segment_sum(
            jnp.ones(res.shape[0], dtype=jnp.float32), assigns, num_segments=K
        )
        total_sums = jax.lax.psum(partial_sums, axis_name="data")
        total_counts = jax.lax.psum(partial_counts, axis_name="data")
        return total_sums, total_counts, assigns

    sums, counts, assigns = step(residual, centroids)
    new_centroids = jnp.where(
        counts[:, None] > 0,
        sums / jnp.maximum(counts[:, None], 1.0),
        centroids,
    )
    return new_centroids, assigns


@jax.jit
def _update_residual(residual, centroids, assigns):
    mesh = get_mesh()

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(_DATA_SPEC, _REP_SPEC, _ASSIGN_SPEC),
        out_specs=_DATA_SPEC,
        **_CHECK_KW,
    )
    def f(res, cent, asn):
        return res - cent[asn]

    return f(residual, centroids, assigns)


@partial(jax.jit, static_argnames=("K",))
def _assign_codes_one_stage(residual, centroids, K: int):
    mesh = get_mesh()

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(_DATA_SPEC, _REP_SPEC),
        out_specs=_ASSIGN_SPEC,
        **_CHECK_KW,
    )
    def f(res, cent):
        c_sq = jnp.sum(cent * cent, axis=1)
        x_sq = jnp.sum(res * res, axis=1, keepdims=True)
        dists = x_sq + c_sq[None, :] - 2.0 * res @ cent.T
        return jnp.argmin(dists, axis=1).astype(jnp.int32)

    return f(residual, centroids)


def _pad_to_multiple(arr: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    n = arr.shape[0]
    pad = (-n) % multiple
    if pad == 0:
        return arr, 0
    tail = np.tile(arr[-1:], (pad, 1))
    return np.concatenate([arr, tail], axis=0), pad


def _put_sharded(arr: np.ndarray) -> jax.Array:
    mesh = get_mesh()
    return jax.device_put(arr, NamedSharding(mesh, _DATA_SPEC))


def _put_replicated(arr: np.ndarray) -> jax.Array:
    mesh = get_mesh()
    return jax.device_put(arr, NamedSharding(mesh, _REP_SPEC))


def train_rq_kmeans(
    embeddings: np.ndarray,
    num_levels: int,
    num_centroids: int,
    n_iters: int,
    seed: int = 42,
) -> np.ndarray:
    mesh = get_mesh()
    n_devices = len(mesh.devices)

    N_orig, D = embeddings.shape
    embeddings_f32 = embeddings.astype(np.float32, copy=False)
    padded, pad = _pad_to_multiple(embeddings_f32, n_devices)
    N = padded.shape[0]
    log.info(
        "data: %d vectors (padded from %d with %d tile rows), dim=%d, sharded across %d device(s)",
        N,
        N_orig,
        pad,
        D,
        n_devices,
    )

    residual = _put_sharded(padded)

    rng = np.random.RandomState(seed)
    codebooks: list[np.ndarray] = []

    for layer in range(num_levels):
        t_layer = time.time()

        init_idx = rng.choice(N_orig, size=num_centroids, replace=False).astype(np.int64)
        init_centroids = np.asarray(jax.device_get(residual[init_idx])).astype(np.float32)
        centroids = _put_replicated(init_centroids)

        for it in range(n_iters):
            t_it = time.time()
            centroids, assigns = _kmeans_iter(residual, centroids, num_centroids)
            if (it + 1) % 5 == 0 or it == n_iters - 1:
                centroids.block_until_ready()
                log.info(
                    "  layer %d iter %d/%d (%.2fs)",
                    layer,
                    it + 1,
                    n_iters,
                    time.time() - t_it,
                )

        codebooks.append(np.asarray(jax.device_get(centroids)))

        if layer < num_levels - 1:
            residual = _update_residual(residual, centroids, assigns)
            residual.block_until_ready()

        log.info("layer %d done in %.1fs", layer, time.time() - t_layer)

    return np.stack(codebooks, axis=0)


def encode_rq(
    embeddings: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    mesh = get_mesh()
    n_devices = len(mesh.devices)
    N_orig, D = embeddings.shape
    M, K, D_cb = codebooks.shape
    assert D_cb == D, f"dim mismatch: data {D} vs codebook {D_cb}"

    padded, pad = _pad_to_multiple(embeddings.astype(np.float32, copy=False), n_devices)
    residual = _put_sharded(padded)

    all_codes = np.empty((padded.shape[0], M), dtype=np.int32)
    for layer in range(M):
        centroids = _put_replicated(codebooks[layer])
        assigns = _assign_codes_one_stage(residual, centroids, K)
        all_codes[:, layer] = np.asarray(jax.device_get(assigns))
        if layer < M - 1:
            residual = _update_residual(residual, centroids, assigns)
            residual.block_until_ready()

    return all_codes[:N_orig]


def evaluate_codebook(
    codebooks: np.ndarray,
    embeddings: np.ndarray,
    max_eval: int = 100_000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    if len(embeddings) > max_eval:
        idx = rng.choice(len(embeddings), max_eval, replace=False)
        embs = embeddings[idx].astype(np.float32)
    else:
        embs = embeddings.astype(np.float32)

    log.info("evaluating on %d vectors", len(embs))
    codes = encode_rq(embs, codebooks)
    recon = codebooks[np.arange(codebooks.shape[0])[:, None], codes.T].sum(axis=0)

    mse = float(((embs - recon) ** 2).mean())
    rel_mse = mse / float((embs**2).mean()) if (embs**2).mean() > 0 else float("inf")
    cos = (embs * recon).sum(axis=1) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(recon, axis=1) + 1e-12
    )
    log.info(
        "MSE=%.6f  rel=%.4f (%.1f%%)  cos mean=%.4f median=%.4f",
        mse,
        rel_mse,
        rel_mse * 100,
        float(cos.mean()),
        float(np.median(cos)),
    )

    per_level = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=codebooks.shape[1])
        nonzero = counts[counts > 0]
        p = nonzero / nonzero.sum()
        entropy = float(-(p * np.log2(p)).sum())
        ppl = float(2**entropy)
        dead = int((counts == 0).sum())
        per_level.append(
            {
                "level": level,
                "entropy_bits": entropy,
                "perplexity": ppl,
                "dead_codes": dead,
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
        )
        log.info(
            "  level %d: H=%.2fb ppl=%.1f dead=%d min=%d max=%d",
            level,
            entropy,
            ppl,
            dead,
            int(counts.min()),
            int(counts.max()),
        )

    return {"mse": mse, "rel_mse": rel_mse, "per_level": per_level}


def _init_mlp(key: jax.Array, dims: list[int]) -> dict:
    params = {}
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        scale = jnp.sqrt(2.0 / dims[i])
        params[f"w{i}"] = jax.random.normal(k, (dims[i], dims[i + 1])) * scale
        params[f"b{i}"] = jnp.zeros(dims[i + 1])
    return params


def _mlp_n_layers(params: dict) -> int:
    return len([k for k in params if k.startswith("w")])


def _mlp_forward(params: dict, x: jax.Array, n_layers: int | None = None) -> jax.Array:
    if n_layers is None:
        n_layers = _mlp_n_layers(params)
    for i in range(n_layers - 1):
        x = jnp.maximum(x @ params[f"w{i}"] + params[f"b{i}"], 0.0)
    return x @ params[f"w{n_layers - 1}"] + params[f"b{n_layers - 1}"]


def _rq_quantize(latent: jax.Array, codebooks: jax.Array):
    M = codebooks.shape[0]
    residual = latent
    quantized_sum = jnp.zeros_like(latent)
    all_codes = []
    all_residuals = []
    commit_loss = jnp.float32(0.0)

    for s in range(M):
        all_residuals.append(residual)
        cb = codebooks[s]
        c_sq = jnp.sum(cb * cb, axis=1)
        r_sq = jnp.sum(residual * residual, axis=1, keepdims=True)
        dists = r_sq + c_sq[None, :] - 2.0 * residual @ cb.T
        idx = jnp.argmin(dists, axis=1).astype(jnp.int32)
        selected = cb[idx]

        quantized_st = residual + jax.lax.stop_gradient(selected - residual)

        commit_loss = commit_loss + jnp.mean((residual - jax.lax.stop_gradient(selected)) ** 2)

        quantized_sum = quantized_sum + quantized_st
        residual = residual - quantized_st
        all_codes.append(idx)

    commit_loss = commit_loss / M
    return quantized_sum, jnp.stack(all_codes, axis=1), commit_loss, all_residuals


def _ema_update_codebooks(
    codebooks: np.ndarray,
    ema_counts: np.ndarray,
    ema_sums: np.ndarray,
    codes: np.ndarray,
    residuals_per_stage: list[np.ndarray],
    decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, K, d = codebooks.shape
    for s in range(M):
        res = residuals_per_stage[s]
        c = codes[:, s]
        counts = np.bincount(c, minlength=K).astype(np.float32)
        sums = np.zeros((K, d), dtype=np.float32)
        np.add.at(sums, c, res)

        ema_counts[s] = decay * ema_counts[s] + (1 - decay) * counts
        ema_sums[s] = decay * ema_sums[s] + (1 - decay) * sums

        n = ema_counts[s] + 1e-5
        codebooks[s] = ema_sums[s] / n[:, None]

    return codebooks, ema_counts, ema_sums


def _revive_dead_codes(
    codebooks: np.ndarray,
    ema_counts: np.ndarray,
    ema_sums: np.ndarray,
    encoder_outputs: np.ndarray,
    rng: np.random.RandomState,
    threshold: float = 1.0,
) -> int:
    M, K, d = codebooks.shape
    total_revived = 0
    for s in range(M):
        dead_mask = ema_counts[s] < threshold
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            continue
        replacements = encoder_outputs[rng.choice(len(encoder_outputs), n_dead, replace=True)]
        codebooks[s, dead_mask] = replacements
        ema_counts[s, dead_mask] = 1.0
        ema_sums[s, dead_mask] = replacements
        total_revived += n_dead
    return total_revived


def train_rqvae(
    embeddings: np.ndarray,
    num_levels: int,
    num_centroids: int,
    latent_dim: int = 64,
    hidden_dim: int = 512,
    num_steps: int = 5000,
    batch_size: int = 4096,
    warmup_steps: int = 500,
    lr: float = 3e-4,
    beta: float = 0.25,
    ema_decay: float = 0.99,
    revival_every: int = 200,
    seed: int = 42,
) -> tuple[dict, dict, np.ndarray]:
    import optax

    N, D = embeddings.shape
    emb_f32 = embeddings.astype(np.float32, copy=False)

    key = jax.random.PRNGKey(seed)
    k_enc, k_dec = jax.random.split(key)
    enc_params = _init_mlp(k_enc, [D, hidden_dim, latent_dim])
    dec_params = _init_mlp(k_dec, [latent_dim, hidden_dim, D])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init((enc_params, dec_params))

    codebooks = None
    ema_counts = None
    ema_sums = None

    rng = np.random.RandomState(seed)

    @jax.jit
    def warmup_step(enc_params, dec_params, opt_state, batch):
        def loss_fn(enc_p, dec_p):
            latent = _mlp_forward(enc_p, batch)
            decoded = _mlp_forward(dec_p, latent)
            cos = jnp.sum(decoded * batch, axis=1) / (
                jnp.linalg.norm(decoded, axis=1) * jnp.linalg.norm(batch, axis=1) + 1e-8
            )
            return (1.0 - cos).mean()

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(enc_params, dec_params)
        updates, opt_state_new = optimizer.update(grads, opt_state, (enc_params, dec_params))
        enc_new, dec_new = optax.apply_updates((enc_params, dec_params), updates)
        return enc_new, dec_new, opt_state_new, loss

    @jax.jit
    def vae_step(enc_params, dec_params, opt_state, batch, codebooks_jax):
        cb_sg = jax.lax.stop_gradient(codebooks_jax)

        def loss_fn(enc_p, dec_p):
            latent = _mlp_forward(enc_p, batch)
            quantized, codes, commit_loss, _ = _rq_quantize(latent, cb_sg)
            decoded = _mlp_forward(dec_p, quantized)
            cos = jnp.sum(decoded * batch, axis=1) / (
                jnp.linalg.norm(decoded, axis=1) * jnp.linalg.norm(batch, axis=1) + 1e-8
            )
            recon_loss = (1.0 - cos).mean()
            return recon_loss + beta * commit_loss, (recon_loss, commit_loss, codes, latent)

        (loss, (recon_loss, commit_loss, codes, latent)), grads = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(enc_params, dec_params)
        updates, opt_state_new = optimizer.update(grads, opt_state, (enc_params, dec_params))
        enc_new, dec_new = optax.apply_updates((enc_params, dec_params), updates)
        return enc_new, dec_new, opt_state_new, loss, recon_loss, commit_loss, codes, latent

    log.info(
        "RQ-VAE training: N=%d D=%d latent=%d hidden=%d levels=%d K=%d",
        N,
        D,
        latent_dim,
        hidden_dim,
        num_levels,
        num_centroids,
    )
    log.info(
        "  steps=%d warmup=%d batch=%d lr=%.1e beta=%.2f ema=%.3f revival=%d",
        num_steps,
        warmup_steps,
        batch_size,
        lr,
        beta,
        ema_decay,
        revival_every,
    )

    t0 = time.time()
    for step in range(num_steps):
        idx = rng.choice(N, batch_size, replace=False)
        batch = jnp.array(emb_f32[idx])

        if step < warmup_steps:
            enc_params, dec_params, opt_state, loss = warmup_step(
                enc_params, dec_params, opt_state, batch
            )
            if step % 100 == 0 or step == warmup_steps - 1:
                log.info(
                    "  warmup %d/%d  loss=%.4f (%.1fs)",
                    step,
                    warmup_steps,
                    float(loss),
                    time.time() - t0,
                )

        else:
            if step == warmup_steps:
                log.info("  warmup done. Initializing codebooks via k-means on latent space...")
                init_size = min(N, 500_000)
                init_idx = rng.choice(N, init_size, replace=False)
                init_emb = emb_f32[init_idx]
                init_latents = np.asarray(
                    jax.device_get(_mlp_forward(enc_params, jnp.array(init_emb)))
                )
                codebooks = train_rq_kmeans(
                    init_latents,
                    num_levels,
                    num_centroids,
                    n_iters=10,
                    seed=seed,
                )
                ema_counts = np.ones((num_levels, num_centroids), dtype=np.float32)
                ema_sums = codebooks.copy() * ema_counts[:, :, None]
                log.info("  codebooks initialized: %s", codebooks.shape)

            cb_jax = jnp.array(codebooks)
            enc_params, dec_params, opt_state, loss, recon_l, commit_l, codes_jax, latent_jax = (
                vae_step(enc_params, dec_params, opt_state, batch, cb_jax)
            )

            codes_np = np.asarray(jax.device_get(codes_jax))
            latent_np = np.asarray(jax.device_get(latent_jax))
            residuals = []
            res = latent_np.copy()
            for s in range(num_levels):
                residuals.append(res.copy())
                res = res - codebooks[s, codes_np[:, s]]
            codebooks, ema_counts, ema_sums = _ema_update_codebooks(
                codebooks,
                ema_counts,
                ema_sums,
                codes_np,
                residuals,
                ema_decay,
            )

            if (step - warmup_steps) > 0 and (step - warmup_steps) % revival_every == 0:
                n_revived = _revive_dead_codes(
                    codebooks,
                    ema_counts,
                    ema_sums,
                    latent_np,
                    rng,
                    threshold=1.0,
                )
                if n_revived > 0:
                    log.info("  step %d: revived %d dead codes", step, n_revived)

            if step % 100 == 0 or step == num_steps - 1:
                dead_per_stage = [(ema_counts[s] < 1.0).sum() for s in range(num_levels)]
                worst_dead = max(dead_per_stage)
                log.info(
                    "  step %d/%d  loss=%.4f  recon=%.4f  commit=%.4f  dead=%d (%.1fs)",
                    step,
                    num_steps,
                    float(loss),
                    float(recon_l),
                    float(commit_l),
                    worst_dead,
                    time.time() - t0,
                )

    log.info("RQ-VAE training done in %.1fs", time.time() - t0)

    enc_np = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), enc_params)
    dec_np = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), dec_params)
    return enc_np, dec_np, codebooks


def encode_rqvae(
    embeddings: np.ndarray,
    enc_params: dict,
    codebooks: np.ndarray,
) -> np.ndarray:
    latent = np.asarray(
        _mlp_forward(
            jax.tree.map(jnp.array, enc_params),
            jnp.array(embeddings.astype(np.float32)),
            2,
        )
    )
    M, K, d = codebooks.shape
    codes = np.empty((len(embeddings), M), dtype=np.int32)
    residual = latent
    for s in range(M):
        cb = codebooks[s]
        c_sq = (cb * cb).sum(axis=1)
        r_sq = (residual * residual).sum(axis=1, keepdims=True)
        dists = r_sq + c_sq[None, :] - 2.0 * residual @ cb.T
        codes[:, s] = dists.argmin(axis=1)
        residual = residual - cb[codes[:, s]]
    return codes


def evaluate_rqvae(
    enc_params: dict,
    dec_params: dict,
    codebooks: np.ndarray,
    embeddings: np.ndarray,
    max_eval: int = 100_000,
    seed: int = 42,
) -> dict:
    rng_np = np.random.RandomState(seed)
    if len(embeddings) > max_eval:
        idx = rng_np.choice(len(embeddings), max_eval, replace=False)
        embs = embeddings[idx].astype(np.float32)
    else:
        embs = embeddings.astype(np.float32)

    log.info("evaluating RQ-VAE on %d vectors", len(embs))

    enc_jax = jax.tree.map(jnp.array, enc_params)
    dec_jax = jax.tree.map(jnp.array, dec_params)
    latent = np.asarray(_mlp_forward(enc_jax, jnp.array(embs)))

    M, K, d = codebooks.shape
    codes = np.empty((len(embs), M), dtype=np.int32)
    residual = latent.copy()
    quantized = np.zeros_like(latent)
    for s in range(M):
        cb = codebooks[s]
        c_sq = (cb * cb).sum(axis=1)
        r_sq = (residual * residual).sum(axis=1, keepdims=True)
        dists = r_sq + c_sq[None, :] - 2.0 * residual @ cb.T
        codes[:, s] = dists.argmin(axis=1)
        selected = cb[codes[:, s]]
        quantized += selected
        residual -= selected

    recon = np.asarray(_mlp_forward(dec_jax, jnp.array(quantized)))

    mse = float(((embs - recon) ** 2).mean())
    rel_mse = mse / float((embs**2).mean())
    cos = (embs * recon).sum(axis=1) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(recon, axis=1) + 1e-12
    )
    log.info(
        "MSE=%.6f  rel=%.4f (%.1f%%)  cos mean=%.4f median=%.4f",
        mse,
        rel_mse,
        rel_mse * 100,
        float(cos.mean()),
        float(np.median(cos)),
    )

    per_level = []
    for level in range(M):
        counts = np.bincount(codes[:, level], minlength=K)
        nonzero = counts[counts > 0]
        p = nonzero / nonzero.sum()
        entropy = float(-(p * np.log2(p)).sum())
        ppl = float(2**entropy)
        dead = int((counts == 0).sum())
        per_level.append(
            {
                "level": level,
                "entropy_bits": entropy,
                "perplexity": ppl,
                "dead_codes": dead,
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
        )
        log.info(
            "  level %d: H=%.2fb ppl=%.1f dead=%d min=%d max=%d",
            level,
            entropy,
            ppl,
            dead,
            int(counts.min()),
            int(counts.max()),
        )

    return {"mse": mse, "rel_mse": rel_mse, "cos_mean": float(cos.mean()), "per_level": per_level}


from sid_io import (
    load_codebook as _load_codebook,
)
from sid_io import (
    save_codebook_npz,
    save_codebook_safetensors,
    save_rqvae_npz,
    save_rqvae_safetensors,
)


def load_codebook_npz(path: str | Path) -> np.ndarray:
    stages, _, _ = _load_codebook(path)
    return stages


def load_rqvae_npz(path: str | Path) -> tuple[dict, dict, np.ndarray]:
    stages, _, rqvae_params = _load_codebook(path)
    if rqvae_params is None:
        raise ValueError(f"{path} is not an RQ-VAE model")
    enc_params, dec_params = rqvae_params
    return enc_params, dec_params, stages


def cmd_train(args):
    log.info("loading embeddings from %s", args.training_data)
    embs = np.load(args.training_data)
    if args.sample_size and args.sample_size < len(embs):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(embs), args.sample_size, replace=False)
        embs = embs[idx]
    log.info("training on %d vectors, dim=%d", len(embs), embs.shape[1])

    t0 = time.time()
    codebooks = train_rq_kmeans(
        embs,
        num_levels=args.num_levels,
        num_centroids=args.num_centroids,
        n_iters=args.iters,
        seed=args.seed,
    )
    log.info("training done in %.1fs", time.time() - t0)

    out = Path(args.codebook_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".safetensors":
        save_codebook_safetensors(out, codebooks, args.iters, args.seed, len(embs))
    else:
        save_codebook_npz(out, codebooks, args.iters, args.seed, len(embs))

    evaluate_codebook(codebooks, embs, max_eval=min(100_000, len(embs)), seed=args.seed)


def cmd_evaluate(args):
    codebooks = load_codebook_npz(args.codebook)
    log.info("loaded codebook: %s  shape=%s", args.codebook, codebooks.shape)

    embs = np.load(args.training_data)
    if args.eval_size and args.eval_size < len(embs):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(embs), args.eval_size, replace=False)
        embs = embs[idx]

    evaluate_codebook(codebooks, embs, max_eval=len(embs), seed=args.seed)


def cmd_train_vae(args):
    log.info("loading embeddings from %s", args.training_data)
    embs = np.load(args.training_data)
    if args.sample_size and args.sample_size < len(embs):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(embs), args.sample_size, replace=False)
        embs = embs[idx]
    log.info("training RQ-VAE on %d vectors, dim=%d", len(embs), embs.shape[1])

    t0 = time.time()
    enc_params, dec_params, codebooks = train_rqvae(
        embs,
        num_levels=args.num_levels,
        num_centroids=args.num_centroids,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        beta=args.beta,
        ema_decay=args.ema_decay,
        revival_every=args.revival_every,
        seed=args.seed,
    )
    log.info("RQ-VAE training done in %.1fs", time.time() - t0)

    out = Path(args.codebook_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_steps=args.steps,
        seed=args.seed,
        n_train=len(embs),
    )
    if out.suffix == ".safetensors":
        save_rqvae_safetensors(out, enc_params, dec_params, codebooks, **save_kwargs)
    else:
        save_rqvae_npz(out, enc_params, dec_params, codebooks, **save_kwargs)

    evaluate_rqvae(
        enc_params, dec_params, codebooks, embs, max_eval=min(100_000, len(embs)), seed=args.seed
    )


def cmd_evaluate_vae(args):
    enc_params, dec_params, codebooks = load_rqvae_npz(args.codebook)
    log.info("loaded RQ-VAE: %s  codebook shape=%s", args.codebook, codebooks.shape)

    embs = np.load(args.training_data)
    if args.eval_size and args.eval_size < len(embs):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(embs), args.eval_size, replace=False)
        embs = embs[idx]

    evaluate_rqvae(enc_params, dec_params, codebooks, embs, max_eval=len(embs), seed=args.seed)


def main():
    ap = argparse.ArgumentParser(description="JAX RQ-KMeans / RQ-VAE codebook trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train RQ-KMeans codebook (vanilla Lloyd's)")
    p_train.add_argument("--training-data", required=True, help=".npy embeddings, [N, D] float32")
    p_train.add_argument("--codebook-out", required=True, help="Output .npz path")
    p_train.add_argument("--sample-size", type=int, default=None, help="Subsample training data")
    p_train.add_argument("--num-levels", type=int, default=6)
    p_train.add_argument("--num-centroids", type=int, default=256)
    p_train.add_argument("--iters", type=int, default=25)
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("evaluate", help="Evaluate RQ-KMeans codebook quality")
    p_eval.add_argument("--codebook", required=True, help=".npz")
    p_eval.add_argument("--training-data", required=True, help=".npy")
    p_eval.add_argument("--eval-size", type=int, default=100_000)
    p_eval.add_argument("--seed", type=int, default=42)

    p_tvae = sub.add_parser(
        "train-vae", help="Train RQ-VAE (MLP encoder + EMA codebooks + MLP decoder)"
    )
    p_tvae.add_argument("--training-data", required=True, help=".npy embeddings, [N, D] float32")
    p_tvae.add_argument(
        "--codebook-out", required=True, help="Output .npz path (includes enc/dec params)"
    )
    p_tvae.add_argument("--sample-size", type=int, default=None, help="Subsample training data")
    p_tvae.add_argument("--num-levels", type=int, default=6)
    p_tvae.add_argument("--num-centroids", type=int, default=256)
    p_tvae.add_argument("--latent-dim", type=int, default=64, help="Encoder output / codebook dim")
    p_tvae.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden layer width")
    p_tvae.add_argument("--steps", type=int, default=5000, help="Total training steps")
    p_tvae.add_argument("--batch-size", type=int, default=4096)
    p_tvae.add_argument(
        "--warmup-steps", type=int, default=500, help="Autoencoder-only warmup steps"
    )
    p_tvae.add_argument("--lr", type=float, default=3e-4)
    p_tvae.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    p_tvae.add_argument(
        "--ema-decay", type=float, default=0.99, help="EMA decay for codebook update"
    )
    p_tvae.add_argument(
        "--revival-every", type=int, default=200, help="Dead code revival interval (steps)"
    )
    p_tvae.add_argument("--seed", type=int, default=42)

    p_evae = sub.add_parser(
        "evaluate-vae", help="Evaluate RQ-VAE quality (encode + quantize + decode)"
    )
    p_evae.add_argument("--codebook", required=True, help=".npz (RQ-VAE format)")
    p_evae.add_argument("--training-data", required=True, help=".npy")
    p_evae.add_argument("--eval-size", type=int, default=100_000)
    p_evae.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "train-vae": cmd_train_vae,
        "evaluate-vae": cmd_evaluate_vae,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
