# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EXTRA_CODEBOOK_LOADERS: list = []


def mlp_forward_np(params: dict, x: np.ndarray) -> np.ndarray:
    n_layers = len([k for k in params if k.startswith("w")])
    for i in range(n_layers - 1):
        x = np.maximum(x @ params[f"w{i}"] + params[f"b{i}"], 0.0)
    return x @ params[f"w{n_layers - 1}"] + params[f"b{n_layers - 1}"]


def load_codebook(path: str | Path) -> tuple[np.ndarray, int, tuple[dict, dict] | None]:
    path = str(path)

    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="np") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        stages = tensors["stages"].astype(np.float32)
        K = int(stages.shape[1])
        if meta.get("model_type") == "rqvae":
            enc_params = {
                k[4:]: tensors[k].astype(np.float32) for k in tensors if k.startswith("enc_")
            }
            dec_params = {
                k[4:]: tensors[k].astype(np.float32) for k in tensors if k.startswith("dec_")
            }
            return stages, K, (enc_params, dec_params)
        return stages, K, None

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        stages = z["stages"].astype(np.float32)
        K = int(stages.shape[1])
        if "model_type" in z and str(z["model_type"]) == "rqvae":
            enc_params = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith("enc_")}
            dec_params = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith("dec_")}
            return stages, K, (enc_params, dec_params)
        return stages, K, None

    for predicate, loader in EXTRA_CODEBOOK_LOADERS:
        if predicate(path):
            return loader(path)
    raise ValueError(f"unsupported codebook format: {path} (expected .safetensors or .npz)")


def is_rqvae(path: str | Path) -> bool:
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="np") as f:
            meta = f.metadata() or {}
        return meta.get("model_type") == "rqvae"
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return "model_type" in z and str(z["model_type"]) == "rqvae"
    return False


def save_codebook_npz(
    path: str | Path,
    codebooks: np.ndarray,
    n_iters: int,
    seed: int,
    n_train: int,
):
    M, K, D = codebooks.shape
    np.savez_compressed(
        str(path),
        stages=codebooks.astype(np.float32),
        num_levels=np.array(M, dtype=np.int32),
        num_centroids=np.array(K, dtype=np.int32),
        embedding_dim=np.array(D, dtype=np.int32),
        training_size=np.array(n_train, dtype=np.int64),
        iters=np.array(n_iters, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
    )
    log.info("saved codebook: %s (%d levels x %d centroids x %d)", path, M, K, D)


def save_rqvae_npz(
    path: str | Path,
    enc_params: dict,
    dec_params: dict,
    codebooks: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    num_steps: int,
    seed: int,
    n_train: int,
):
    M, K, d = codebooks.shape
    save_dict = {
        "stages": codebooks.astype(np.float32),
        "model_type": np.array("rqvae", dtype=object),
        "latent_dim": np.array(latent_dim, dtype=np.int32),
        "hidden_dim": np.array(hidden_dim, dtype=np.int32),
        "num_levels": np.array(M, dtype=np.int32),
        "num_centroids": np.array(K, dtype=np.int32),
        "embedding_dim": np.array(np.asarray(enc_params["w0"]).shape[0], dtype=np.int32),
        "training_size": np.array(n_train, dtype=np.int64),
        "num_steps": np.array(num_steps, dtype=np.int32),
        "seed": np.array(seed, dtype=np.int32),
    }
    for k, v in enc_params.items():
        save_dict[f"enc_{k}"] = np.asarray(v)
    for k, v in dec_params.items():
        save_dict[f"dec_{k}"] = np.asarray(v)
    np.savez_compressed(str(path), **save_dict)
    log.info("saved RQ-VAE: %s (latent=%d, %d levels x %d centroids)", path, d, M, K)


def save_codebook_safetensors(
    path: str | Path,
    codebooks: np.ndarray,
    n_iters: int,
    seed: int,
    n_train: int,
):
    from safetensors.numpy import save_file

    M, K, D = codebooks.shape
    tensors = {"stages": codebooks.astype(np.float32)}
    metadata = {
        "model_type": "rq_kmeans",
        "num_levels": str(M),
        "num_centroids": str(K),
        "embedding_dim": str(D),
        "training_size": str(n_train),
        "iters": str(n_iters),
        "seed": str(seed),
    }
    save_file(tensors, str(path), metadata=metadata)
    log.info("saved codebook: %s (%d levels x %d centroids x %d)", path, M, K, D)


def save_rqvae_safetensors(
    path: str | Path,
    enc_params: dict,
    dec_params: dict,
    codebooks: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    num_steps: int,
    seed: int,
    n_train: int,
):
    from safetensors.numpy import save_file

    M, K, d = codebooks.shape
    tensors = {"stages": codebooks.astype(np.float32)}
    for k, v in enc_params.items():
        tensors[f"enc_{k}"] = np.asarray(v).astype(np.float32)
    for k, v in dec_params.items():
        tensors[f"dec_{k}"] = np.asarray(v).astype(np.float32)
    metadata = {
        "model_type": "rqvae",
        "latent_dim": str(latent_dim),
        "hidden_dim": str(hidden_dim),
        "num_levels": str(M),
        "num_centroids": str(K),
        "embedding_dim": str(np.asarray(enc_params["w0"]).shape[0]),
        "training_size": str(n_train),
        "num_steps": str(num_steps),
        "seed": str(seed),
    }
    save_file(tensors, str(path), metadata=metadata)
    log.info("saved RQ-VAE: %s (latent=%d, %d levels x %d centroids)", path, d, M, K)


def assign_sids(embeddings: np.ndarray, codebook_path: str | Path) -> np.ndarray:
    codebook_path = str(codebook_path)
    stages, K, rqvae_params = load_codebook(codebook_path)
    M = stages.shape[0]

    if rqvae_params is not None:
        enc_params, _ = rqvae_params
        log.info("RQ-VAE: %d stages x %d centroids, latent_d=%d", M, K, stages.shape[2])
        t0 = time.time()
        latent = mlp_forward_np(enc_params, embeddings.astype(np.float32))
        N = len(latent)
        sids = np.zeros((N, M), dtype=np.int32)
        residual = latent.copy()
        for s in range(M):
            cb = stages[s]
            c_sq = (cb * cb).sum(axis=1)
            r_sq = (residual * residual).sum(axis=1, keepdims=True)
            dists = r_sq + c_sq[None, :] - 2.0 * residual @ cb.T
            sids[:, s] = dists.argmin(axis=1)
            residual -= cb[sids[:, s]]
        log.info("encoded %d posts via RQ-VAE in %.1fs", N, time.time() - t0)
        return sids

    log.info("RQ: %d stages x %d centroids, d=%d", M, K, stages[0].shape[0] if M > 0 else 0)
    N = len(embeddings)
    sids = np.zeros((N, M), dtype=np.int32)
    residual = embeddings.astype(np.float32).copy()
    BATCH = 100_000

    t0 = time.time()
    for si in range(M):
        centroids = stages[si]
        c_sq = (centroids * centroids).sum(axis=1)
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            batch = residual[start:end]
            dists = (
                (batch * batch).sum(axis=1, keepdims=True) + c_sq[None, :] - 2 * batch @ centroids.T
            )
            assigns = dists.argmin(axis=1)
            sids[start:end, si] = assigns
            residual[start:end] -= centroids[assigns]
        log.info("  stage %d: %.1fs", si, time.time() - t0)

    log.info("encoded %d posts in %.1fs", N, time.time() - t0)
    return sids


if importlib.util.find_spec("sid_io_internal") is not None:
    pass
