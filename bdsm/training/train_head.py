#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

_RUNTIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime")
sys.path.insert(0, os.path.abspath(_RUNTIME))

import heads
import jax
import jax.numpy as jnp
import labels
import loss as task_loss
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
from metrics import average_precision
from task_heads import SEED, head_logits, init_head_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_head")


def load_cached(split_dir: str) -> dict:
    xs, ys, uids, sources = [], [], [], []
    n_nonfinite = 0
    files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no cached cls parquet under {split_dir}")
    for fpath in files:
        t = pq.read_table(fpath)
        strength = np.asarray(t.column("label_strength").to_pylist())
        cls_col = t.column("cls").combine_chunks()
        cls_all = (
            cls_col.flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(len(cls_col), -1)
            .astype(np.float32)
        )
        finite = np.isfinite(cls_all).all(axis=1)
        n_nonfinite += int((~finite).sum())
        keep = (strength == "strong") & finite
        cls = cls_all[keep]
        lbls = [lb for lb, k in zip(t.column("labels").to_pylist(), keep, strict=True) if k]
        xs.append(cls)
        ys.append(np.stack([labels.labels_to_vectors(lb)[0] for lb in lbls]))
        uids.append(np.asarray(t.column("user_id").to_pylist(), dtype=np.int64)[keep])
        sources.extend(
            s for s, k in zip(t.column("label_source").to_pylist(), keep, strict=True) if k
        )
    log.info(f"{split_dir}: dropped {n_nonfinite} rows with non-finite CLS")
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "user_id": np.concatenate(uids),
        "source": np.asarray(sources),
    }


def head_scores(params: dict, x: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp_local

    return np.asarray(
        jax.nn.sigmoid(
            head_logits({k: jnp_local.asarray(v) for k, v in params.items()}, jnp_local.asarray(x))
        )
    )


def evaluate(params: dict, holdout: dict) -> dict:
    scores = head_scores(params, holdout["x"])
    out = {"n_eval_users": len(scores)}
    for h in heads.HEADS:
        out[f"pr_auc/{h.name}"] = average_precision(holdout["y"][:, h.index], scores[:, h.index])
        out[f"pos/{h.name}"] = int(holdout["y"][:, h.index].sum())
    return out


def save_checkpoint(run_dir: str, step: int, params: dict, run_config: dict) -> None:
    step_dir = os.path.join(run_dir, f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    if set(params) == {"w0", "b0"}:
        arrays = {
            "abuse_v3/head/w": np.asarray(params["w0"]),
            "abuse_v3/head/b": np.asarray(params["b0"]),
        }
    else:
        arrays = {f"abuse_v3/head_mlp/{k}": np.asarray(v) for k, v in params.items()}
    np.savez(os.path.join(step_dir, "head_params.npz"), **arrays)
    with open(os.path.join(step_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=1)
    with open(os.path.join(run_dir, "latest.txt"), "w") as f:
        f.write(step_dir)


def write_holdout_scores(path: str, holdout: dict, params: dict) -> None:
    scores = head_scores(params, holdout["x"]).astype(np.float32)
    cols = {
        "user_id": pa.array(holdout["user_id"].tolist(), pa.int64()),
        "labels": pa.array(
            [[h.name for h in heads.HEADS if holdout["y"][i, h.index]] for i in range(len(scores))],
            pa.list_(pa.string()),
        ),
        "label_source": pa.array(holdout["source"].tolist(), pa.string()),
        "label_strength": pa.array(["strong"] * len(scores), pa.string()),
    }
    for h in heads.HEADS:
        cols[f"score_{h.name}"] = pa.array(scores[:, h.index].tolist(), pa.float32())
    pq.write_table(pa.table(cols), path, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls-dir", required=True, help="dir with train/ and holdout/ caches")
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument(
        "--head-hidden-dims",
        default="",
        help="comma-separated hidden widths; empty = linear head; "
        "the shipped configuration passes 512,256 with "
        "--batch-size 256 --peak-lr 1e-3 --steps 20000",
    )
    ap.add_argument(
        "--holdout-scores-out", default=None, help="write final holdout score parquet here"
    )
    args = ap.parse_args()

    with open(os.path.join(args.backbone_dir, "MANIFEST.json")) as f:
        manifest = json.load(f)
    cfg = manifest["source_config"]
    label_smoothing = cfg["label_smoothing"]
    reverse_ce_weight = cfg["reverse_ce_weight"]

    train = load_cached(os.path.join(args.cls_dir, "train"))
    holdout = load_cached(os.path.join(args.cls_dir, "holdout"))
    emb_dim = train["x"].shape[1]
    log.info(
        f"train: {len(train['x'])} users, holdout: {len(holdout['x'])} users, cls dim {emb_dim}"
    )
    overlap = set(train["user_id"].tolist()) & set(holdout["user_id"].tolist())
    assert not overlap, f"train/holdout user overlap in caches: {len(overlap)}"

    hidden_dims = tuple(int(d) for d in args.head_hidden_dims.split(",") if d)
    params = init_head_params(emb_dim, hidden_dims)
    n_trainable = sum(int(np.prod(v.shape)) for v in params.values())
    log.info(
        f"trainable params: {n_trainable:,} (backbone frozen by construction — cached activations)"
    )
    assert n_trainable <= 5_000_000, f"trainable {n_trainable:,} exceeds the 5M spec cap"

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.peak_lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.peak_lr * 0.1,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule))
    opt_state = optimizer.init(params)

    def loss_fn(p, x, y, mask):
        return task_loss.masked_sce_focal_loss(
            head_logits(p, x),
            y,
            mask,
            label_smoothing=label_smoothing,
            reverse_ce_weight=reverse_ce_weight,
            focal_gamma=args.focal_gamma,
        )

    @jax.jit
    def train_step(p, opt_st, x, y, mask):
        (loss_val, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, x, y, mask)
        grad_norm = optax.global_norm(grads)
        updates, new_opt = optimizer.update(grads, opt_st, p)
        return optax.apply_updates(p, updates), new_opt, loss_val, grad_norm, stats

    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    run_config = {
        "run_name": args.run_name,
        "seed": SEED,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "peak_lr": args.peak_lr,
        "warmup_steps": args.warmup_steps,
        "lr_end_ratio": 0.1,
        "grad_clip": 1.0,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": label_smoothing,
        "reverse_ce_weight": reverse_ce_weight,
        "head_hidden_dims": list(hidden_dims),
        "n_trainable_params": int(n_trainable),
        "n_train_users": len(train["x"]),
        "n_holdout_users": len(holdout["x"]),
        "backbone_npz": os.path.join(args.backbone_dir, "backbone.npz"),
        "backbone_sha256": manifest["file_sha256"],
        "head_registry_hash": heads.registry_hash(),
        "head_names": list(heads.HEAD_NAMES),
        "cls_dir": args.cls_dir,
        "training_mode": "cached-cls (frozen backbone activations precomputed once)",
    }
    log.info(json.dumps(run_config, indent=1))

    rng = np.random.default_rng(SEED)
    n = len(train["x"])
    order = rng.permutation(n)
    cursor = 0
    x_all = jnp.asarray(train["x"])
    y_all = jnp.asarray(train["y"].astype(np.float32))
    mask_all = jnp.ones_like(y_all)

    t0 = time.time()
    for step in range(args.steps):
        if cursor + args.batch_size > n:
            order = rng.permutation(n)
            cursor = 0
        idx = jnp.asarray(order[cursor : cursor + args.batch_size])
        cursor += args.batch_size

        params, opt_state, loss_val, grad_norm, stats = train_step(
            params, opt_state, x_all[idx], y_all[idx], mask_all[idx]
        )

        if step == 0 and args.focal_gamma > 0:
            assert "focal_mean_weight" in stats, (
                "focal loss not wired: focal_mean_weight missing from stats"
            )
            log.info(f"step 0: focal_mean_weight={float(stats['focal_mean_weight']):.4f}")

        if step % 500 == 0:
            lv, gn = float(loss_val), float(grad_norm)
            if not (np.isfinite(lv) and np.isfinite(gn)):
                raise RuntimeError(
                    f"non-finite loss/grad at step {step}: loss={lv} grad_norm={gn} — "
                    "stopping for investigation (never zero-and-continue)"
                )
            log.info(
                f"[{step:6d}/{args.steps}] loss={lv:.4f} grad={gn:.3f} "
                f"lr={float(schedule(step)):.2e}"
            )

        if step > 0 and step % args.eval_every == 0:
            params_np = {k: np.asarray(v) for k, v in params.items()}
            m = evaluate(params_np, holdout)
            log.info(
                f"eval@{step}: "
                + json.dumps(
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
                )
            )
            save_checkpoint(run_dir, step, params_np, run_config)

    params_np = {k: np.asarray(v) for k, v in params.items()}
    m = evaluate(params_np, holdout)
    log.info(
        "final eval: "
        + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    )
    save_checkpoint(run_dir, args.steps, params_np, run_config)
    if args.holdout_scores_out:
        write_holdout_scores(args.holdout_scores_out, holdout, params_np)
        log.info(f"holdout scores -> {args.holdout_scores_out}")
    log.info(f"done in {time.time() - t0:.0f}s: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
