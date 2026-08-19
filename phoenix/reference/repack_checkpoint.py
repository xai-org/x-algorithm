#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zlib
from collections import namedtuple
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from xai_checkpointing.tree_util import tree_to_dict

_TAG = "orbax-ckpt"
_KEEP_ALWAYS = ("params", "emb_table", "emb_table_state", "rng", "step", "post_embeddings")
_KEEP_OPT = "opt_state"

_METADATA_DROP = (
    "original_cluster",
    "checkpoint_expiry",
    "retention_override",
    "priority",
    "last_touched",
    "soft_delete_ttl",
    "hard_delete_ttl",
    "is_reshared",
    "is_reloaded",
)
_RUN_REDACT = {"user": "public", "commit_hash": "redacted"}

_INTERNAL_ROOTS = ("data", "root", "home", "mnt", "scratch")
_INTERNAL_PREFIXES = tuple(f"/{r}/" for r in _INTERNAL_ROOTS)
_REL_ANCHOR = "phoenix_index"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _cpu_device():
    return jax.devices("cpu")[0]


def _restore_full(src_orbax: Path) -> dict:
    return ocp.StandardCheckpointer().restore(src_orbax)


def _reduce_tree(full: dict, keep_opt_state: bool) -> dict:
    sh = jax.sharding.SingleDeviceSharding(_cpu_device())
    keep = list(_KEEP_ALWAYS) + ([_KEEP_OPT] if keep_opt_state else [])

    def prune(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                pv = prune(v)
                if pv is not None:
                    out[k] = pv
            return out or None
        if node is None:
            return None
        return jax.device_put(np.asarray(node), sh)

    reduced = {}
    for k in keep:
        if k not in full:
            continue
        pv = prune(full[k])
        if pv is not None:
            reduced[k] = pv
    if "params" not in reduced:
        raise RuntimeError("source checkpoint has no 'params' subtree; is --src an orbax dir?")
    return reduced


class _NoCompressionArrayHandler(ocp.type_handlers.ArrayHandler):
    def _get_json_tspec_write(self, *args, **kwargs):
        spec = super()._get_json_tspec_write(*args, **kwargs)
        for codec in spec["metadata"]["codecs"]:
            cfg = codec["configuration"]
            cfg["codecs"] = [c for c in cfg["codecs"] if c["name"] != "zstd"]
        return spec


def _save_reduced(reduced: dict, dst_orbax: Path) -> None:
    if not hasattr(jax._src.config, "enable_memories"):
        _FakeState = namedtuple("_FakeState", ["value"])
        jax._src.config.enable_memories = _FakeState(value=True)
    ocp.type_handlers.register_type_handler(jax.Array, _NoCompressionArrayHandler(), override=True)
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler(use_zarr3=True), 900)
    save_args = jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=None), reduced)
    checkpointer.save(dst_orbax, args=ocp.args.PyTreeSave(reduced, save_args=save_args))
    checkpointer.wait_until_finished()
    checkpointer.close()


_ONES_MESH = {"stage": 1, "expert": 1, "replica": 1, "data": 1, "seq": 1, "model": 1}


def _checksum_dict(reduced: dict) -> dict:
    flat = tree_to_dict(reduced)
    shapes, dtypes, shardings, shard_ck, global_ck = {}, {}, {}, {}, {}
    for name, arr in flat.items():
        if arr is None:
            continue
        a = np.asarray(arr)
        adler = zlib.adler32(a.tobytes()) & 0xFFFFFFFF
        shapes[name] = list(a.shape)
        dtypes[name] = a.dtype.str
        shardings[name] = {"spec": [None] * a.ndim, "mesh": dict(_ONES_MESH)}
        shard_ck[name] = [adler]
        global_ck[name] = adler
    return {
        "version": 3,
        "format": {"shard_checksums": "adler32", "global_checksums": "global_adler32"},
        "shapes": shapes,
        "dtypes": dtypes,
        "shardings": shardings,
        "shard_checksums": shard_ck,
        "global_checksums": global_ck,
    }


def _scrub_path_value(value):
    if not isinstance(value, str) or not value.startswith(_INTERNAL_PREFIXES):
        return value
    if _REL_ANCHOR in value:
        return value[value.index(_REL_ANCHOR) :]
    return os.path.basename(value.rstrip("/"))


def _scrub_json_paths(node):
    if isinstance(node, dict):
        return {k: _scrub_json_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub_json_paths(v) for v in node]
    return _scrub_path_value(node)


def _scrub_sidecars(src: Path, dst: Path) -> list[str]:
    actions = []

    cfg = src / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        (dst / "config.json").write_text(json.dumps(_scrub_json_paths(data), indent=2))
        actions.append("config.json: internal fs paths -> relative")

    run = src / "run.json"
    if run.exists():
        data = json.loads(run.read_text())
        redacted = [k for k in _RUN_REDACT if k in data]
        for k in redacted:
            data[k] = _RUN_REDACT[k]
        (dst / "run.json").write_text(json.dumps(data, indent=2))
        actions.append(f"run.json: redacted {redacted}")

    meta = src / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        dropped = [k for k in _METADATA_DROP if k in data]
        for k in dropped:
            data.pop(k, None)
        (dst / "metadata.json").write_text(json.dumps(data, indent=2))
        actions.append(f"metadata.json: dropped {dropped}")

    completed = src / "completed"
    if completed.exists():
        shutil.copyfile(completed, dst / "completed")
        actions.append("completed: copied")

    if (src / "nfs_scan.json").exists():
        actions.append("nfs_scan.json: DROPPED")

    return actions


def repack(src: Path, dst: Path, keep_opt_state: bool) -> None:
    src_orbax = src / _TAG
    if not src_orbax.is_dir():
        raise FileNotFoundError(f"no {_TAG}/ under --src {src}")
    dst.mkdir(parents=True, exist_ok=True)
    dst_orbax = dst / _TAG
    if dst_orbax.exists():
        raise FileExistsError(f"{dst_orbax} already exists; delete --dst and rerun")

    print(f"[repack] loading full checkpoint from {src_orbax}", flush=True)
    full = _restore_full(src_orbax)
    print(f"[repack] source top-level keys: {sorted(full.keys())}", flush=True)

    reduced = _reduce_tree(full, keep_opt_state)
    kept = sorted(reduced.keys())
    n_tensors = len([v for v in tree_to_dict(reduced).values() if v is not None])
    print(f"[repack] keeping {kept} ({n_tensors} tensors)", flush=True)

    _save_reduced(reduced, dst_orbax)
    print(f"[repack] re-saved reduced tree via trainer orbax writer -> {dst_orbax}", flush=True)

    ck = _checksum_dict(reduced)
    (dst / "checksums.0.json").write_text(json.dumps(ck))
    print(
        f"[repack] regenerated checksums.0.json over {len(ck['shard_checksums'])} tensors",
        flush=True,
    )

    for action in _scrub_sidecars(src, dst):
        print(f"[repack] scrub: {action}", flush=True)

    src_size, dst_size = _dir_size(src), _dir_size(dst)
    pct = 100.0 * (1.0 - dst_size / src_size) if src_size else 0.0
    print(
        f"[repack] DONE  size {src_size / 1e6:.1f} MB -> {dst_size / 1e6:.1f} MB "
        f"({pct:.0f}% smaller)  dst={dst}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="source checkpoint dir (contains orbax-ckpt/)")
    ap.add_argument("--dst", required=True, help="destination dir for the repacked checkpoint")
    ap.add_argument(
        "--keep-opt-state",
        action="store_true",
        help="also keep opt_state (resumable training; ~2x size)",
    )
    args = ap.parse_args(argv)
    repack(Path(args.src), Path(args.dst), args.keep_opt_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
