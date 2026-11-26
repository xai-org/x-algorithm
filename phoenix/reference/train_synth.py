#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT_ROOT = HERE.parent

_DEFAULT_CONFIG = "home_direct_packed_nano_offline_kafka_dump"
_DEFAULT_STEPS = 500
_DEFAULT_OUT = "./checkpoints"
_VALID_BATCHES_FILE = ".valid_batches.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="train_synth.py",
        description=(
            "Train the nano recsys model on a locally generated synthetic dump "
            "(no Kafka / cluster / production data)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data",
        required=True,
        metavar="DIR",
        help=(
            "path to the synthetic dump directory produced by dump_gen.py "
            "(must contain %s + partition=*/ parquet files)." % _VALID_BATCHES_FILE
        ),
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        metavar="N",
        help="number of training steps (max_steps). Default: %(default)s.",
    )
    ap.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="NAME",
        help="shipped trainer config name to launch. Default: %(default)s.",
    )
    ap.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        metavar="DIR",
        help=(
            "checkpoint output directory (relative recommended, keeps the run "
            "host-agnostic). Default: %(default)s."
        ),
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "write per-step training metrics (loss included) to "
            "<OUT>/run/metrics.jsonl, one JSON object per step (sets the "
            "trainer's enable_metrics_file)."
        ),
    )
    return ap.parse_args(argv)


def _read_num_partitions(data_dir: Path) -> int:
    vb_path = data_dir / _VALID_BATCHES_FILE
    if not vb_path.is_file():
        raise SystemExit(
            f"error: {vb_path} not found. Generate a dump first, e.g.:\n"
            f"    python reference/dump_gen.py --out {data_dir} "
            f"--seed 20260721 --partitions 4"
        )
    try:
        meta = json.loads(vb_path.read_text())
        return int(meta["num_partitions"])
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"error: could not read 'num_partitions' from {vb_path}: {exc}") from exc


def _count_dump_rows(data_dir: Path) -> int:
    import pyarrow.parquet as pq

    total = 0
    for f in sorted(data_dir.glob("partition=*/*/batch_*.parquet")):
        total += pq.ParquetFile(f).metadata.num_rows
    return total


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    data_dir = Path(args.data).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"error: --data directory does not exist: {data_dir}")
    num_partitions = _read_num_partitions(data_dir)

    for path in (str(EXPORT_ROOT), str(HERE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from xrex.driver.run import pin_visible_devices, run_driver_train

    pin_visible_devices(1)

    from xrex.configs.xrecsys import CONFIGS as _RANKING_CONFIGS
    from xrex.configs.xrecsys_two_tower import CONFIGS as _RETRIEVAL_CONFIGS

    CONFIGS = {**_RANKING_CONFIGS, **_RETRIEVAL_CONFIGS}

    if args.config not in CONFIGS:
        available = "\n  ".join(sorted(CONFIGS))
        raise SystemExit(
            f"error: unknown --config {args.config!r}. Available configs:\n  {available}"
        )
    cfg = CONFIGS[args.config]

    cfg.dataset.path = str(data_dir.resolve())
    cfg.dataset.num_kafka_partitions = num_partitions

    bs = int(cfg.bs_per_device)
    available_steps = max(1, _count_dump_rows(data_dir) // bs - num_partitions)
    steps = int(args.steps)
    if steps > available_steps:
        print(
            f"train_synth: NOTE --steps {steps} exceeds the dump's capacity; "
            f"clamping to {available_steps} steps "
            f"(dump rows // batch_size {bs}, minus a reader margin). "
            f"Generate a larger dump (dump_gen.py --num-rows) for longer runs."
        )
        steps = available_steps
    cfg.max_steps = steps

    ckpt = cfg.checkpoint_config
    ckpt.checkpoint_dir = str(Path(args.out).resolve())
    ckpt.checkpoint_every_n = max(1, steps // 2)
    ckpt.save_final_checkpoint = True
    ckpt.copy_port = 0

    cfg.evals = []

    if args.metrics:
        cfg.enable_metrics_file = True

    print(
        f"train_synth: config={args.config} data={data_dir} "
        f"num_kafka_partitions={num_partitions} max_steps={cfg.max_steps} "
        f"checkpoint_dir={ckpt.checkpoint_dir} "
        f"(checkpoint_every_n={ckpt.checkpoint_every_n}, save_final=True)"
        + (f" metrics={ckpt.checkpoint_dir}/run/metrics.jsonl" if args.metrics else "")
    )

    run_driver_train(None, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
