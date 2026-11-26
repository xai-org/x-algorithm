# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GENERATION_RE = re.compile(r"_(\d{10})\.parquet$")
META_FILENAME = "meta.json"
STAGED_ARRAYS = (
    "post_ids",
    "author_ids",
    "dataset_types",
    "author_hashes",
    "item_hashes",
    "post_sids_u16",
)
_HASH_CHUNK_ROWS = 1 << 20


@dataclasses.dataclass(frozen=True)
class StagerConfig:
    root: str
    dataset_names: tuple[str, ...]
    max_posts: int
    read_post_sid: bool
    use_post_sid: bool
    sid_num_levels: int
    hash_table: dict
    keep_generations: int = 3
    poll_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_posts <= 0 or self.sid_num_levels <= 0:
            raise ValueError("corpus rows and SID levels must be positive")
        if self.keep_generations <= 0 or self.poll_seconds <= 0:
            raise ValueError("retention and poll interval must be positive")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> StagerConfig:
        raw = json.loads(payload)
        raw["dataset_names"] = tuple(raw["dataset_names"])
        return cls(**raw)

    def fingerprint(self) -> str:
        identity = dataclasses.asdict(self)
        for runtime_only in ("root", "keep_generations", "poll_seconds"):
            identity.pop(runtime_only)
        blob = json.dumps(identity, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def staging_dir(self) -> Path:
        return Path(self.root) / f"cfg_{self.fingerprint()}"


def hash_table_config(hash_table: Any) -> dict:
    keys = hash_table.hash_keys
    return {
        "output_vocab_size": int(hash_table.output_vocab_size),
        "hash_keys": {
            field.name: getattr(keys, field.name)
            for field in dataclasses.fields(keys)
            if not field.name.startswith("_")
        },
    }


def _hash_table_from_config(config: Mapping) -> Any:
    from xrex.models.recsys_embedding import HashKeys, HashTable

    return HashTable(
        hash_keys=HashKeys(**config["hash_keys"]),
        output_vocab_size=config["output_vocab_size"],
    )


def dataset_logical_paths(dataset_names: Sequence[str]) -> dict[str, str] | None:
    from xrex.data.retrieval_dataset import RetrievalDataset

    paths: dict[str, str] = {}
    for name in dataset_names:
        dataset = RetrievalDataset[name]
        if dataset in RetrievalDataset.home_slices():
            return None
        path = dataset.path
        if path is None or "://" in path:
            return None
        paths[name] = path
    return paths


def pinned_path(logical_path: str, generation: int) -> str:
    base = Path(logical_path)
    return str(base.with_name(f"{base.stem}_{generation}{base.suffix}"))


def select_generation(logical_paths: Mapping[str, str]) -> int | None:
    generations = []
    for logical in logical_paths.values():
        try:
            resolved = os.path.realpath(logical, strict=True)
        except OSError:
            return None
        match = GENERATION_RE.search(os.path.basename(resolved))
        if match is None:
            return None
        generations.append(int(match.group(1)))
    if not generations:
        return None
    generation = min(generations)
    for logical in logical_paths.values():
        if not os.path.isfile(pinned_path(logical, generation)):
            return None
    return generation


def compute_staged_arrays(
    post_ids: np.ndarray,
    author_ids: np.ndarray,
    dataset_types: np.ndarray,
    post_sids_raw: np.ndarray | None,
    hash_table: Any,
    *,
    use_post_sid: bool,
    sid_num_levels: int,
) -> dict[str, np.ndarray]:
    num_rows = len(post_ids)
    author_hashes = np.empty((num_rows, hash_table.hash_keys.num_author_hashes), np.int32)
    item_hashes = np.empty((num_rows, hash_table.hash_keys.num_item_hashes), np.int32)
    for start in range(0, num_rows, _HASH_CHUNK_ROWS):
        end = min(start + _HASH_CHUNK_ROWS, num_rows)
        author_hashes[start:end] = hash_table.get_author_hash(author_ids[start:end])
        item_hashes[start:end] = hash_table.get_item_hash(post_ids[start:end])
    if use_post_sid and post_sids_raw is not None:
        post_sids_u16 = (np.asarray(post_sids_raw, dtype=np.int32) + 1).astype(np.uint16)
    else:
        post_sids_u16 = np.zeros((num_rows, sid_num_levels), dtype=np.uint16)
    return {
        "post_ids": np.ascontiguousarray(post_ids, dtype=np.int64),
        "author_ids": np.ascontiguousarray(author_ids, dtype=np.int64),
        "dataset_types": np.ascontiguousarray(dataset_types, dtype=np.int32),
        "author_hashes": author_hashes,
        "item_hashes": item_hashes,
        "post_sids_u16": post_sids_u16,
    }


def load_generation(config: StagerConfig, generation: int) -> tuple[np.ndarray, ...]:
    from xrex.data.retrieval_dataset import RetrievalDataset

    logical = dataset_logical_paths(config.dataset_names)
    if logical is None:
        raise ValueError(
            "Corpus staging requires local generation-stable datasets; "
            "HOME_COLD/HOME_HOT use the synchronous loader"
        )
    datasets = [RetrievalDataset[name] for name in config.dataset_names]
    paths = {RetrievalDataset[name]: pinned_path(p, generation) for name, p in logical.items()}
    return RetrievalDataset.load_datasets(
        datasets,
        max_posts=config.max_posts,
        read_post_sid=config.read_post_sid,
        sid_num_levels=config.sid_num_levels,
        paths=paths,
    )


def list_staged_generations(staging_dir: Path) -> list[int]:
    if not staging_dir.is_dir():
        return []
    generations = []
    for entry in staging_dir.iterdir():
        match = re.fullmatch(r"gen_(\d+)", entry.name)
        if match and (entry / META_FILENAME).is_file():
            generations.append(int(match.group(1)))
    return sorted(generations)


def write_staged_generation(
    staging_dir: Path, generation: int, arrays: Mapping[str, np.ndarray], meta: Mapping
) -> Path:
    final = staging_dir / f"gen_{generation}"
    if (final / META_FILENAME).is_file():
        return final
    staging_dir.mkdir(parents=True, exist_ok=True)
    tmp = staging_dir / f".gen_{generation}.tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for name in STAGED_ARRAYS:
        np.save(tmp / f"{name}.npy", arrays[name], allow_pickle=False)
    (tmp / META_FILENAME).write_text(json.dumps(dict(meta), sort_keys=True))
    os.rename(tmp, final)
    return final


def open_staged_generation(staging_dir: Path, generation: int) -> dict[str, np.ndarray]:
    directory = staging_dir / f"gen_{generation}"
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in STAGED_ARRAYS
    }


def validate_staged_arrays(
    config: StagerConfig, generation: int, arrays: Mapping[str, np.ndarray], meta: Mapping
) -> None:
    if meta.get("fingerprint") != config.fingerprint() or meta.get("generation") != generation:
        raise ValueError("staged corpus identity differs from requested configuration")
    if meta.get("rows") != config.max_posts:
        raise ValueError("staged corpus row count differs from configured capacity")
    keys = config.hash_table["hash_keys"]
    expected = {
        "post_ids": ((config.max_posts,), np.dtype(np.int64)),
        "author_ids": ((config.max_posts,), np.dtype(np.int64)),
        "dataset_types": ((config.max_posts,), np.dtype(np.int32)),
        "author_hashes": ((config.max_posts, len(keys["author_hash_scales"])), np.dtype(np.int32)),
        "item_hashes": ((config.max_posts, len(keys["item_hash_scales"])), np.dtype(np.int32)),
        "post_sids_u16": ((config.max_posts, config.sid_num_levels), np.dtype(np.uint16)),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"staged {name} has {value.shape}/{value.dtype}, expected {shape}/{dtype}"
            )


def read_staged_meta(staging_dir: Path, generation: int) -> dict:
    return json.loads((staging_dir / f"gen_{generation}" / META_FILENAME).read_text())


def prune_staged_generations(staging_dir: Path, keep: int) -> None:
    for generation in list_staged_generations(staging_dir)[:-keep]:
        shutil.rmtree(staging_dir / f"gen_{generation}", ignore_errors=True)
    if staging_dir.is_dir():
        for entry in staging_dir.iterdir():
            if entry.name.startswith(".gen_") and not entry.name.endswith(f".{os.getpid()}"):
                shutil.rmtree(entry, ignore_errors=True)


def stage_generation(config: StagerConfig, generation: int, hash_table: Any) -> Path:
    start = time.perf_counter()
    post_ids, author_ids, dataset_types, post_sids_raw = load_generation(config, generation)
    loaded = time.perf_counter()
    arrays = compute_staged_arrays(
        post_ids,
        author_ids,
        dataset_types,
        post_sids_raw,
        hash_table,
        use_post_sid=config.use_post_sid,
        sid_num_levels=config.sid_num_levels,
    )
    hashed = time.perf_counter()
    logical = dataset_logical_paths(config.dataset_names) or {}
    meta = {
        "generation": generation,
        "fingerprint": config.fingerprint(),
        "rows": int(len(post_ids)),
        "real_rows": int(np.count_nonzero(post_ids)),
        "files": {name: pinned_path(p, generation) for name, p in logical.items()},
        "load_seconds": loaded - start,
        "hash_seconds": hashed - loaded,
        "staged_at": time.time(),
    }
    final = write_staged_generation(config.staging_dir(), generation, arrays, meta)
    logger.info(
        "Staged retrieval corpus generation %d (%d rows) in %.2fs "
        "(load %.2fs, hash %.2fs, write %.2fs) at %s",
        generation,
        meta["rows"],
        time.perf_counter() - start,
        meta["load_seconds"],
        meta["hash_seconds"],
        time.perf_counter() - hashed,
        final,
    )
    return final


def poll_once(config: StagerConfig, hash_table: Any) -> int | None:
    logical = dataset_logical_paths(config.dataset_names)
    if logical is None:
        return None
    generation = select_generation(logical)
    if generation is None:
        return None
    staging_dir = config.staging_dir()
    if generation not in list_staged_generations(staging_dir):
        stage_generation(config, generation, hash_table)
    prune_staged_generations(staging_dir, config.keep_generations)
    return generation


def agree_on_generation(
    local_generations: Sequence[int],
    allgather: Callable[[np.ndarray], np.ndarray],
    *,
    window: int = 3,
) -> int | None:
    newest = sorted(local_generations)[-window:]
    for generation in newest:
        if not 0 < generation < 2**31:
            raise ValueError(f"generation {generation} does not fit in int32")
    local = np.full(window, -1, dtype=np.int32)
    local[: len(newest)] = newest
    gathered = np.asarray(allgather(local)).reshape(-1, window)
    common = set(int(g) for g in gathered[0] if g >= 0)
    for row in gathered[1:]:
        common &= set(int(g) for g in row if g >= 0)
    return max(common) if common else None


def run_loop(config: StagerConfig, *, parent_pid: int | None = None) -> None:
    hash_table = _hash_table_from_config(config.hash_table)
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        import pyarrow as pa

        pa.set_cpu_count(8)
        pa.set_io_thread_count(8)
    except Exception:
        logger.exception("could not cap pyarrow threads")
    while True:
        if parent_pid is not None and os.getppid() != parent_pid:
            logger.info("parent %d exited; stopping retrieval corpus stager", parent_pid)
            return
        try:
            poll_once(config, hash_table)
        except Exception:
            logger.exception("retrieval corpus staging failed; retrying next poll")
        time.sleep(config.poll_seconds)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname).1s] retrieval_corpus_stager: %(message)s",
        stream=sys.stderr,
    )
    run_loop(StagerConfig.from_json(args.config_json), parent_pid=args.parent_pid)


if __name__ == "__main__":
    main()
