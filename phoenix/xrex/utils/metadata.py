# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import subprocess
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from serde import serde
from serde.json import from_dict, from_json, to_json

from xrex.utils.launch_env import CHECKPOINT_DIR, XAI_USER

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

ORBAX_TMP_DIR_SUFFIX = ".orbax-checkpoint-tmp-"


_ELAPSED_SAMPLES = "elapsed_samples"
_ELAPSED_TOKENS = "elapsed_tokens"
_CKPT_INDEX = "checkpoint_index"
_CKPT_EXPIRY = "checkpoint_expiry"
_STEP = "step"

COMPLETED_FILENAME = "completed"
METADATA_FILENAME = "metadata.json"

MetadataFromCheckpoint = namedtuple(
    "MetadataFromCheckpoint",
    [_ELAPSED_SAMPLES, _CKPT_INDEX, _ELAPSED_TOKENS, _STEP],
)


def checkpoint_dir_or_default(checkpoint_dir: str | None):
    if checkpoint_dir is not None:
        return checkpoint_dir
    return os.getenv("CHECKPOINT_DIR", CHECKPOINT_DIR)


def metadata_file_exists(checkpoint_path: Path):
    metadata_path = checkpoint_path / METADATA_FILENAME
    completion_path = checkpoint_path / COMPLETED_FILENAME
    return metadata_path.exists() or completion_path.exists()


def read_checkpoint_metadata(metadata_path: Path) -> MetadataFromCheckpoint:
    record = json.loads(metadata_path.read_text())
    return MetadataFromCheckpoint(
        record[_ELAPSED_SAMPLES],
        record[_CKPT_INDEX],
        record[_ELAPSED_TOKENS],
        record.get(_STEP),
    )


def write_checkpoint_metadata(
    metadata_path: Path,
    elapsed_samples: int,
    checkpoint_index: int,
    checkpoint_expiry: str,
    elapsed_tokens: int | None,
    step: int,
) -> None:
    with metadata_path.open("x") as f:
        json.dump(
            {
                _ELAPSED_SAMPLES: elapsed_samples,
                _CKPT_INDEX: checkpoint_index,
                _ELAPSED_TOKENS: elapsed_tokens,
                _CKPT_EXPIRY: checkpoint_expiry,
                _STEP: step,
            },
            f,
        )


def read_metadata_file(checkpoint_path: Path) -> MetadataFromCheckpoint | None:
    metadata_path = checkpoint_path / METADATA_FILENAME
    if metadata_path.exists():
        try:
            return read_checkpoint_metadata(metadata_path)
        except Exception as e:
            logger.error(f"Unable to read checkpoint completion file: {str(e)}")

    completion_path = checkpoint_path / COMPLETED_FILENAME
    try:
        elapsed_samples = completion_path.read_text()
        checkpoint_index = None
        elapsed_tokens = None
        step = None
        return MetadataFromCheckpoint(elapsed_samples, checkpoint_index, elapsed_tokens, step)
    except Exception as e:
        logger.error(f"Unable to read checkpoint completion file: {str(e)}")
        return None


def write_metadata_file(
    checkpoint_path: Path,
    elapsed_samples: int,
    checkpoint_index: int,
    checkpoint_expiry: str,
    elapsed_tokens: int | None,
    step: int,
):
    write_checkpoint_metadata(
        checkpoint_path / METADATA_FILENAME,
        elapsed_samples,
        checkpoint_index,
        checkpoint_expiry,
        elapsed_tokens,
        step,
    )

    with (checkpoint_path / COMPLETED_FILENAME).open("x") as f:
        f.write(str(elapsed_samples))


@serde
@dataclass
class Run:
    name: str
    run_id: str
    parent_id: Optional[str]
    fork_id: str
    lineage_id: str
    user: str
    commit_hash: str
    created_at: datetime.datetime
    reuse_run_id: bool = False
    config_name: Optional[str] = None

    @classmethod
    def new_from_scratch(cls, name: str, reuse_run_id: bool = False) -> Run:
        run_id = (
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:16] + "-0"
            if reuse_run_id
            else _new_run_id()
        )
        return cls(
            name=name,
            run_id=run_id,
            parent_id=None,
            fork_id=run_id,
            lineage_id=run_id,
            user=XAI_USER,
            commit_hash=_commit_hash(),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            reuse_run_id=reuse_run_id,
        )

    def new_child(self, name: str, reuse_run_id: bool = False) -> Run:
        run_id = (
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:16] + "-0"
            if reuse_run_id
            else _new_run_id()
        )

        if name != self.name:
            fork_id = run_id
        else:
            fork_id = self.fork_id

        return Run(
            name=name,
            run_id=run_id,
            parent_id=self.run_id,
            fork_id=fork_id,
            lineage_id=self.lineage_id,
            user=XAI_USER,
            commit_hash=_commit_hash(),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            reuse_run_id=reuse_run_id,
        )

    def new_restart(self, reuse_run_id: bool = False) -> Run:
        return (
            self
            if reuse_run_id
            else Run(
                name=self.name,
                run_id=_increment_restart_count(self.run_id),
                parent_id=self.run_id,
                fork_id=self.fork_id,
                lineage_id=self.lineage_id,
                user=self.user,
                commit_hash=self.commit_hash,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                reuse_run_id=reuse_run_id,
            )
        )

    def checkpoint_path(self, base_dir: str | None, elapsed_samples: int) -> str:
        base_dir = checkpoint_dir_or_default(base_dir)
        return os.path.join(
            base_dir, self.name, f"elapsed_samples_{elapsed_samples:018d}", self.run_id
        )

    def run_id_parts(self) -> tuple[str, int]:
        return _into_parts(self.run_id)


class LoadType(Enum):
    RESTART = "restart"
    MANUAL_LOAD = "manual_load"


@serde
@dataclass
class CheckpointMeta:
    run_id: str
    elapsed_samples: int
    checkpoint_index: int | None
    path: str
    load_type: LoadType
    source: str
    format: str
    elapsed_tokens: int = 0

    def is_manual_load(self) -> bool:
        if self.load_type == LoadType.MANUAL_LOAD:
            return True
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return from_dict(cls, data)


@runtime_checkable
class Jsonable(Protocol):
    def to_json(self) -> str: ...


def guess_checkpoint_format(path):
    if (path / "orbax-ckpt").exists():
        return "orbax"
    if (path / "ckpt-0" / "tensor00000_000").exists():
        return "pickle"
    if any(ORBAX_TMP_DIR_SUFFIX not in p.name for p in path.glob("orbax-ckpt*")):
        return "orbax"
    raise ValueError(f"Could not determine format of checkpoint at {path}")


def has_committed_payload(checkpoint_path: Path) -> bool:
    if (checkpoint_path / "ckpt-0" / "tensor00000_000").exists():
        return True
    final_names = set()
    tmp_names = []
    for entry in checkpoint_path.glob("orbax-ckpt*"):
        if ORBAX_TMP_DIR_SUFFIX in entry.name:
            tmp_names.append(entry.name)
        else:
            final_names.add(entry.name)
    if not final_names:
        return False
    return all(name.split(ORBAX_TMP_DIR_SUFFIX, 1)[0] in final_names for name in tmp_names)


def _is_loadable_checkpoint(checkpoint_path: Path) -> bool:
    if not (checkpoint_path / COMPLETED_FILENAME).exists():
        return False
    if has_committed_payload(checkpoint_path):
        return True
    logger.warning(
        "Ignoring checkpoint at %s: it has a %r marker but its Orbax data was never"
        " committed (crash between the marker write and the tmp-dir rename?)",
        checkpoint_path,
        COMPLETED_FILENAME,
    )
    return False


class MetadataProvider(ABC):
    @abstractmethod
    def discover_checkpoint(
        self, run_name: Optional[str], load: Optional[str] = None
    ) -> Optional[CheckpointMeta]: ...

    @abstractmethod
    def get_run_info(self, checkpoint: CheckpointMeta) -> Run: ...

    @abstractmethod
    def record_run(self, run: Run, config: Jsonable): ...

    @abstractmethod
    def record_checkpoint(
        self,
        run: Run,
        base_dir: str | None,
        elapsed_samples: int,
        checkpoint_index: int,
        checkpoint_ttl: int,
        elapsed_tokens: int,
        step: int,
    ): ...


PATH1_RE = re.compile(r"elapsed_(samples|tokens)_(?P<samples>\d+)")
PATH2_RE = re.compile(r"(?P<run_id>[A-Za-z0-9_-]+)")


def _matching_subdirs(search_dir: Path, pattern):
    for entry in os.scandir(search_dir):
        if not entry.is_dir:
            continue
        if m := pattern.match(entry.name):
            yield search_dir / entry.name, m


def _search_for_latest_path_manual(search_dir: Path):
    search_dir = search_dir.resolve()

    if not search_dir.exists() or not search_dir.is_dir():
        return None

    m1 = PATH1_RE.fullmatch(search_dir.name)
    if m1 is None:
        m1 = PATH1_RE.fullmatch(search_dir.parent.name)
        m2 = PATH2_RE.fullmatch(search_dir.name)
        if m1 and m2 and _is_loadable_checkpoint(search_dir):
            return m2["run_id"], search_dir, int(m1["samples"])
        return None

    candidates: list[tuple[int, str, Path]] = []
    for candidate, m2 in _matching_subdirs(search_dir, PATH2_RE):
        if _is_loadable_checkpoint(candidate):
            candidates.append((int(m1["samples"]), m2["run_id"], candidate))

    if candidates:
        elapsed_samples, run_id, path = max(candidates)
        return run_id, path, elapsed_samples
    return None


def _search_for_latest_path(search_dir: Path) -> Optional[tuple[str, Path, int]]:
    if not search_dir.exists() or not search_dir.is_dir():
        return None

    logger.info("Looking for checkpoints in %s", search_dir)

    if (search_dir / "latest").exists():
        path = (search_dir / "latest").resolve()
        m2 = PATH2_RE.match(path.name)
        m1 = PATH1_RE.match(path.parent.name)
        if m1 and m2 and _is_loadable_checkpoint(path):
            logger.info("Found via symlink %s", path)
            return m2["run_id"], path, int(m1["samples"])

    candidates: list[tuple[int, str, Path]] = []
    for path1, m1 in _matching_subdirs(search_dir, PATH1_RE):
        for candidate, m2 in _matching_subdirs(path1, PATH2_RE):
            if _is_loadable_checkpoint(candidate):
                candidates.append((int(m1["samples"]), m2["run_id"], candidate))

    if candidates:
        elapsed_samples, run_id, path = max(candidates)
        return run_id, path, elapsed_samples
    return None


class FileSystemProvider(MetadataProvider):
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.current_config: Optional[Jsonable] = None

    def discover_checkpoint(
        self, run_name: Optional[str], load: Optional[str] = None
    ) -> Optional[CheckpointMeta]:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        found = None

        if run_name is not None:
            load_type = LoadType.RESTART
            run_dir = self.checkpoint_dir / run_name
            found = _search_for_latest_path(run_dir)

        if found is None and load is not None:
            load_type = LoadType.MANUAL_LOAD
            found = _search_for_latest_path_manual(Path(load))

        if found is None:
            return None

        run_id, path, elapsed_samples = found

        completion_data = read_metadata_file(path)
        checkpoint_index = None
        if completion_data is not None and completion_data[1] is not None:
            checkpoint_index = completion_data[1]
        elapsed_tokens = 0
        if completion_data is not None:
            elapsed_tokens = completion_data.elapsed_tokens or 0

        return CheckpointMeta(
            run_id=run_id,
            elapsed_samples=elapsed_samples,
            checkpoint_index=checkpoint_index,
            path=str(path),
            load_type=load_type,
            source="filesystem",
            format=guess_checkpoint_format(path),
            elapsed_tokens=elapsed_tokens,
        )

    def get_run_info(self, checkpoint: CheckpointMeta) -> Run:
        meta_file = Path(checkpoint.path) / "run.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"A completed checkpoint was found at {checkpoint.path}, "
                f"but is missing metadata {meta_file.name!r}"
            )
        with meta_file.open("r") as f:
            data = f.read()
        return from_json(Run, data)

    def record_run(self, run: Run, config: Jsonable):
        self.current_config = config

    def record_checkpoint(
        self,
        run: Run,
        base_dir: str | None,
        elapsed_samples: int,
        checkpoint_index: int,
        checkpoint_ttl: int,
        elapsed_tokens: int,
        step: int,
    ):
        base_dir = checkpoint_dir_or_default(base_dir)
        path = Path(run.checkpoint_path(base_dir, elapsed_samples))

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        with (path / "run.json").open("w") as f:
            f.write(to_json(run))

        if self.current_config is not None:
            with (path / "config.json").open("w") as f:
                f.write(self.current_config.to_json())

        checkpoint_expiry = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=checkpoint_ttl)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        write_metadata_file(
            path, elapsed_samples, checkpoint_index, checkpoint_expiry, elapsed_tokens, step
        )

        rpath = os.path.join(base_dir, run.name)
        symlink = os.path.join(rpath, "latest")
        if os.path.islink(symlink):
            os.remove(symlink)
        try:
            os.symlink(str(path.relative_to(rpath)), symlink)
        except FileExistsError:
            logger.warn("Did not symlink %s as %s because the latter exists", path, symlink)


@dataclass
class MetadataManager:
    providers: list[MetadataProvider]
    __init_load: Optional[str] = field(init=False, default=None)
    __init_config: Optional[Jsonable] = field(init=False, default=None)
    __current_run: Optional[Run] = field(init=False, default=None)
    __checkpoint: Optional[CheckpointMeta] = field(init=False, default=None)

    def init_or_restart(
        self, run_name: str, config: Jsonable, load: Optional[str] = None
    ) -> tuple[Run, Optional[CheckpointMeta]]:
        if self.__current_run is None:
            return self.init_run(run_name, config, load)
        return self.restart_run()

    def init_run(
        self, run_name: str, config: Jsonable, load: Optional[str] = None
    ) -> tuple[Run, Optional[CheckpointMeta]]:
        self.__init_load = load
        self.__init_config = config
        found = self.discover_checkpoint(run_name, load)

        if found is not None:
            checkpoint, provider = found
            rank_logger.info(f"Discovered checkpoint to load:\n{checkpoint}")
            parent_run = provider.get_run_info(checkpoint)
            self.__current_run = parent_run.new_child(
                run_name, reuse_run_id=getattr(self.__init_config, "reuse_run_id", False)
            )
            self.__checkpoint = checkpoint
            self._record_run(self.__current_run, config)
            return self.__current_run, checkpoint

        rank_logger.info(f"No checkpoint found for run {run_name!r}, starting from scratch")
        self.__current_run = Run.new_from_scratch(
            run_name, reuse_run_id=getattr(self.__init_config, "reuse_run_id", False)
        )
        self._record_run(self.__current_run, config)
        return self.__current_run, None

    def checkpoint(self) -> Optional[CheckpointMeta]:
        return self.__checkpoint

    def discover_checkpoint(
        self, run_name: Optional[str] = None, load: Optional[str] = None
    ) -> Optional[tuple[CheckpointMeta, MetadataProvider]]:
        if run_name is None and load is None:
            raise ValueError(
                "Either `run_name` or `load` need to be specified to discover a checkpoint."
            )
        found = []
        for provider in self.providers:
            checkpoint = provider.discover_checkpoint(run_name=run_name, load=load)
            if checkpoint is not None:
                found.append((checkpoint, provider))

        if found:
            latest = sorted(found, key=lambda c: c[0].elapsed_samples)[-1]
            return latest

        if load is not None:
            raise FileNotFoundError(f"User specified checkpoint load={load} not found!")
        return None

    def restart_run(self) -> tuple[Run, Optional[CheckpointMeta]]:
        if self.__current_run is None:
            raise RuntimeError("No run was initialized, did you call `init_run` first?")
        if self.__init_config is None:
            raise RuntimeError("No train config was initialized, did you call `init_run` first?")

        run_name = self.__current_run.name
        load = self.__init_load

        for provider in self.providers:
            checkpoint = provider.discover_checkpoint(run_name=run_name, load=load)
            if checkpoint is not None:
                rank_logger.info(f"Discovered checkpoint to load: {checkpoint}")
                if checkpoint.load_type == LoadType.RESTART:
                    self.__current_run = self.__current_run.new_restart(
                        reuse_run_id=getattr(self.__init_config, "reuse_run_id", False)
                    )
                else:
                    parent_run = provider.get_run_info(checkpoint)
                    self.__current_run = parent_run.new_child(
                        run_name, reuse_run_id=getattr(self.__init_config, "reuse_run_id", False)
                    )
                self.__checkpoint = checkpoint
                self._record_run(self.__current_run, self.__init_config)
                return self.__current_run, checkpoint

        rank_logger.info(f"No checkpoint found on restart for {run_name!r}, starting from scratch")
        self.__current_run = Run.new_from_scratch(
            run_name, reuse_run_id=getattr(self.__init_config, "reuse_run_id", False)
        )
        self._record_run(self.__current_run, self.__init_config)
        return self.__current_run, None

    def record_completed_checkpoint(
        self,
        base_dir: str | None,
        elapsed_samples: int,
        checkpoint_index: int,
        elapsed_tokens: int,
        step: int,
        checkpoint_ttl: int = datetime.timedelta(weeks=2).total_seconds(),
    ):
        from xrex.utils.toolbox_notify import notify_checkpoint_async

        if self.__current_run is None:
            raise RuntimeError("No run was initialized, did you call `init_run` first?")
        run = self.__current_run
        for provider in self.providers:
            provider.record_checkpoint(
                run,
                base_dir,
                elapsed_samples=elapsed_samples,
                checkpoint_index=checkpoint_index,
                checkpoint_ttl=checkpoint_ttl,
                elapsed_tokens=elapsed_tokens,
                step=step,
            )
        path = run.checkpoint_path(base_dir, elapsed_samples)
        rank_logger.info(f"Recorded checkpoint {elapsed_samples=} {path=}")
        notify_checkpoint_async(path)

    def _record_run(self, run: Run, config: Jsonable):
        for provider in self.providers:
            provider.record_run(run, config)


def build_metadata_manager(checkpoint_dir: str | None) -> MetadataManager:
    checkpoint_dir = checkpoint_dir_or_default(checkpoint_dir)
    providers: list[MetadataProvider] = [
        FileSystemProvider(checkpoint_dir=checkpoint_dir),
    ]
    return MetadataManager(providers)


def _new_run_id() -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")
    chars = string.ascii_uppercase + string.digits
    rand_chars = "".join(random.choice(chars) for _ in range(8))
    restart_count = 0
    return f"{ts}{rand_chars}-{restart_count}"


def _increment_restart_count(run_id: str) -> str:
    base, restart_count = _into_parts(run_id)
    return f"{base}-{restart_count + 1}"


def _into_parts(run_id: str) -> tuple[str, int]:
    base, restart_count = run_id.split("-")
    return base, int(restart_count)


def _git_bin() -> str:
    bin_path = os.getenv("GIT_BIN_PATH")
    if bin_path:
        p = Path(bin_path)
        return str(p.resolve()) if not p.is_absolute() else bin_path
    return "git"


def _commit_hash() -> str:
    from xrex.utils.launch_env import read_experiment_git_commit_hash

    kube_commit = read_experiment_git_commit_hash()
    if kube_commit:
        return kube_commit

    xai_root = os.getenv("XAI_ROOT")

    git = _git_bin()
    try:
        out = subprocess.check_output([git, "rev-parse", "HEAD"], text=True, cwd=xai_root)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "<unknown>"
    else:
        commit_hash = out[:7]

    try:
        proc = subprocess.run([git, "diff", "--quiet"], cwd=xai_root)
    except (FileNotFoundError, OSError):
        return commit_hash
    dirty = "-dirty" if proc.returncode else ""
    return f"{commit_hash}{dirty}"
