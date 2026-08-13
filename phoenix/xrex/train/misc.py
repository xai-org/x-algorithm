# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import datetime
from typing import Literal, NamedTuple, Protocol

import haiku as hk
import jax
import optax
from xai_configlib import Config, configclass

from xrex.models.model_utils import Parameter


@configclass
class CheckpointConfig(Config):
    checkpoint_every_n: int = 0
    checkpoint_keep_every_nth: int = 10
    checkpoint_keep_last_n: int = 2
    save_final_checkpoint: bool = False

    copy_port: int = 0

    checkpoint_disk_every_s: int = 0

    shm_max_entries: int = 5

    from_checkpoint: bool = False
    no_opt_state: bool = False

    no_loading: str = ""

    rename_patterns: dict[str, str] | None = None
    rename_state_patterns: dict[str, str] | None = None

    checkpoint_dir: str | None = None

    extra_checkpoint_dirs: dict[str, str] | None = None

    timeout_secs: int = 900

    write_checksums: bool = True
    verify_checksums: bool = True

    checkpoint_chunked: bool = True
    checkpoint_compressed: bool = True
    checkpoint_chunk_size_bytes: int = 1024 * 1024 * 4

    checkpoint_ttl: int = datetime.timedelta(weeks=2).total_seconds()

    replication_mode: Literal["full", "dp_only", "none"] = "dp_only"
    save_method: Literal["orbax", "tensorstore"] = "orbax"

    replica_axis: str = "replica"

    def __post_init__(self):
        if self.shm_max_entries < 1:
            raise ValueError(f"shm_max_entries must be >= 1, got {self.shm_max_entries}")

    def should_save(self, step: int):
        return step and self.checkpoint_every_n and step % self.checkpoint_every_n == 0


class TrainingStateProtocol(Protocol):
    @property
    def params(self) -> hk.Params: ...

    @property
    def opt_state(self) -> optax.OptState | None: ...

    @property
    def rng(self) -> jax.Array: ...

    @property
    def step(self) -> jax.Array: ...

    def purge_opt_state(self) -> "TrainingStateProtocol": ...


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState | None
    rng: jax.Array
    step: jax.Array
    ema_ref_params: hk.Params | None = None

    def purge_opt_state(self):
        return self._replace(opt_state=None, ema_ref_params=None)


class PostEmbeddings(NamedTuple):
    post_ids: jax.Array
    author_ids: jax.Array
    embeddings: Parameter
    dataset_types: jax.Array


class RecsysInferenceState(NamedTuple):
    params: hk.Params
    rng: jax.Array
    step: jax.Array
    opt_state: optax.OptState | None
    post_embeddings: PostEmbeddings

    def purge_opt_state(self):
        return self._replace(opt_state=None)


class RecsysTrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState | None
    rng: jax.Array
    step: jax.Array
    emb_table: Parameter
    emb_table_state: optax.OptState | None
    offset_keys: jax.Array | None
    offset_values: jax.Array | None
    post_embeddings: PostEmbeddings
    rce_ema: dict[str, jax.Array] | None = None
    calib_ema: dict[str, jax.Array] | None = None

    def purge_opt_state(self):
        return self._replace(opt_state=None, emb_table_state=None)
