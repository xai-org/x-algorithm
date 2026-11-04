# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import datetime
from dataclasses import field
from typing import Literal, NamedTuple, Protocol

import haiku as hk
import jax
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.models.model_utils import Parameter


@configclass
class CheckpointConfig(Config):
    checkpoint_every_n: int = 0
    checkpoint_keep_every_nth: int = 10
    checkpoint_keep_last_n: int = 2
    save_final_checkpoint: bool = False

    copy_port: int = 0

    copy_port_tls_cert: str = ""
    copy_port_tls_key: str = ""
    copy_port_tls_client_ca: str = ""
    copy_port_tls_ca: str = ""
    copy_port_tls_server_name: str = ""
    copy_port_tls_client_cert: str = ""
    copy_port_tls_client_key: str = ""

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
    checkpoint_ocdbt_target_data_file_size: int | None = None
    encrypt: bool = False
    encryption_key_id: str | None = None
    encryption_context: dict[str, str] = field(default_factory=dict)
    encryption_chunk_size_bytes: int = 8 * 1024 * 1024

    save_concurrent_gb: int | None = None

    restore_concurrent_gb: int | None = 32

    restore_streamed: bool = False

    restore_window_gb: float | None = None

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
    mol_side_table: Parameter | None = None


def replicated_global_array(host_array: np.ndarray, mesh: Mesh) -> jax.Array:
    return jax.make_array_from_callback(
        host_array.shape, NamedSharding(mesh, P(None, None)), lambda index: host_array[index]
    )


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
