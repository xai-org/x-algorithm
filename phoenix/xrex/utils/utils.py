# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import contextlib
import logging
import os
import resource
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from types import TracebackType
from typing import (
    Any,
    ClassVar,
    Self,
    TypeVar,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.xla_metadata import set_xla_metadata

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


def ffn_size(
    emb_size: int,
    widening_factor: float,
    gated_mlp: bool = False,
    base_size: int = 32,
    force_ffn_base_size: bool = True,
) -> int:
    ffn_size = int(widening_factor * emb_size)
    if gated_mlp:
        ffn_size = ffn_size * 2 // 3

    if force_ffn_base_size:
        ffn_size = ((ffn_size + (base_size - 1)) // base_size) * base_size

    logger.debug(f"emb_size: {emb_size} adjusted ffn_size: {ffn_size}")
    return ffn_size


def remat_less(
    layer: F,
    *,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
    prevent_cse: bool = True,
) -> F:
    del static_argnums, policy, prevent_cse
    return layer


def disable_core_dumps() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def increase_fd_limit() -> None:
    limit = 1 << 20
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    except ValueError:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, hard), hard))


def compute_num_elements(leaves: Any) -> int:
    def compute(x: Any) -> int:
        if hasattr(x, "shape"):
            return int(np.prod(x.shape))
        return 0

    leaves = jax.tree_util.tree_flatten(leaves)[0]
    return sum(map(compute, leaves))


def compute_num_stored_elements(params):
    return sum(x.addressable_data(0).size for x in jax.tree.leaves(params)) * jax.device_count()


def get_bytes_in_use() -> int:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None:
        return 0
    return memory_stats["bytes_in_use"]


def get_peak_bytes_in_use() -> int:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None:
        return 0
    return memory_stats["peak_bytes_in_use"]


def safe_cast(x: Any, target_dtype: Any) -> Any:
    if hasattr(x, "astype") and hasattr(x, "dtype") and x.dtype.kind == "f":
        return x.astype(target_dtype)
    elif isinstance(x, jax.ShapeDtypeStruct) and x.dtype.kind == "f":
        return jax.ShapeDtypeStruct(x.shape, target_dtype, sharding=getattr(x, "sharding", None))
    else:
        return x


def cast_bfloat16(x: Any) -> Any:
    return safe_cast(x, jnp.bfloat16)


DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}


def selective_tree_cast(cast_rule: dict[str, str] = {}, default_dtype: str | None = None):
    def _cast_to_dtype(value, dtype: str):
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Invalid dtype: {dtype}")
        return safe_cast(value, DTYPE_MAP[dtype])

    def _cast_with_path(x):
        def cast_with_path(path, value):
            path_str = "/".join(str(key.key) if hasattr(key, "key") else str(key) for key in path)

            for keep_pattern, keep_dtype in cast_rule.items():
                if keep_pattern in path_str:
                    return _cast_to_dtype(value, keep_dtype)

            if default_dtype is not None:
                return _cast_to_dtype(value, default_dtype)
            return value

        return jax.tree_util.tree_map_with_path(cast_with_path, x)

    return _cast_with_path


def flatten_dict(d: dict[Any, Any]) -> dict[str, Any]:
    new_d: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kf, vf in flatten_dict(v).items():
                new_d[f"{k}/{kf}"] = vf
        else:
            new_d[str(k)] = v
    return new_d


class Summary:
    _ACTIVE: ClassVar[list[Summary]] = []

    def __init__(self, strict=True) -> None:
        self._summarized: dict[str, Any] = {}
        self.strict = strict

    def __enter__(self) -> Self:
        self._ACTIVE.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._ACTIVE.remove(self)

    def add(self, key: str, value: Any) -> None:
        if key in self._summarized and self.strict:
            raise RuntimeError("duplicated summarization key")
        self._summarized[key] = value

    def get(self) -> dict[str, Any]:
        return self._summarized

    @classmethod
    def summarize(cls, key: str, value: Any) -> None:
        for summarizer in cls._ACTIVE:
            summarizer.add(key, value)


def summarize(key: str, value: Any) -> None:
    Summary.summarize(key, value)


class TensorDumpManager:
    def __init__(self):
        self.dump_file_list: set[str] = set()
        self.global_idx: int = 0

    def dump_to_file(
        self,
        folder: str | None,
        name: str,
        value: jax.Array | dict[str, jax.Array] | None,
        append_global_idx: bool = False,
    ) -> None:
        if not folder:
            return

        if value is None:
            return

        os.makedirs(folder, exist_ok=True)

        def func(value):
            if name not in self.dump_file_list:
                self.dump_file_list.add(name)
                rank_logger.info(f"skipping dump {name}")
                return

            append_global_idx = os.getenv("XAI_DUMP_TENSOR_APPEND_GLOBAL_IDX", "0") == "1"
            if append_global_idx:
                new_name = f"{name}_{self.global_idx}"
            else:
                new_name = name

            def to_numpy(x):
                if x.dtype is jnp.bfloat16.dtype:
                    x = x.astype(np.float32)
                x = np.array(x)
                return x

            if isinstance(value, dict):
                output_filename = os.path.join(folder, f"jax_dump_{new_name}.npz")
                np.savez(output_filename, **jax.tree.map(to_numpy, value))
            else:
                output_filename = os.path.join(folder, f"jax_dump_{new_name}.npy")
                np.save(output_filename, to_numpy(value))
            rank_logger.info(
                f"Dumped a tensor to {output_filename}. Shape = {jax.tree.map(lambda x: x.shape, value)}"
            )

        jax.debug.callback(func, value)

    def increment_global_idx(self, cnt: int) -> None:
        def func(cnt):
            print(f"increment_global_idx: {cnt}")
            self.global_idx += cnt

        jax.debug.callback(func, cnt)

    def sleep_before_next_step(self, sleep_time: int) -> None:
        def func(sleep_time):
            rank_logger.info(
                f"Dumped logits, waiting {sleep_time} sec to give opportunity to Ctrl-C before next step"
            )
            time.sleep(sleep_time)

        jax.debug.callback(func, sleep_time)


_default_tensor_dumper = TensorDumpManager()


def dump_to_file(folder: str | None, name: str, value: jax.Array) -> None:
    _default_tensor_dumper.dump_to_file(folder, name, value)


def unroll_layer_dumps(layer_dumps):
    unrolled_layer_dumps = []
    for dump in layer_dumps:
        unrolled_layer_dumps.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), dump))
    return unrolled_layer_dumps


def dump_block_outputs(config, layer_dumps, total_num_layers: int, n_layers: int | None = None):
    if config.debug_tensor_dump_only_target_logprobs:
        return

    n_layers = total_num_layers if n_layers is None else n_layers
    for i in range(n_layers):
        names = [
            "attn_input",
            "q",
            "k",
            "v",
            "attn_output",
            "attn_pre_ln_hid",
            "attn_post_ln_output",
            "mlp_in",
            "mlp_out",
            "mlp_post_ln_output",
        ]
        assert len(layer_dumps) == len(names), (len(layer_dumps), len(names))
        assert len(layer_dumps[0]) == total_num_layers, (len(layer_dumps[0]), total_num_layers)
        for j, name in enumerate(names):
            if layer_dumps[j] is None:
                continue
            batch_size = layer_dumps[j][i].shape[0]
            start_pos = config.debug_tensor_dump_start_pos
            num_dumps = min(config.debug_tensor_dump_num_dumps, batch_size - start_pos)
            for k in [x + start_pos for x in range(max(0, num_dumps))]:
                try:
                    dump_to_file(
                        config.debug_tensor_dump_output_folder,
                        f"{name}_{i}_{k}",
                        jax.tree.map(lambda x, i=i, k=k: x[i][k], layer_dumps[j]),
                    )
                except Exception as e:
                    rank_logger.warning(f"Failed to dump {name}_{i} with error {e}")
                    rank_logger.warning(f"{layer_dumps[j]=}")
                    raise


def compute_norm(tree: Any):
    with (
        set_xla_metadata(_xla_collective_group="metrics"),
        jax.named_scope("compute_norm"),
    ):
        norms_squared = jax.tree.map(lambda x: jnp.sum(jnp.square(x)), tree)
        norms_squared, _ = jax.tree_util.tree_flatten(norms_squared)
        return jnp.sqrt(jnp.sum(jnp.stack(norms_squared)))


@contextlib.contextmanager
def alarm_after(timeout):
    signal.alarm(int(timeout))
    try:
        yield
    finally:
        signal.alarm(0)


class PreemptionNotifier:
    def __init__(self):
        self.preempted = threading.Event()

    def enable(self):
        self.preempted.clear()
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGQUIT, self._on_signal)

    def disable(self):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    def _on_signal(self, signum, frame):
        self.preempted.set()
        name = signal.Signals(signum).name
        sys.stderr.write(f"Received {name} (preemption), setting flag for checkpoint\n")
        if frame is not None:
            traceback.print_stack(frame, file=sys.stderr)
        sys.stderr.flush()


def gelu_fn(x: jnp.ndarray, approximate: bool = False):
    return jax.nn.gelu(x, approximate=approximate)
