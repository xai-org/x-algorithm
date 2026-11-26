# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import math
import os
import random
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax.ad_checkpoint import print_saved_residuals
from jax.sharding import Mesh

from xrex.train.parallel_config import ParallelConfig
from xrex.utils.gpu import GpuArch, gpu_arch


def free_port() -> int:
    for _ in range(100):
        port = random.randrange(10000, 30000)
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free test coordinator port")


def visible_gpu_count() -> int:
    if (visible := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
        return len(visible.replace(",", " ").split())
    try:
        gpu_indices = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len(gpu_indices.split())


def run_python_processes(
    args_by_process: Sequence[Sequence[str]],
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for args in args_by_process
    ]
    outputs = [process.communicate(timeout=timeout)[0] for process in processes]
    failures = [
        f"--- process {index}\n{output[-6000:]}"
        for index, (process, output) in enumerate(zip(processes, outputs, strict=True))
        if process.returncode
    ]
    assert not failures, "\n".join(failures)


def run_on_cpu_devices(num_devices: int, check: Callable[[], None]) -> None:
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} "
        f"--xla_force_host_platform_device_count={num_devices}",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    run_check = (
        "import importlib, sys; getattr(importlib.import_module(sys.argv[1]), sys.argv[2])()"
    )
    run_python_processes([["-c", run_check, check.__module__, check.__name__]], env=env)


def relative_error(got: npt.ArrayLike, want: npt.ArrayLike) -> float:
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / max(float(np.linalg.norm(want)), 1e-12))


def require_h100() -> None:
    if jax.default_backend() != "gpu" or gpu_arch() != GpuArch.H100:
        pytest.skip("Needs an H100")


def trainer_mesh(expert: int = 1, data: int = 1, model: int = 1) -> Mesh:
    sizes = dict.fromkeys(ParallelConfig().axis_names, 1) | {
        "expert": expert,
        "data": data,
        "model": model,
    }
    devices = jax.devices()[: math.prod(sizes.values())]
    return Mesh(np.array(devices).reshape(tuple(sizes.values())), tuple(sizes))


def assert_bitwise_equal(got, want) -> None:
    for (path, want_leaf), got_leaf in zip(
        jax.tree_util.tree_leaves_with_path(want), jax.tree.leaves(got), strict=True
    ):
        np.testing.assert_array_equal(
            *(np.atleast_1d(np.asarray(x)).view(np.uint8) for x in (got_leaf, want_leaf)),
            err_msg=jax.tree_util.keystr(path),
        )


def assert_relative_errors_below(got, want, tolerance: float) -> None:
    for (path, want_leaf), got_leaf in zip(
        jax.tree_util.tree_leaves_with_path(want), jax.tree.leaves(got), strict=True
    ):
        error = relative_error(got_leaf, want_leaf)
        assert error < tolerance, f"{jax.tree_util.keystr(path)}: relative error {error:.1e}"


def saved_residual_counts(fn: Callable, args, capsys) -> collections.Counter[str]:
    print_saved_residuals(fn, *args)
    return collections.Counter(line.split()[0] for line in capsys.readouterr().out.splitlines())


def home_like_transformer(fp8_gemms: str = "off"):
    from xrex.models.attention import AttentionConfig
    from xrex.models.remat import RematType
    from xrex.models.transformer import FeedForwardConfig, TransformerConfig

    return TransformerConfig(
        attn_config=AttentionConfig(
            key_size=32,
            num_q_heads=10,
            num_kv_heads=2,
            causal=False,
            qk_norm=True,
            rotary=True,
            qkv_merge=True,
            attn_impl="jax_attn",
        ),
        ffn_config=FeedForwardConfig(widening_factor=2),
        emb_size=128,
        num_layers=2,
        sequence_len=256,
        use_layer_stack=True,
        unroll_layer_stack=True,
        use_remat=True,
        remat_policy=RematType.SAVE_H100_RECSYS,
        fp8_gemms=fp8_gemms,
    )


def transformer_loss(config, mesh: Mesh, batch: int = 1):
    import haiku as hk

    from xrex.models.sharding_context import make_legacy_sharding_context
    from xrex.models.transformer import Transformer

    shape = (batch, config.sequence_len, config.emb_size)
    embeddings = jax.random.normal(jax.random.PRNGKey(1), shape).astype(jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.PRNGKey(2), shape)

    def forward(embeddings):
        ones = jnp.ones(embeddings.shape[:2], jnp.int32)
        transformer = Transformer(config, make_legacy_sharding_context(mesh))
        return transformer(embeddings, mask=ones, segment_ids=ones).output

    transformed = hk.transform(forward)

    def loss(params, embeddings, cotangent):
        out = transformed.apply(params, jax.random.PRNGKey(0), embeddings)
        return jnp.sum(out.astype(jnp.float32) * cotangent)

    return transformed, loss, embeddings, cotangent
