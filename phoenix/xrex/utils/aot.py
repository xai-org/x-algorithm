# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import errno
import hashlib
import io
import logging
import os
import pickle
import socket
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import filelock
import jax
import numpy as np
from jax import numpy as jnp
from jax._src.cache_key import IgnoreCallbacks, _hash_string, _hash_xla_flags, _remove_callbacks
from jax._src.lib import version_str as jaxlib_version_str
from jax._src.lib import xla_client as xc
from jax.experimental import multihost_utils
from jax.extend.backend import backend_xla_version, get_backend
from jax.stages import Compiled, Lowered, Traced, Wrapped
from jaxlib.mlir import ir

from xrex.utils.profiler import read_fdo_profile

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


AOT_CACHE_VERSION: str = "20260705"


def is_host_callbacks(obj: Any) -> bool:
    return (
        isinstance(obj, list) and len(obj) > 0 and callable(obj[0]) and "callback" in repr(obj[0])
    )


@dataclass
class TracedWithOptions:
    traced: Traced
    options: dict[str, Any] = None
    add_location_info: bool = False


@dataclass
class AOTInfo:
    cache_hit: bool
    buffer_size: int
    elapsed_compile: float | None = None
    elapsed_replicate: float = 0.0
    elapsed_load: float = 0.0


@dataclass(frozen=True)
class PartiallySerialized:
    serialized_executable: bytes
    in_shardings_mem_kinds: Any
    out_shardings_mem_kinds: Any
    no_kwargs: bool
    compile_options: bytes | None = None


def clear_xla_dump_flags(compile_options: xc.CompileOptions | None) -> xc.CompileOptions | None:
    if compile_options is None:
        return None

    ebo = compile_options.executable_build_options
    debug_opts = ebo.debug_options

    debug_opts.xla_dump_to = ""

    debug_opts.xla_dump_hlo_module_re = ""
    debug_opts.xla_dump_hlo_pass_re = ""

    debug_opts.xla_dump_hlo_as_text = False
    debug_opts.xla_dump_hlo_as_proto = False

    dump_flag_prefixes = ("xla_dump_",)
    filtered_overrides = [
        (key, value)
        for key, value in compile_options.env_option_overrides
        if not key.startswith(dump_flag_prefixes)
    ]
    filtered_overrides.append(("xla_dump_full_hlo_config", False))
    compile_options.env_option_overrides = filtered_overrides

    return compile_options


def remap_compile_options_device_assignment(
    compile_options: xc.CompileOptions | None,
    execution_devices: Sequence[xc.Device],
) -> xc.CompileOptions | None:
    if compile_options is None:
        return None

    device_assignment = compile_options.device_assignment
    if device_assignment is None:
        return compile_options

    replica_count = device_assignment.replica_count()
    computation_count = device_assignment.computation_count()
    expected_devices = replica_count * computation_count
    if len(execution_devices) != expected_devices:
        raise ValueError(
            "Cannot remap XLA compile options device assignment: "
            f"assignment expects {expected_devices} devices "
            f"({replica_count=} x {computation_count=}), "
            f"but got {len(execution_devices)} execution devices."
        )

    current_device_ids = np.asarray([d.id for d in execution_devices], dtype=np.int64)
    compile_options.device_assignment = xc.DeviceAssignment.create(
        current_device_ids.reshape(replica_count, computation_count, order="F")
    )
    return compile_options


def load_compiled_from_serialized(
    lowered: Lowered,
    partially_serialized: PartiallySerialized,
    execution_devices: Sequence[xc.Device],
):
    xla_lowered = lowered._lowering
    assert isinstance(xla_lowered, jax.interpreters.pxla.MeshComputation), (
        f"lowered must be a MeshComputation instance. Got: {type(lowered)}"
    )
    host_callbacks = xla_lowered.compile_args.get("host_callbacks", None)

    compile_options = None
    serialized_compile_options = getattr(partially_serialized, "compile_options", None)
    if serialized_compile_options is not None:
        compile_options = xc.CompileOptions.ParseFromString(serialized_compile_options)
        compile_options = clear_xla_dump_flags(compile_options)
        compile_options = remap_compile_options_device_assignment(
            compile_options, execution_devices
        )

    unloaded_executable = _JaxPjrtUnpickler(
        io.BytesIO(partially_serialized.serialized_executable),
        get_backend(),
        execution_devices=execution_devices,
        host_callbacks=host_callbacks,
        compile_options=compile_options,
    ).load()

    if hasattr(partially_serialized, "in_shardings_mem_kinds"):
        restored_in_shardings = [
            s.with_memory_kind(m)
            for (s, m) in zip(
                unloaded_executable.input_shardings,
                partially_serialized.in_shardings_mem_kinds,
            )
        ]
        unloaded_executable.input_shardings = restored_in_shardings
    else:
        logger.error("Please update your AOT cache to onboard mem kinds.")

    if hasattr(partially_serialized, "out_shardings_mem_kinds"):
        restored_out_shardings = [
            s.with_memory_kind(m)
            for (s, m) in zip(
                unloaded_executable.output_shardings,
                partially_serialized.out_shardings_mem_kinds,
            )
        ]
        unloaded_executable.output_shardings = restored_out_shardings

    return jax.stages.Compiled(
        unloaded_executable.load(),
        [],
        lowered.args_info,
        lowered.out_tree,
        no_kwargs=partially_serialized.no_kwargs,
    )


def partially_serialize(compiled: Compiled, execution_devices: Sequence[xc.Device]):
    unloaded_exec = getattr(compiled._executable, "_unloaded_executable", None)
    if unloaded_exec is None:
        raise ValueError("Compilation does not support serialization")

    in_shardings_mem_kinds = [s.memory_kind for s in unloaded_exec.input_shardings]
    out_shardings_mem_kinds = [s.memory_kind for s in unloaded_exec.output_shardings]

    compile_options = getattr(unloaded_exec, "compile_options", None)
    serialized_compile_options = None
    if compile_options is not None:
        serialized_compile_options = compile_options.SerializeAsString()

    with io.BytesIO() as file:
        _JaxPjrtPickler(file, execution_devices).dump(unloaded_exec)
        return PartiallySerialized(
            file.getvalue(),
            in_shardings_mem_kinds,
            out_shardings_mem_kinds,
            compiled._no_kwargs,
            serialized_compile_options,
        )


class _JaxPjrtPickler(pickle.Pickler):
    def __init__(self, file, execution_devices: Sequence[xc.Device]):
        super().__init__(file)
        if not all(isinstance(d, xc.Device) for d in execution_devices):
            raise ValueError(
                f"execution_devices must be a list of xc.Device. Got: {execution_devices}"
            )
        self.device_to_index: dict[int, int] = {d.id: i for i, d in enumerate(execution_devices)}

    def persistent_id(self, obj):
        if isinstance(obj, xc.LoadedExecutable):
            return ("exec", obj.client.serialize_executable(obj))
        if isinstance(obj, xc._xla.Executable):
            return ("exec", obj.serialize())
        if isinstance(obj, xc.Device):
            if obj.id not in self.device_to_index:
                raise pickle.PicklingError(f"Unknown device: {obj}")
            return ("device", self.device_to_index[obj.id])
        if isinstance(obj, xc.Client):
            return ("client",)
        if is_host_callbacks(obj):
            return ("callbacks",)
        if isinstance(obj, xc.CompileOptions):
            return ("compile_options", obj.SerializeAsString())
        return None


class _JaxPjrtUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file,
        backend: xc.Client,
        execution_devices: Sequence[xc.Device],
        host_callbacks: Any,
        compile_options: xc.CompileOptions | None,
    ):
        super().__init__(file)
        self.backend = backend
        device_backend = execution_devices[0].client
        if device_backend != backend:
            raise ValueError(
                "Execution devices belong to a client other than `backend`. Got "
                f"backend client: {(backend.platform, backend.platform_version)} "
                "and execution devices client: "
                f"{(device_backend.platform, device_backend.platform_version)}"
            )
        self.execution_devices = xc.DeviceList(tuple(execution_devices))
        self.host_callbacks = host_callbacks
        self.compile_options = compile_options

    def persistent_load(self, pid):
        if pid[0] == "exec":
            compile_options = self.compile_options
            if self.host_callbacks:
                return self.backend.deserialize_executable(
                    pid[1],
                    executable_devices=self.execution_devices,
                    compile_options=compile_options,
                    host_callbacks=self.host_callbacks,
                )
            return self.backend.deserialize_executable(
                pid[1],
                executable_devices=self.execution_devices,
                compile_options=compile_options,
            )
        if pid[0] == "device":
            device_index = pid[1]
            if device_index < 0 or device_index >= len(self.execution_devices):
                raise pickle.UnpicklingError(f"Device index out of bounds: {device_index}")
            return self.execution_devices[device_index]
        if pid[0] == "client":
            return self.backend
        if pid[0] == "callbacks":
            return self.host_callbacks
        if pid[0] == "compile_options":
            compile_options = xc.CompileOptions.ParseFromString(pid[1])
            compile_options = clear_xla_dump_flags(compile_options)
            compile_options = remap_compile_options_device_assignment(
                compile_options, self.execution_devices
            )
            return compile_options
        raise pickle.UnpicklingError(f"Unknown object type: {pid[0]}")


def _lowered_name(lowered: Lowered):
    sym_name = lowered.compiler_ir(dialect="stablehlo").operation.attributes["sym_name"]
    return ir.StringAttr(sym_name).value


@dataclass(frozen=True)
class HashContext:
    hlo: str
    args_info: str
    environment: str
    additional_content: str = None


def read_annotation_file(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Compiler annotation file {file_path} not found.")
    with p.open("r") as f:
        return f.read()


def get_sanitized_ir_text(lowered: Lowered, add_loc: bool) -> str:
    module = lowered.compiler_ir(dialect="stablehlo")
    with module.context:
        sanitized = _remove_callbacks(module.operation.clone(), IgnoreCallbacks.ALL)
        return sanitized.operation.get_asm(enable_debug_info=add_loc)


def get_environment_info() -> str:
    backend = get_backend()
    device_kinds = ",".join(sorted({d.device_kind for d in backend.devices()}))
    return "\n".join(
        [
            f"jax={jax.version.__version__}",
            f"jaxlib={jaxlib_version_str}",
            f"xla_client_version={xc._version}",
            f"backend_xla_version={backend_xla_version()}",
            f"platform={backend.platform}",
            f"platform_version={backend.platform_version}",
            f"runtime_type={backend.runtime_type}",
            f"device_kinds={device_kinds}",
        ]
    )


def get_cache_key(lowered: Lowered, extra_info: list[str] = None, add_loc: bool = False):
    hash_obj = hashlib.sha256()
    _hash_string(hash_obj, AOT_CACHE_VERSION)
    environment = get_environment_info()
    _hash_string(hash_obj, environment)

    _hash_xla_flags(hash_obj, [])
    sanitized_ir = get_sanitized_ir_text(lowered, add_loc)
    _hash_string(hash_obj, sanitized_ir)
    _hash_string(hash_obj, str(lowered.args_info))
    _hash_string(hash_obj, str(lowered.out_tree))

    lowering = lowered._lowering
    if not isinstance(lowering, jax.interpreters.pxla.MeshComputation):
        raise NotImplementedError("AOT is not implemented for non-XLA lowerings.")

    compile_args = lowering.compile_args
    if "global_in_avals" in compile_args:
        _hash_string(hash_obj, str(compile_args["global_in_avals"]))
    if "global_out_avals" in compile_args:
        _hash_string(hash_obj, str(compile_args["global_out_avals"]))
    if "in_shardings" in compile_args:
        _hash_string(hash_obj, str(compile_args["in_shardings"]))
    if "out_shardings" in compile_args:
        _hash_string(hash_obj, str(compile_args["out_shardings"]))
    if "in_layouts" in compile_args:
        _hash_string(hash_obj, str(compile_args["in_layouts"]))
    if "out_layouts" in compile_args:
        _hash_string(hash_obj, str(compile_args["out_layouts"]))

    additional_content = ":".join(extra_info) if extra_info else None
    if additional_content:
        _hash_string(hash_obj, additional_content)
    return _lowered_name(lowered) + "_" + hash_obj.digest().hex(), HashContext(
        hlo=sanitized_ir,
        args_info=str(lowered.args_info),
        environment=environment,
        additional_content=additional_content,
    )


def _replicate(array):
    devices = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())
    global_mesh = jax.sharding.Mesh(devices, ("processes", "local_devices"))
    pspec = jax.sharding.PartitionSpec("processes")
    array = multihost_utils.host_local_array_to_global_array(
        jnp.expand_dims(array, axis=0), global_mesh, pspec
    )
    return np.array(jnp.sum(array, axis=0, dtype=np.uint8))


def replicate(source: np.ndarray | None = None) -> np.ndarray:
    if source is None:
        size = jnp.zeros((1,), dtype=np.uint32)
    else:
        size = jnp.array([source.nbytes], dtype=np.uint32)

    size = _replicate(size.view(np.uint8)).view(np.uint32)

    if source is None:
        buf = jnp.zeros((size.item(),), dtype=np.uint8)
    else:
        assert size.item() == source.nbytes
        assert source.dtype == np.uint8
        buf = source
    return _replicate(buf)


@contextmanager
def RetryFileLock(lock_file: Path, *, poll_interval: float):
    assert poll_interval > 0, f"require {poll_interval=} > 0"
    lock = filelock.FileLock(lock_file, timeout=-1, poll_interval=poll_interval)

    while True:
        try:
            lock.acquire()
            break
        except OSError as e:
            if e.errno not in (errno.ENOLCK, errno.ESTALE, errno.ENOENT):
                raise
            if e.errno == errno.ENOENT and not lock_file.parent.exists():
                raise FileNotFoundError(
                    f"Lock file parent directory does not exist: {lock_file.parent}"
                ) from e
            logger.warning(
                "Transient error acquiring lock %s: %s. Retrying...",
                lock_file,
                e,
            )
            time.sleep(poll_interval)
        except NotImplementedError as e:
            if "use SoftFileLock instead" in str(e):
                logger.warning(
                    "FileSystem does not appear to support flock. Falling back to SoftFileLock for %s",
                    lock_file,
                )
                lock = filelock.SoftFileLock(lock_file, timeout=-1, poll_interval=poll_interval)
            else:
                raise

    try:
        yield
    finally:
        try:
            lock.release()
        except OSError:
            try:
                os.remove(lock_file)
            except Exception:
                pass


def compile_or_load_all_traced(
    all_traced: dict[str, TracedWithOptions],
    aot_cache_dir: str | Path,
    fdo_profile_dir: str | None = None,
    run_dir: str | None = None,
    aot_dump: bool = False,
    devices: Sequence[xc.Device] | None = None,
) -> tuple[dict[str, Compiled], dict[str, dict[str, float]]]:
    if devices is None:
        devices = []

    if not aot_cache_dir:
        raise ValueError("AOT cache directory must be provided.")

    aot_cache_dir = Path(aot_cache_dir)
    rank = jax.process_index()
    dump_hlo_dir = Path(run_dir) / "hlos" if run_dir else Path(f"/tmp/hlos/{jax.process_index()}")

    if len(devices) == 0:
        devices = get_backend().devices()
    elif isinstance(devices, np.ndarray):
        devices = devices.flatten()
    assert isinstance(devices[0], xc.Device), (
        f"compile_or_load_all_traced only accepts a list of devices but got: {devices}"
    )

    if rank == 0:
        aot_cache_dir.mkdir(parents=True, exist_ok=True)

    def _compile_or_load(name, traced_with_options, metrics: dict[str, dict[str, float]]):
        profile = read_fdo_profile(name, fdo_profile_dir)
        traced, options, add_location_info = (
            traced_with_options.traced,
            deepcopy(traced_with_options.options),
            traced_with_options.add_location_info,
        )

        extra_info = []
        dump_all_hlos = options.pop("dump_all_hlos", False) if options else False
        if profile:
            extra_info.append(profile)
        if options:
            compiler_annotation = options.get("xla_gpu_compiler_annotations_file", "")
            if compiler_annotation:
                options["xla_dump_latency_hiding_schedule"] = True
            extra_info.append(
                read_annotation_file(compiler_annotation) if compiler_annotation else ""
            )
            extra_info.extend([f"{key}={value}" for key, value in options.items()])

        device_idx_str = ",".join([str(d.process_index) for d in devices])
        extra_info.append(f"process_count={jax.process_count()}")
        extra_info.append(f"device_count={jax.device_count()}")
        extra_info.append(f"local_device_count={jax.local_device_count()}")
        extra_info.append(f"device_idx_str={device_idx_str}")

        lowered = traced.lower()
        cache_key, hash_context = get_cache_key(
            lowered, extra_info=extra_info, add_loc=add_location_info
        )
        cache_path = aot_cache_dir / cache_key

        if aot_dump and rank == 0:
            aot_dump_path = aot_cache_dir.joinpath(cache_key + "_debug")
            if not aot_dump_path.exists():
                with open(aot_dump_path, "w") as f:
                    f.write("==== Hlo ====\n\n")
                    f.write(hash_context.hlo)
                    f.write("\n==== Args Info ====\n\n")
                    f.write(hash_context.args_info)
                    f.write("\n==== Environment ====\n\n")
                    f.write(hash_context.environment)
                    f.write("\n==== Additional Content ====\n\n")
                    f.write(hash_context.additional_content or "")
                    f.write("\n==== END ====\n")

        aot_info: AOTInfo | None = None
        elapsed_compile = None
        if rank == 0:
            cache_key_buf = np.frombuffer(cache_key.encode("utf-8"), dtype=np.uint8)
            replicate(source=cache_key_buf)

            is_hit = cache_path.is_file()

            if not is_hit:
                lock_file = cache_path.with_suffix(".lock")
                with RetryFileLock(lock_file, poll_interval=1.0):
                    if cache_path.is_file():
                        is_hit = True
                        compiled = None
                    else:
                        logger.info(f"Compiling executable {name} on {socket.gethostname()}.")
                        compiler_options = {
                            "fdo_profile": None if not profile else profile.encode("utf-8"),
                            "xla_dump_to": str(dump_hlo_dir),
                            "xla_dump_hlo_as_text": True,
                            "xla_dump_hlo_module_re": "compute_grad|apply_grad|update",
                            "xla_dump_hlo_pass_re": ".*" if dump_all_hlos else "",
                            "xla_gpu_shard_autotuning": False,
                        }
                        if options and "xai" not in jax.__version__:
                            options.pop("xla_gpu_compiler_annotations_file", None)
                        if options:
                            compiler_options.update(options)

                        s = time.perf_counter()
                        compiled = lowered.compile(compiler_options=compiler_options)
                        elapsed_compile = time.perf_counter() - s
                        serialized_exec = partially_serialize(compiled, execution_devices=devices)

                        temp_cache_path = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
                        with open(temp_cache_path, "wb") as f:
                            cloudpickle.dump(serialized_exec, f)
                            os.fsync(f.fileno())
                        os.replace(temp_cache_path, cache_path)

                        logger.info(
                            f"Materialized executable {_lowered_name(lowered)} to {cache_path}."
                        )

                if compiled is not None:
                    try:
                        flops = (compiled.cost_analysis() or {}).get("flops", None)
                        logger.info(f"Compiled executable {name} flops: {flops}")
                        memory = compiled.memory_analysis()
                        logger.info(f"Compiled executable {name} memory: {memory}")
                    except Exception:
                        pass

            buf = np.memmap(cache_path, mode="r")

            s = time.perf_counter()
            buf = replicate(source=buf)
            elapsed = time.perf_counter() - s
            aot_info = AOTInfo(
                cache_hit=is_hit,
                buffer_size=buf.size,
                elapsed_replicate=elapsed,
                elapsed_compile=elapsed_compile,
            )
        else:
            other_key = replicate()
            other_key = other_key.tobytes().decode("utf-8")
            assert cache_key == other_key, (
                f"AOT cache key mismatch within JAX runtime: rank #0 had {cache_key}, while other rank has {other_key}"
            )

            buf = replicate()

        try:
            s = time.perf_counter()
            partially_serialized = cloudpickle.loads(buf)
            compiled = load_compiled_from_serialized(
                lowered,
                partially_serialized,
                execution_devices=devices,
            )
            elapsed = time.perf_counter() - s
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to unpickle compiled executable at {cache_path}: {e}")
            if rank == 0:
                try:
                    os.remove(cache_path)
                except Exception as e:
                    pass
            raise

        temp_size = compiled.memory_analysis().temp_size_in_bytes
        output_size = compiled.memory_analysis().output_size_in_bytes
        host_offload_size = compiled.memory_analysis().host_temp_size_in_bytes

        GiB = 1 << 30
        format_GiB = lambda bytes: f"{bytes / GiB:.2f} GiB"
        format_secs = lambda secs: f"{round(secs, 2)}s"
        aot_summary: dict[str, str | int | None] = {
            "temp_size": format_GiB(temp_size),
            "host_offload_size": format_GiB(host_offload_size),
            "output_size": format_GiB(output_size),
            "cache_key": cache_key,
        }

        if aot_info is not None:
            assert rank == 0
            aot_info.elapsed_load = elapsed
            aot_summary["cache_hit"] = aot_info.cache_hit
            aot_summary["size"] = aot_info.buffer_size
            aot_summary["replicate_time"] = format_secs(aot_info.elapsed_replicate)
            aot_summary["load_time"] = format_secs(aot_info.elapsed_load)
            aot_summary["compile_time"] = (
                None if not aot_info.elapsed_compile else format_secs(aot_info.elapsed_compile)
            )

        rank_logger.info(
            f"Memory usage of executable {name}: {', '.join(f'{k}: {v}' for k, v in aot_summary.items())}"
        )

        assert name not in metrics
        metrics[name] = {
            "temp_size": temp_size,
            "host_offload_size": host_offload_size,
            "output_size": output_size,
        }

        return compiled

    metrics: dict[str, dict[str, float]] = {}
    all_compiled = {n: _compile_or_load(n, t, metrics) for n, t in all_traced.items()}
    return all_compiled, metrics


class JittedOrCompiled(Wrapped):
    __slots__: tuple[str, ...] = ("fun_name", "jitted")

    def __init__(self, jitted, name=None):
        self.fun_name = name
        self.jitted = jitted

    def lower(self, *args, **kwargs):
        if isinstance(self.jitted, Compiled):
            raise ValueError("Cannot lower a compiled jit function.")
        return self.jitted.lower(*args, **kwargs)

    def trace(self, *args, **kwargs) -> Traced:
        if isinstance(self.jitted, (Traced, Compiled)):
            raise TypeError(f"Unsupported type: {type(self.jitted)}.")
        return self.jitted.trace(*args, **kwargs)

    def name(self):
        if self.fun_name is not None:
            return self.fun_name
        if isinstance(self.jitted, Wrapped):
            return getattr(self.jitted._fun, "__name__", str(self.jitted._fun))
        raise TypeError(f"Unsupported type: {type(self.jitted)}.")

    def __call__(self, *args, **kwargs):
        return self.jitted(*args, **kwargs)
