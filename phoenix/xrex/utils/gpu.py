# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import ctypes
import logging
import os
import re
import subprocess
from ctypes import byref, c_int, c_size_t
from enum import Enum
from functools import cache

from xrex.utils import cluster

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


class GpuArch(Enum):
    UNKNOWN = 0
    CPU = 1
    A100 = 2
    H100 = 3
    H200 = 4
    GB200 = 5
    GB300 = 6


def string_to_gpu_arch(arch_str: str) -> GpuArch:
    arch_str = arch_str.upper()
    if "H100" in arch_str or not arch_str:
        return GpuArch.H100
    if "H200" in arch_str:
        return GpuArch.H200
    if "A100" in arch_str:
        return GpuArch.A100
    if "GB200" in arch_str:
        return GpuArch.GB200
    if "GB300" in arch_str:
        return GpuArch.GB300
    if "CPU" in arch_str:
        return GpuArch.CPU
    raise ValueError(f"Unknown GPU architecture: {arch_str}")


def _check_cuda_errors(status):
    if status != 0:
        raise RuntimeError(f"CUDA error: {status}")


@cache
def gpu_arch():
    arch_env = os.getenv("MACHINE_TYPE", "").strip()
    if arch_env:
        return string_to_gpu_arch(arch_env)

    try:
        cuda_lib = None
        legal_driver_paths = [
            "/usr/lib/aarch64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
        ]

        for path in legal_driver_paths:
            if os.path.exists(path):
                cuda_lib = ctypes.CDLL(path)
                break

        if cuda_lib is None:
            try:
                cuda_lib = ctypes.CDLL("libcuda.so.1")
            except OSError:
                pass

        if cuda_lib is None:
            raise ValueError("No CUDA driver library found.")

        status = cuda_lib.cuInit(0)
        _check_cuda_errors(status)

        device_count = c_int()
        status = cuda_lib.cuDeviceGetCount(byref(device_count))
        _check_cuda_errors(status)

        if device_count.value == 0:
            raise ValueError("No GPU devices found.")

        device = c_int()
        device_name = ctypes.create_string_buffer(256)

        status = cuda_lib.cuDeviceGet(byref(device), 0)
        _check_cuda_errors(status)

        status = cuda_lib.cuDeviceGetName(device_name, c_size_t(256), device)
        _check_cuda_errors(status)

        device_name = device_name.value.decode("utf-8")
        rank_logger.info(f"Detected GPU architecture: {device_name}")
    except Exception as e:
        rank_logger.error(
            f"Failed to get GPU architecture from CUDA driver: {e}\nFalling back to CPU."
        )
        return GpuArch.CPU

    if "H100" in device_name:
        return GpuArch.H100
    if "H200" in device_name:
        return GpuArch.H200
    if "A100" in device_name:
        return GpuArch.A100
    if "GB200" in device_name:
        return GpuArch.GB200
    if "GB300" in device_name:
        return GpuArch.GB300
    raise ValueError(f"Unknown GPU architecture: {device_name}")


_GPU_ARCH_TO_PEAK_TFLOPS: dict[GpuArch, float] = {
    GpuArch.H100: 989.5,
    GpuArch.H200: 989.5,
    GpuArch.GB200: 2500,
    GpuArch.GB300: 2500,
}


@cache
def peak_tflops() -> float:
    gpu = gpu_arch()
    assert gpu in _GPU_ARCH_TO_PEAK_TFLOPS, f"Unknown GPU architecture: {gpu}!"
    return _GPU_ARCH_TO_PEAK_TFLOPS[gpu]


_NUM_PHYSICAL_DEVICES_PER_NODE = {
    GpuArch.CPU: 1,
    GpuArch.H100: 8,
    GpuArch.H200: 8,
    GpuArch.A100: 8,
    GpuArch.GB200: 4,
    GpuArch.GB300: 4,
}


@cache
def num_physical_devices_per_node() -> int:
    if os.getenv("NUM_PHYSICAL_DEVICES_PER_NODE") is not None:
        return int(os.getenv("NUM_PHYSICAL_DEVICES_PER_NODE"))

    arch = GpuArch.H100
    try:
        arch = gpu_arch()
    except Exception as e:
        rank_logger.error(f"Failed to get GPU architecture: {e}")
        rank_logger.error("Default to H100.")
    return _NUM_PHYSICAL_DEVICES_PER_NODE[arch]


NUM_PHYSICAL_DEVICES_PER_NODE = num_physical_devices_per_node()


def _read_sysfs(path: str) -> str:
    with open(path) as f:
        return f.read()


@cache
def _gpu_index_to_local_numa_node() -> dict[int, int]:
    mapping: dict[int, int] = {}
    probe_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as e:
        rank_logger.warning(f"NUMA pin: could not query GPU PCI ids ({e}); skipping")
        return mapping

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, _, bus_id = line.partition(",")
        idx_str = idx_str.strip()
        bus_id = bus_id.strip().lower()
        if not idx_str or not bus_id:
            continue
        if len(bus_id.split(":", 1)[0]) == 8:
            bus_id = bus_id[4:]
        try:
            node = int(_read_sysfs(f"/sys/bus/pci/devices/{bus_id}/numa_node").strip())
        except (OSError, ValueError):
            continue
        if node < 0:
            continue
        try:
            cpulist = _read_sysfs(f"/sys/devices/system/node/node{node}/cpulist").strip()
        except OSError:
            cpulist = ""
        if not cpulist:
            continue
        try:
            mapping[int(idx_str)] = node
        except ValueError:
            continue

    if len(set(mapping.values())) < 2:
        return {}
    return mapping


def _running_in_ci() -> bool:
    return cluster.get_user() == "ci"


def _running_on_a_devbox() -> bool:
    return "HOST_NODE_ADDR" not in os.environ


def _get_max_clocks() -> tuple[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"], stderr=subprocess.STDOUT, text=True
    )
    pattern = re.compile(r"Memory.+?(?P<memory>\d+).+?Graphics.+?(?P<graphics>\d+)", re.DOTALL)
    match = pattern.search(out)
    if not match:
        raise RuntimeError("Could not determine max GPU clocks")
    num_physically_attached_gpus = len(
        [line for line in out.splitlines() if line.startswith("GPU")]
    )
    assert num_physically_attached_gpus >= NUM_PHYSICAL_DEVICES_PER_NODE, (
        f"Expected {NUM_PHYSICAL_DEVICES_PER_NODE} GPUs but found {num_physically_attached_gpus}"
    )
    return int(match["memory"]), int(match["graphics"])


def set_gpu_clocks_to_max() -> None:
    if os.getenv("NO_CUDA") or _running_in_ci() or _running_on_a_devbox():
        return
    try:
        memory, graphics = _get_max_clocks()
        subprocess.check_output(
            ["nvidia-smi", "-ac", f"{memory},{graphics}"], stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        rank_logger.info(e.stdout)
        raise


def pin_process_to_local_numa(gpu_index: int | str) -> None:
    try:
        gpu_index = int(gpu_index)
    except (TypeError, ValueError):
        return
    node = _gpu_index_to_local_numa_node().get(gpu_index)
    if node is None:
        return
    try:
        import numa

        numa.schedule.run_on_nodes(node)
        numa.memory.set_membind_nodes(node)
        rank_logger.info(f"NUMA pin: gpu {gpu_index} -> node {node} (cpu affinity + membind)")
    except Exception as e:
        rank_logger.warning(f"NUMA pin: failed to bind gpu {gpu_index} to node {node} ({e})")
