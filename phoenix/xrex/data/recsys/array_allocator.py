# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
import mmap
import threading
import weakref
from collections import defaultdict

import numpy as np
import numpy.typing as npt

ALLOCATOR_BLOCK_BYTES = 64 << 20
ARRAY_ALIGNMENT_BYTES = 64


class ArrayAllocator:
    def __init__(self, block_bytes: int = ALLOCATOR_BLOCK_BYTES):
        self._block_bytes: int = block_bytes
        self._free_blocks: defaultdict[int, list[mmap.mmap]] = defaultdict(list)
        self._current_block: Lease | None = None
        self._bytes_used: int = 0
        self._lock: threading.RLock = threading.RLock()

    def __reduce__(self):
        return ArrayAllocator, (self._block_bytes,)

    def empty(self, shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
        dtype = np.dtype(dtype)
        assert dtype.kind != "O", dtype
        nbytes = math.prod(shape) * dtype.itemsize

        with self._lock:
            if nbytes > self._block_bytes:
                block = self._take_block(-(-nbytes // self._block_bytes) * self._block_bytes)
                return np.ndarray(shape, dtype, buffer=block)

            offset = -(-self._bytes_used // ARRAY_ALIGNMENT_BYTES) * ARRAY_ALIGNMENT_BYTES

            if self._current_block is None or offset + nbytes > self._block_bytes:
                self._current_block = self._take_block(self._block_bytes)
                offset = 0

            out = np.ndarray(shape, dtype, buffer=self._current_block, offset=offset)
            self._bytes_used = offset + nbytes
            return out

    def zeros(self, shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
        return self.full(shape, 0, dtype)

    def ones(self, shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
        return self.full(shape, 1, dtype)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: npt.DTypeLike) -> np.ndarray:
        out = self.empty(shape, dtype)
        out.fill(fill_value)
        return out

    def copy(self, src: np.ndarray, dtype: npt.DTypeLike) -> np.ndarray:
        out = self.empty(src.shape, dtype)
        np.copyto(out, src, casting="unsafe")
        return out

    def _take_block(self, nbytes: int) -> "Lease":
        free = self._free_blocks[nbytes]
        return Lease(free.pop() if free else self._map_block(nbytes), free, self._lock)

    @staticmethod
    def _map_block(nbytes: int) -> mmap.mmap:
        mem = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

        try:
            mem.madvise(23)
        except OSError:
            np.frombuffer(mem, np.uint8)[:: mmap.PAGESIZE] = 0

        return mem


class Lease:
    def __init__(self, mem: mmap.mmap, free: list[mmap.mmap], lock: threading.RLock):
        self._mem: mmap.mmap = mem
        weakref.finalize(self, Lease._release, mem, free, lock).atexit = False

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._mem)

    @staticmethod
    def _release(mem: mmap.mmap, free: list[mmap.mmap], lock: threading.RLock) -> None:
        with lock:
            free.append(mem)
