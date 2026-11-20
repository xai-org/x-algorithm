# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import collections
import dataclasses
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol, final

import numpy as np
from numpy import typing as npt

from xrex.data import rust_ext

rank_logger = logging.getLogger("rank")

URL_EMBEDDING_DIM = 1024
_AVAILABLE_AT_BYTES = 8
URL_EMBEDDING_VALUE_BYTES = _AVAILABLE_AT_BYTES + 2 * URL_EMBEDDING_DIM
_ERROR_BACKOFF_S = 10.0
_LOG_INTERVAL_S = 300.0
_XDS_BOOTSTRAP_ENVS = ("GRPC_XDS_BOOTSTRAP", "GRPC_XDS_BOOTSTRAP_CONFIG")


DEFAULT_MODEL_VERSION = "v8_2"


def url_embedding_key(url_id: int, model_version: str = DEFAULT_MODEL_VERSION) -> str:
    return f"url_{url_id}_{model_version}_emb"


def post_embedding_key(post_id: int, model_version: str) -> str:
    return f"{post_id}_{model_version}_emb"


def parse_url_embedding_value(value: bytes | None) -> npt.NDArray[np.float16] | None:
    if value is None or len(value) != URL_EMBEDDING_VALUE_BYTES:
        return None
    return np.frombuffer(value, dtype="<f2", offset=_AVAILABLE_AT_BYTES).astype(np.float16)


class MGetClient(Protocol):
    def mget(self, keys: list[str]) -> tuple[Sequence[bytes | None], Sequence[int]]: ...


@dataclasses.dataclass
class UrlEmbeddingLookupStats:
    rows: int = 0
    sampled_out_rows: int = 0
    positions: int = 0
    filled_positions: int = 0
    fallback_filled_positions: int = 0
    unique_ids: int = 0
    cache_hits: int = 0
    negative_cache_hits: int = 0
    redis_hits: int = 0
    redis_misses: int = 0
    redis_unanswered: int = 0
    mget_calls: int = 0
    mget_errors: int = 0
    mget_skipped: int = 0
    mget_seconds: float = 0.0

    def add(self, other: UrlEmbeddingLookupStats) -> None:
        for field in dataclasses.fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))


@final
class HistoryUrlEmbeddingLookup:
    def __init__(
        self,
        client: MGetClient,
        cache_entries: int,
        sample_rate: float,
        rng: np.random.Generator,
        clock: Callable[[], float],
        negative_ttl_s: float = 0.0,
        key_fn: Callable[[int], str] = url_embedding_key,
        name: str = "History URL embeddings",
    ) -> None:
        if cache_entries < 0:
            raise ValueError(f"cache_entries must be >= 0, got {cache_entries}")
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        if negative_ttl_s < 0:
            raise ValueError(f"negative_ttl_s must be >= 0, got {negative_ttl_s}")
        self._client = client
        self._key_fn = key_fn
        self._name = name
        self._cache_entries = cache_entries
        self._sample_rate = sample_rate
        self._rng = rng
        self._clock = clock
        self._slots: collections.OrderedDict[int, int] = collections.OrderedDict()
        self._vectors = np.zeros((cache_entries, URL_EMBEDDING_DIM), np.float16)
        self._negative_ttl_s = negative_ttl_s
        self._absent: collections.OrderedDict[int, float] = collections.OrderedDict()
        self._mget_paused_until = float("-inf")
        self._pending = UrlEmbeddingLookupStats()
        self._totals = UrlEmbeddingLookupStats()
        self._last_log = clock()
        self._lock = threading.Lock()

    @classmethod
    def from_cache(
        cls,
        cache: str,
        dc: str,
        timeout_s: float,
        cache_entries: int,
        sample_rate: float,
        negative_ttl_s: float,
        key_fn: Callable[[int], str] = url_embedding_key,
        name: str = "History URL embeddings",
    ) -> HistoryUrlEmbeddingLookup:
        source = next((name for name in _XDS_BOOTSTRAP_ENVS if os.environ.get(name)), "no")
        client = rust_ext.load("xai_recsys_kafka_reader").XdsRedisMget(cache, dc, timeout_s)
        endpoints = client.endpoint_count()
        if endpoints == 0:
            raise RuntimeError(
                f"xDS discovery found no endpoints for Dragonfly {cache!r} in {dc!r} with the "
                f"{source} bootstrap; check its discovery server"
            )
        rank_logger.info(
            "%s cache %s (%s, bootstrap %s): %d endpoints",
            name,
            cache,
            dc,
            source,
            endpoints,
        )
        return cls(
            client,
            cache_entries,
            sample_rate,
            np.random.default_rng(),
            time.monotonic,
            negative_ttl_s,
            key_fn,
            name,
        )

    def __call__(
        self,
        url_ids: npt.NDArray[np.int64],
        valid: npt.NDArray[np.bool_],
        out: npt.NDArray[np.float16],
    ) -> None:
        batch_size, history_len, _ = url_ids.shape
        assert out.shape == (batch_size, history_len, URL_EMBEDDING_DIM), out.shape
        stats = UrlEmbeddingLookupStats(rows=batch_size)
        with self._lock:
            sampled = self._rng.random(batch_size) < self._sample_rate
        stats.sampled_out_rows = batch_size - int(sampled.sum())
        rows, cols, ranks = np.nonzero(valid & sampled[:, None, None])
        positions = rows * history_len + cols
        stats.positions = len(np.unique(positions))
        if len(rows) > 0:
            unique_ids, inverse = np.unique(url_ids[rows, cols, ranks], return_inverse=True)
            hit, vectors = self._lookup(unique_ids, stats)
            usable = hit[inverse]
            filled, first = np.unique(positions[usable], return_index=True)
            chosen = np.flatnonzero(usable)[first]
            out[rows[chosen], cols[chosen]] = vectors[inverse[chosen]]
            stats.filled_positions = len(filled)
            stats.fallback_filled_positions = int((ranks[chosen] > 0).sum())
        with self._lock:
            self._pending.add(stats)
            self._totals.add(stats)
            self._maybe_log()

    def take_stats(self) -> UrlEmbeddingLookupStats:
        with self._lock:
            stats, self._pending = self._pending, UrlEmbeddingLookupStats()
        return stats

    def _lookup(
        self, unique_ids: npt.NDArray[np.int64], stats: UrlEmbeddingLookupStats
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float16]]:
        hit = np.zeros(len(unique_ids), np.bool_)
        vectors = np.zeros((len(unique_ids), URL_EMBEDDING_DIM), np.float16)
        misses: list[int] = []
        hit_indices: list[int] = []
        hit_slots: list[int] = []
        with self._lock:
            now = self._clock()
            for index, url_id in enumerate(unique_ids.tolist()):
                slot = self._slots.get(url_id)
                if slot is not None:
                    self._slots.move_to_end(url_id)
                    hit_indices.append(index)
                    hit_slots.append(slot)
                    continue
                absent_until = self._absent.get(url_id)
                if absent_until is not None:
                    if now < absent_until:
                        stats.negative_cache_hits += 1
                        continue
                    del self._absent[url_id]
                misses.append(index)
            hit[hit_indices] = True
            vectors[hit_indices] = self._vectors[hit_slots]
            paused = now < self._mget_paused_until
        stats.unique_ids = len(unique_ids)
        stats.cache_hits = len(unique_ids) - len(misses) - stats.negative_cache_hits
        if not misses:
            return hit, vectors
        if paused:
            stats.mget_skipped = 1
            return hit, vectors
        miss_ids = unique_ids[misses].tolist()
        start = self._clock()
        stats.mget_calls = 1
        try:
            values, failed = self._client.mget([self._key_fn(url_id) for url_id in miss_ids])
        except OSError as error:
            self._pause_mgets(stats, f"no shard answered: {error!r}")
            return hit, vectors
        finally:
            stats.mget_seconds = self._clock() - start
        unanswered = set(failed)
        stats.redis_unanswered = len(unanswered)
        if unanswered:
            self._pause_mgets(stats, f"{len(unanswered)}/{len(miss_ids)} keys unanswered")
        found: list[tuple[int, npt.NDArray[np.float16]]] = []
        absent: list[int] = []
        for position, (index, url_id, value) in enumerate(
            zip(misses, miss_ids, values, strict=True)
        ):
            if position in unanswered:
                continue
            vector = parse_url_embedding_value(value if isinstance(value, bytes) else None)
            if vector is None:
                stats.redis_misses += 1
                absent.append(url_id)
                continue
            stats.redis_hits += 1
            hit[index] = True
            vectors[index] = vector
            found.append((url_id, vector))
        if self._cache_entries > 0 and (found or absent):
            with self._lock:
                for url_id, vector in found:
                    self._absent.pop(url_id, None)
                    slot = self._slots.get(url_id)
                    if slot is not None:
                        self._slots.move_to_end(url_id)
                    else:
                        if len(self._slots) < self._cache_entries:
                            slot = len(self._slots)
                        else:
                            _, slot = self._slots.popitem(last=False)
                        self._slots[url_id] = slot
                    self._vectors[slot] = vector
                if self._negative_ttl_s > 0 and absent:
                    absent_until = self._clock() + self._negative_ttl_s
                    for url_id in absent:
                        self._absent.pop(url_id, None)
                        self._absent[url_id] = absent_until
                    while len(self._absent) > self._cache_entries:
                        self._absent.popitem(last=False)
        return hit, vectors

    def _pause_mgets(self, stats: UrlEmbeddingLookupStats, reason: str) -> None:
        stats.mget_errors = 1
        with self._lock:
            self._mget_paused_until = self._clock() + _ERROR_BACKOFF_S
            first_error = self._totals.mget_errors == 0
        if first_error:
            rank_logger.warning(
                "%s MGET failed (%s); misses read zeros for %.0fs",
                self._name,
                reason,
                _ERROR_BACKOFF_S,
            )

    def _maybe_log(self) -> None:
        now = self._clock()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        self._last_log = now
        totals = self._totals
        rank_logger.info(
            "%s: rows=%d sampled_out=%d positions=%d filled=%d "
            "fallback_filled=%d unique_ids=%d cache_hits=%d "
            "negative_cache_hits=%d redis_hits=%d redis_misses=%d redis_unanswered=%d "
            "mget_calls=%d mget_errors=%d mget_skipped=%d mget_mean_ms=%.1f cache_size=%d "
            "negative_cache_size=%d",
            self._name,
            totals.rows,
            totals.sampled_out_rows,
            totals.positions,
            totals.filled_positions,
            totals.fallback_filled_positions,
            totals.unique_ids,
            totals.cache_hits,
            totals.negative_cache_hits,
            totals.redis_hits,
            totals.redis_misses,
            totals.redis_unanswered,
            totals.mget_calls,
            totals.mget_errors,
            totals.mget_skipped,
            1000.0 * totals.mget_seconds / max(totals.mget_calls, 1),
            len(self._slots),
            len(self._absent),
        )
