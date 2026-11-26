# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Protocol, final

logger = logging.getLogger("rank")

WINDOW_UNKNOWN = 1 << 62

COORDINATION_TIMEOUT_MS = 2 * 60 * 60 * 1000

MAX_FAILED_COUNTS = 3


@dataclass(frozen=True)
class DataEnd:
    end_batch_id: int
    max_steps: int


class Coordination(Protocol):
    def max_over_processes(self, key: str, round_: int, value: int) -> int: ...

    def broadcast_from_process_0(self, key: str, round_: int, value: str | None) -> str: ...


@final
class KeyValueCoordination:
    def __init__(self, client, process_index: int, process_count: int):
        self._client = client
        self._process_index = process_index
        self._process_count = process_count

    def max_over_processes(self, key: str, round_: int, value: int) -> int:
        from xrex.train.trainer_recsys import max_over_processes

        return max_over_processes(self._client, key, round_, value, COORDINATION_TIMEOUT_MS)

    def broadcast_from_process_0(self, key: str, round_: int, value: str | None) -> str:
        if self._process_count == 1:
            assert value is not None
            return value
        round_key = f"{key}/{round_}"
        if self._process_index == 0:
            assert value is not None
            self._client.key_value_set(round_key, value)
            if round_ > 0:
                self._client.key_value_delete(f"{key}/{round_ - 1}")
        return self._client.blocking_key_value_get(round_key, COORDINATION_TIMEOUT_MS)


_ERROR = "error"


def agree(
    coordination: Coordination, key: str, round_: int, decide: Callable[[], dict] | None
) -> dict:
    failure: Exception | None = None
    message: str | None = None
    if decide is not None:
        try:
            message = json.dumps(decide())
        except Exception as e:
            failure = e
            cause = f" ({e.__cause__!r})" if e.__cause__ is not None else ""
            message = json.dumps({_ERROR: f"{type(e).__name__}: {e}{cause}"})
    payload = json.loads(coordination.broadcast_from_process_0(key, round_, message))
    if failure is not None:
        raise failure
    if _ERROR in payload:
        raise RuntimeError(f"follow_data_end: process 0 failed: {payload[_ERROR]}")
    return payload


def can_move_end(end_batch_id: int, furthest_window_start: int, window: int) -> bool:
    last_window_start = ((end_batch_id - 1) // window) * window
    return furthest_window_start + window < last_window_start


@final
class _Probe:
    def __init__(self, probe: Callable[[int], DataEnd | None], end_batch_id: int):
        self.result: DataEnd | None = None
        self.error: BaseException | None = None
        self._done = threading.Event()

        def run() -> None:
            try:
                self.result = probe(end_batch_id)
            except BaseException as e:
                self.error = e
            finally:
                self._done.set()

        threading.Thread(target=run, name="follow_data_end_probe", daemon=True).start()

    def done(self) -> bool:
        return self._done.is_set()


@final
class DataEndFollower:
    def __init__(
        self,
        *,
        data_end: DataEnd,
        window: int,
        check_every_n: int,
        is_process_0: bool,
        probe: Callable[[int], DataEnd | None],
        coordination: Coordination,
        key_prefix: str = "follow_data_end",
    ):
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if check_every_n <= 0:
            raise ValueError(f"check_every_n must be positive, got {check_every_n}")
        self._data_end = data_end
        self._window = window
        self._check_every_n = check_every_n
        self._is_process_0 = is_process_0
        self._probe_fn = probe
        self._coordination = coordination
        self._key_prefix = key_prefix
        self._round = 0
        self._frozen = False
        self._failed_counts = 0
        self._probe: _Probe | None = None
        if is_process_0:
            self._probe = _Probe(probe, data_end.end_batch_id)

    @property
    def data_end(self) -> DataEnd:
        return self._data_end

    @property
    def frozen(self) -> bool:
        return self._frozen

    def check(self, soft_step: int, reader_window_start: int | None) -> DataEnd | None:
        if self._frozen or soft_step == 0 or soft_step % self._check_every_n:
            return None

        furthest = self._coordination.max_over_processes(
            f"{self._key_prefix}/window",
            self._round,
            WINDOW_UNKNOWN if reader_window_start is None else reader_window_start,
        )
        round_ = self._round
        self._round += 1
        decision = agree(
            self._coordination,
            f"{self._key_prefix}/decision",
            round_,
            (lambda: self._decide(furthest)) if self._is_process_0 else None,
        )

        if decision.get("frozen"):
            self._frozen = True
            self._probe = None
            logger.info(
                "follow_data_end: a reader reached window %d, within one window of the end; "
                "the data end stays at batch_id %d (max_steps=%d)",
                furthest,
                self._data_end.end_batch_id,
                self._data_end.max_steps,
            )
            return None
        if "end_batch_id" not in decision:
            return None

        new = DataEnd(end_batch_id=decision["end_batch_id"], max_steps=decision["max_steps"])
        old = self._data_end
        self._data_end = new
        logger.info(
            "follow_data_end: data end moved from batch_id %d to %d, max_steps %d -> %d",
            old.end_batch_id,
            new.end_batch_id,
            old.max_steps,
            new.max_steps,
        )
        if self._is_process_0:
            self._probe = _Probe(self._probe_fn, new.end_batch_id)
        return new

    def _decide(self, furthest_window_start: int) -> dict:
        if furthest_window_start == WINDOW_UNKNOWN:
            return {}
        if not can_move_end(self._data_end.end_batch_id, furthest_window_start, self._window):
            return {"frozen": True}

        probe = self._probe
        if probe is None or not probe.done():
            return {}
        if probe.error is not None:
            self._failed_counts += 1
            if self._failed_counts >= MAX_FAILED_COUNTS:
                raise RuntimeError(
                    "follow_data_end: counting steps to the new data end failed "
                    f"{self._failed_counts} times in a row"
                ) from probe.error
            logger.warning(
                "follow_data_end: counting steps to the new data end failed (%d of %d in a row); "
                "trying again at the next check: %r",
                self._failed_counts,
                MAX_FAILED_COUNTS,
                probe.error,
            )
            self._probe = _Probe(self._probe_fn, self._data_end.end_batch_id)
            return {}
        self._failed_counts = 0
        found = probe.result
        if found is None or found.end_batch_id <= self._data_end.end_batch_id:
            self._probe = _Probe(self._probe_fn, self._data_end.end_batch_id)
            return {}
        if found.max_steps < self._data_end.max_steps:
            raise RuntimeError(
                f"follow_data_end: more data ({self._data_end.end_batch_id} -> "
                f"{found.end_batch_id} batch_ids) counted fewer steps "
                f"({self._data_end.max_steps} -> {found.max_steps})"
            )
        return {"end_batch_id": found.end_batch_id, "max_steps": found.max_steps}
