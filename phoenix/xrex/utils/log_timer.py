# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, overload

logger = logging.getLogger(__name__)


def duration(secs: float) -> str:
    neg = "-" if secs < 0 else ""
    secs = abs(secs)

    if secs < 1:
        fmt = f"{secs * 1e3:.0f}ms"
    elif secs < 60:
        fmt = f"{secs:.2f}s"
    elif secs < 60 * 60:
        mins, secs = divmod(secs, 60)
        fmt = f"{mins:.0f}min{secs:.0f}s"
    else:
        hrs, secs = divmod(secs, 60 * 60)
        mins, secs = divmod(secs, 60)
        fmt = f"{hrs:.0f}h{mins:.0f}min{secs:.0f}s"

    return f"{neg}{fmt}"


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def __repr__(self) -> str:
        return duration(self.elapsed())

    def elapsed(self) -> float:
        return time.time() - self.start

    __float__ = elapsed

    def __rtruediv__(self, dividend):
        return dividend / float(self)


class C(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


F = TypeVar("F", bound=C)


@overload
def log_elapsed_time(logger_or_func: logging.Logger) -> Callable[[F], F]: ...


@overload
def log_elapsed_time(logger_or_func: F) -> F: ...


def log_elapsed_time(logger_or_func: logging.Logger | Callable[[F], F]) -> F | Callable[[F], F]:
    def decorator(f: F) -> F:
        def wrap(*args: Any, **kwargs: Any) -> Any:
            timer = Timer()
            try:
                return f(*args, **kwargs)
            finally:
                elapsed = timer.elapsed()
                if elapsed > 0.001:
                    target_logger = (
                        logger_or_func
                        if logger_or_func and not callable(logger_or_func)
                        else logger
                    )
                    target_logger.info(f"{f.__qualname__:s} function took {duration(elapsed)}")

        return wrap

    if callable(logger_or_func):
        return decorator(logger_or_func)
    return decorator
