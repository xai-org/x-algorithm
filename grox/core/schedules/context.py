import logging
import signal
from multiprocessing import Manager
from multiprocessing.managers import DictProxy, SyncManager
from typing import Any

type ScheduleContext = DictProxy[str, Any]
logger = logging.getLogger(__name__)
_manager: SyncManager | None = None


def get_manager() -> SyncManager:
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


def new_context() -> ScheduleContext:
    manager = get_manager()
    return manager.dict(
        task_queue=manager.Queue(),
        resp_queue=manager.Queue(),
        shutdown_event=manager.Event(),
        queue_connection_shutdown_event=manager.Event(),
    )


def prevent_default() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def shutdown_context(context: ScheduleContext) -> None:
    context["shutdown_event"].set()


def queue_connection_shutdown_context(context: ScheduleContext) -> None:
    context["queue_connection_shutdown_event"].set()


def cleanup() -> None:
    if _manager is not None:
        logger.warning("Shutting down context manager")
        _manager.shutdown()
