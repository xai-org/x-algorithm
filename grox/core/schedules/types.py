import time
from typing import Any

from pydantic import BaseModel, Field

from grox.core.data_loaders.data_types import (
    Post,
    User,
)


def _ext_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class ExtendableModel(BaseModel):
    extensions: dict[str, Any] = Field(default_factory=dict)

    def ext[T](self, cls: type[T]) -> T | None:
        return self.extensions.get(_ext_key(cls))

    def set_ext(self, value: Any) -> None:
        self.extensions[_ext_key(type(value))] = value


class TaskPayload(ExtendableModel):
    payload_id: str
    post: Post | None = None
    user: User | None = None
    plans: set[str] = Field(default_factory=set)
    task_type: str | None = None
    embedding: list[float] | None = None


class TaskResult(BaseModel):
    task: TaskPayload
    task_started_at: float
    task_finished_at: float = Field(default_factory=time.perf_counter)
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class TaskContext:
    def __init__(self, task: TaskPayload):
        self._states: dict[type[Any], Any] = {}
        self.payload = task
        self.plans: set[str] = task.plans.copy()
        self.start_time = time.perf_counter()
        self.errors: list[Exception] = []

    def state[T](self, cls: type[T]) -> T:
        try:
            return self._states[cls]
        except KeyError:
            instance = cls()
            self._states[cls] = instance
            return instance
