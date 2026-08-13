from abc import ABC

from grox.config.env import is_prod

from grox.core.schedules.types import TaskContext


class DisableTaskRule(ABC):
    DISABLE_REASON: str = ""

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return False

    @classmethod
    def disable_reason(cls) -> str | None:
        return cls.DISABLE_REASON


class DisableTaskForNonProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_prod
