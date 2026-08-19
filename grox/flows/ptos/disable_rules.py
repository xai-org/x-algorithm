from grox.config.env import is_ptos_prod

from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskRule


class DisableTaskForNonPtosProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-ptos-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_ptos_prod
