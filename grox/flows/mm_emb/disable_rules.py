from grox.config.env import is_mm_emb_prod

from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskRule


class DisableTaskForNonMmEmbProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-mm-emb-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_mm_emb_prod
