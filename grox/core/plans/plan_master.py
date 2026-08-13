import asyncio
import time

import grox.core.loading
from grox.core.plans.plan import Plan
from grox.core.plans.registry import registered_plans
from grox.core.schedules.types import TaskPayload, TaskResult


class PlanMaster:
    _plans: list[Plan] | None = None

    @classmethod
    def _all_plans(cls) -> list[Plan]:
        if cls._plans is None:
            if not grox.core.loading._loaded:
                raise RuntimeError(
                    "grox.core.loading.load_all() has not run in this process; call it at startup before executing plans"
                )
            cls._plans = [plan_cls() for plan_cls in registered_plans()]
        return cls._plans

    @classmethod
    async def exec(cls, task: TaskPayload) -> TaskResult:
        results = await asyncio.gather(*[p.execute(task) for p in cls._all_plans()])
        result = cls.merge_results(task, [r for r in results if r is not None])
        return result

    @classmethod
    def merge_results(cls, task: TaskPayload, results: list[TaskResult]) -> TaskResult:
        if not results:
            now = time.perf_counter()
            return TaskResult(task=task, task_started_at=now, task_finished_at=now)
        return TaskResult(
            task=task,
            task_started_at=min(r.task_started_at for r in results),
            task_finished_at=max(r.task_finished_at for r in results),
            success=all(r.success for r in results),
            error="\n".join(
                [r.error or "unknown error" for r in results if not r.success]
            ),
        )
