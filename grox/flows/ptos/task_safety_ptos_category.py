import logging

from monitor.metrics import Metrics

from grox.config.config import ModelName
from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.ptos.classifier import SafetyPtosCategoryClassifier
from grox.flows.ptos.mode import SafetyPtosMode
from grox.flows.ptos.state import SafetyPtosState

logger = logging.getLogger(__name__)


class TaskSafetyPtosCategoryDetection(TaskWithPost):
    classifiers = {
        SafetyPtosMode.STANDARD: SafetyPtosCategoryClassifier(
            ModelName.GROK_4_MINI_CRITICAL_SAFETY,
            mode=SafetyPtosMode.STANDARD,
            use_gemma=True,
        ),
        SafetyPtosMode.DELUXE: SafetyPtosCategoryClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.DELUXE,
            use_gemma=False,
        ),
        SafetyPtosMode.RECOVERY: SafetyPtosCategoryClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG,
            mode=SafetyPtosMode.RECOVERY,
            use_gemma=True,
        ),
        SafetyPtosMode.BACKFILL: SafetyPtosCategoryClassifier(
            ModelName.GROK_X_ALGO_EXTRA, mode=SafetyPtosMode.BACKFILL, use_gemma=False
        ),
        SafetyPtosMode.LIVE_CLUSTER_ANCHORS: SafetyPtosCategoryClassifier(
            ModelName.GROK_4_MINI_CRITICAL_SAFETY,
            mode=SafetyPtosMode.LIVE_CLUSTER_ANCHORS,
            use_gemma=True,
        ),
    }

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        mode = SafetyPtosMode.from_task_type(ctx.payload.task_type)
        active_classifier = cls.classifiers[mode]
        metric_prefix = (
            "task.safety_ptos_deluxe_category"
            if mode.is_deluxe
            else "task.safety_ptos_category"
        )
        mode_suffix = "" if mode == SafetyPtosMode.STANDARD else f" ({mode.value})"

        safety_annotations = await active_classifier.classify_post(post)
        ctx.state(SafetyPtosState).annotations = safety_annotations

        safety_categories = safety_annotations.violatedPolicies or []
        violation_count = len(safety_categories)
        has_violations = violation_count > 0
        Metrics.counter(f"{metric_prefix}.classified.count").add(1)

        if has_violations:
            Metrics.counter(f"{metric_prefix}.has_violations.count").add(1)
            Metrics.counter(f"{metric_prefix}.violations.count").add(violation_count)

            violation_details = []
            for violation in safety_categories:
                Metrics.counter(f"{metric_prefix}.violations_by_category.count").add(
                    1, attributes={"category": violation.category.value}
                )
                violation_details.append(
                    f"{violation.category.value}({violation.score})"
                )

            logger.info(
                f"Post {post.id}: Found {violation_count} violations{mode_suffix} - Details: {', '.join(violation_details)}"
            )
        else:
            Metrics.counter(f"{metric_prefix}.no_violations.count").add(1)
            logger.info(f"Post {post.id}: No safety violations detected{mode_suffix}")

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
