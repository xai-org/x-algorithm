import logging

from monitor.metrics import Metrics

from grox.config.config import ModelName
from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.ptos.classifier import SafetyPtosPolicyClassifier
from grox.flows.ptos.constants import GEMMA_PTOS_REALTIME
from grox.flows.ptos.mode import SafetyPtosMode
from grox.flows.ptos.state import (
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPostAnnotations,
    SafetyPtosState,
    SafetyPtosViolatedPolicy,
)

logger = logging.getLogger(__name__)


class TaskPtosSpamDetection(TaskWithPost):
    policy_classifier = SafetyPtosPolicyClassifier(
        model_name=ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
        mode=SafetyPtosMode.STANDARD,
        use_gemma=True,
        gemma_model_name=GEMMA_PTOS_REALTIME,
    )

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        violation = SafetyPtosViolatedPolicy(
            category=SafetyPolicyCategory.Spam,
            reason="Real Time Signals Flow Triggered for spam detection",
        )
        safety_policy = await cls.policy_classifier.classify_policy_for_violation(
            post, violation
        )

        Metrics.counter("task.ptos_spam_detection.classified.count").add(1)

        if safety_policy and safety_policy.policyType != SafetyPolicyType.NoViolation:
            Metrics.counter("task.ptos_spam_detection.violation.count").add(
                1, attributes={"policy_type": safety_policy.policyType.name}
            )
            logger.info(
                f"Post {post.id}: PtosSpamViolation DetectedViolation - {safety_policy.policyType.name}"
            )
            violation.safetyPolicy = safety_policy
            ctx.state(SafetyPtosState).annotations = SafetyPostAnnotations(
                violatedPolicies=[violation]
            )
        else:
            Metrics.counter("task.ptos_spam_detection.no_violation.count").add(1)
            logger.info(f"Post {post.id}: PtosSpamViolation NoViolation")
            ctx.state(SafetyPtosState).annotations = SafetyPostAnnotations(
                violatedPolicies=[]
            )

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
