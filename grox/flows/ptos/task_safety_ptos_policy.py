import logging

from grox.core.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.core.schedules.types import TaskContext
from grox.core.data_loaders.data_types import Post
from grox.core.lm.post import PostRenderer
from grox.flows.ptos.state import (
    SafetyPolicy,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPtosState,
    SafetyPtosViolatedPolicy,
)
from grox.flows.ptos.classifier import (
    SafetyPtosChildSafetyPolicyClassifier,
    SafetyPtosPolicyClassifier,
    SafetyPtosPolicyCrossValidator,
)
from grox.config.config import ModelName
from grox.flows.ptos.mode import SafetyPtosMode
from grox.flows.ptos.constants import GEMMA, HIGH_FAV_THRESHOLD
from grox.flows.ptos.prior_nsfw import post_is_already_flagged_nsfw

logger = logging.getLogger(__name__)


class TaskSafetyPtosPolicyDetection(TaskWithPost):
    child_safety_classifier = SafetyPtosChildSafetyPolicyClassifier(
        gemma_model_name=GEMMA
    )
    cross_validator = SafetyPtosPolicyCrossValidator()

    classifiers = {
        SafetyPtosMode.STANDARD: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.STANDARD,
        ),
        SafetyPtosMode.DELUXE: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.DELUXE,
        ),
        SafetyPtosMode.RECOVERY: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG,
            mode=SafetyPtosMode.RECOVERY,
        ),
        SafetyPtosMode.BACKFILL: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_X_ALGO_EXTRA,
            mode=SafetyPtosMode.BACKFILL,
            gemma_model_name=None,
        ),
        SafetyPtosMode.LIVE_CLUSTER_ANCHORS: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.LIVE_CLUSTER_ANCHORS,
        ),
    }

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        annotations = ctx.state(SafetyPtosState).annotations
        if not annotations:
            return

        mode = SafetyPtosMode.from_task_type(ctx.payload.task_type)
        active_classifier = cls.classifiers[mode]
        metric_prefix = (
            "task.safety_ptos_deluxe_policy"
            if mode.is_deluxe
            else "task.safety_ptos_policy"
        )

        violations = list(annotations.violatedPolicies or [])

        for violation in violations:
            try:
                if (
                    violation.category == SafetyPolicyCategory.AdultContent
                    and await post_is_already_flagged_nsfw(ctx, post)
                ):
                    Metrics.counter(
                        f"{metric_prefix}.skipped_adult_content_already_nsfw.count"
                    ).add(1)
                    violation.safetyPolicy = SafetyPolicy(
                        policyType=SafetyPolicyType.AdultContentSexualHard,
                        reason="post was previously flagged nsfw by ptos",
                    )
                elif violation.category == SafetyPolicyCategory.ChildSafety:
                    policy = await cls.child_safety_classifier.classify_policy(post)
                    violation.safetyPolicy = await cls.cross_validator.validate(
                        violation.category, post, policy
                    )
                elif violation.category in (
                    SafetyPolicyCategory.ViolentMedia,
                    SafetyPolicyCategory.IllegalAndRegulatedBehaviors,
                ):
                    policy = await active_classifier.classify_policy_for_violation(
                        post, violation
                    )
                    violation.safetyPolicy = await cls.cross_validator.validate(
                        violation.category, post, policy
                    )
                else:
                    violation.safetyPolicy = (
                        await active_classifier.classify_policy_for_violation(
                            post, violation
                        )
                    )
            except Exception as e:
                Metrics.counter(f"{metric_prefix}.classify_error.count").add(
                    1, attributes={"category": violation.category.value}
                )
                if not mode.graceful_classify_errors:
                    raise
                logger.warning(
                    f"Post {post.id}: policy classification failed for {violation.category.value}, continuing: {e}"
                )
                continue
            if violation.safetyPolicy:
                cls._record_policy_metrics(metric_prefix, violation)

        recheck = await cls._high_fav_adult_recheck(
            ctx, post, mode, active_classifier, violations, metric_prefix
        )
        if recheck is not None:
            violations.append(recheck)

        annotations.violatedPolicies = violations

    @classmethod
    async def _high_fav_adult_recheck(
        cls,
        ctx: TaskContext,
        post: Post,
        mode: SafetyPtosMode,
        active_classifier: SafetyPtosPolicyClassifier,
        violations: list[SafetyPtosViolatedPolicy],
        metric_prefix: str,
    ) -> SafetyPtosViolatedPolicy | None:
        if not mode.is_deluxe or post.get_fav_count() < HIGH_FAV_THRESHOLD:
            return None
        if any(v.category == SafetyPolicyCategory.AdultContent for v in violations):
            return None
        if not PostRenderer.has_media(post) or await post_is_already_flagged_nsfw(
            ctx, post
        ):
            return None

        recheck = SafetyPtosViolatedPolicy(
            category=SafetyPolicyCategory.AdultContent,
            reason="high-fav adult content recheck",
            score=50,
        )
        try:
            policy = await active_classifier.classify_policy_for_violation(
                post, recheck
            )
        except Exception as e:
            Metrics.counter(f"{metric_prefix}.classify_error.count").add(
                1, attributes={"category": recheck.category.value}
            )
            logger.warning(
                f"Post {post.id}: high-fav adult recheck failed, continuing: {e}"
            )
            return None
        if policy is None:
            return None
        recheck.safetyPolicy = policy
        Metrics.counter(f"{metric_prefix}.high_fav_adult_recheck.count").add(
            1, attributes={"policy_type": policy.policyType.name}
        )
        if policy.policyType == SafetyPolicyType.NoViolation:
            return None
        cls._record_policy_metrics(metric_prefix, recheck)
        return recheck

    @classmethod
    def _record_policy_metrics(
        cls, metric_prefix: str, violation: SafetyPtosViolatedPolicy
    ) -> None:
        Metrics.counter(f"{metric_prefix}.classified_total.count").add(1)
        category_key = {
            SafetyPolicyCategory.ViolentMedia: "violent_media",
            SafetyPolicyCategory.AdultContent: "adult_content",
            SafetyPolicyCategory.Spam: "spam",
            SafetyPolicyCategory.IllegalAndRegulatedBehaviors: "illegal_and_regulated_behaviors",
            SafetyPolicyCategory.HateOrAbuse: "hate_or_abuse",
            SafetyPolicyCategory.ViolentSpeech: "violent_speech",
            SafetyPolicyCategory.SuicideOrSelfHarm: "suicide_or_self_harm",
            SafetyPolicyCategory.ChildSafety: "child_safety",
            SafetyPolicyCategory.AgeGatingHarmfulText: "age_gating_harmful_text",
        }.get(violation.category)
        if category_key:
            Metrics.counter(
                f"{metric_prefix}.classified_{category_key}_violations.count"
            ).add(1)
            Metrics.counter(
                f"{metric_prefix}.classified_{category_key}_policy_types.count"
            ).add(1, attributes={"policy_type": violation.safetyPolicy.policyType.name})

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
