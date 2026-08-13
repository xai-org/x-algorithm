import logging

from monitor.metrics import Metrics
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
)

from grox.config.config import ModelName
from grox.core.data_loaders.data_types import Post
from grox.core.lm.post import PostRenderer
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.ptos.classifier import (
    SafetyPtosChildSafetyPolicyClassifier,
    SafetyPtosPolicyClassifier,
)
from grox.flows.ptos.constants import GEMMA, HIGH_FAV_THRESHOLD
from grox.flows.ptos.mode import SafetyPtosMode
from grox.flows.ptos.state import (
    SafetyPolicy,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPtosState,
    SafetyPtosViolatedPolicy,
)

logger = logging.getLogger(__name__)


class TaskSafetyPtosPolicyDetection(TaskWithPost):
    child_safety_classifier = SafetyPtosChildSafetyPolicyClassifier(
        gemma_model_name=GEMMA
    )

    classifiers = {
        SafetyPtosMode.STANDARD: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.STANDARD,
            use_gemma=True,
            gemma_model_name=GEMMA,
        ),
        SafetyPtosMode.DELUXE: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.DELUXE,
            use_gemma=False,
            gemma_model_name=GEMMA,
        ),
        SafetyPtosMode.RECOVERY: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG,
            mode=SafetyPtosMode.RECOVERY,
            use_gemma=True,
            gemma_model_name=GEMMA,
        ),
        SafetyPtosMode.BACKFILL: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_X_ALGO_EXTRA,
            mode=SafetyPtosMode.BACKFILL,
            use_gemma=False,
            gemma_model_name=GEMMA,
        ),
        SafetyPtosMode.LIVE_CLUSTER_ANCHORS: SafetyPtosPolicyClassifier(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL,
            mode=SafetyPtosMode.LIVE_CLUSTER_ANCHORS,
            use_gemma=True,
            gemma_model_name=GEMMA,
        ),
    }

    _result_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()

    @classmethod
    async def _post_is_already_flagged_nsfw(cls, post: Post) -> bool:
        try:
            result = await cls._result_direct_mh.fetch(int(post.id))
        except Exception as e:
            Metrics.counter(
                "task.safety_ptos_deluxe_policy.nsfw_lookup_error.count"
            ).add(1)
            logger.warning(
                f"Post {post.id}: NSFW MH lookup failed, treating as not flagged: {e}"
            )
            return False
        return bool(
            result and result.safetyBoolMetadata and result.safetyBoolMetadata.isNsfw
        )

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

        injected_recheck = None
        if mode.is_deluxe and post.get_fav_count() >= HIGH_FAV_THRESHOLD:
            if (
                not any(
                    v.category == SafetyPolicyCategory.AdultContent for v in violations
                )
                and PostRenderer.has_media(post)
                and not await cls._post_is_already_flagged_nsfw(post)
            ):
                injected_recheck = SafetyPtosViolatedPolicy(
                    category=SafetyPolicyCategory.AdultContent,
                    reason="high-fav adult content recheck",
                    score=50,
                )
                violations.append(injected_recheck)

        for violation in violations:
            if (
                violation.category == SafetyPolicyCategory.AdultContent
                and await cls._post_is_already_flagged_nsfw(post)
            ):
                Metrics.counter(
                    f"{metric_prefix}.skipped_adult_content_already_nsfw.count"
                ).add(1)
                violation.safetyPolicy = SafetyPolicy(
                    policyType=SafetyPolicyType.AdultContentSexualHard,
                    reason="post was previously flagged nsfw by ptos",
                )
            elif violation.category == SafetyPolicyCategory.ChildSafety:
                violation.safetyPolicy = (
                    await cls.child_safety_classifier.classify_policy(post)
                )
            else:
                violation.safetyPolicy = (
                    await active_classifier.classify_policy_for_violation(
                        post, violation
                    )
                )
            if violation.safetyPolicy:
                cls._record_policy_metrics(metric_prefix, violation)

        if injected_recheck is not None:
            policy = injected_recheck.safetyPolicy
            if not policy or policy.policyType == SafetyPolicyType.NoViolation:
                violations.remove(injected_recheck)

        annotations.violatedPolicies = violations

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
