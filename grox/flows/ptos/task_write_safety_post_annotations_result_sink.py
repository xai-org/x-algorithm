import asyncio
import logging
import time

from monitor.metrics import Metrics
from strato_http.queries.apply_label_from_ptos import (
    SafetyLabelType,
    StratoApplyLabelFromPtos,
)
from strato_http.queries.bounce_post_for_self_harm_encouragement import (
    StratoBouncePostForSelfHarmEncouragement,
)
from strato_http.queries.data_types import (
    FoundMetadata,
    SafetyBoolMetadata,
    SafetyPolicy,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPostAnnotations,
    SafetyPostAnnotationsResult,
    SafetyPtosViolatedPolicy,
)
from strato_http.queries.grok_ptos_delete_labels import StratoGrokPtosDeleteLabels
from strato_http.queries.is_test_user import StratoIsTestUser
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
    StratoSafetyPostAnnotationsResultKafka,
    StratoSafetyPostAnnotationsResultMh,
)
from strato_http.queries.send_support_messages_for_self_harm import (
    StratoSendSupportMessagesForSelfHarm,
)
from strato_http.queries.suspend_user_for_child_sexual_exploitation import (
    StratoSuspendUserForChildSexualExploitation,
)

from grox.core.data_loaders.data_types import Image, Post, Video
from grox.core.data_loaders.strato_loader import UserStratoLoader
from grox.core.lm.post import PostRenderer
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task
from grox.flows.ptos.constants import HIGH_FAV_THRESHOLD
from grox.flows.ptos.disable_rules import DisableTaskForNonPtosProd
from grox.flows.ptos.state import SafetyPtosState

logger = logging.getLogger(__name__)


class TaskWriteSafetyPostAnnotationsResultSink(Task):
    DISABLE_RULES = [DisableTaskForNonPtosProd]

    _strato_mh = StratoSafetyPostAnnotationsResultMh()
    _strato_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()
    _strato_kafka = StratoSafetyPostAnnotationsResultKafka()
    _strato_apply_label_from_ptos = StratoApplyLabelFromPtos()
    _strato_bounce_post_for_self_harm_encouragement = (
        StratoBouncePostForSelfHarmEncouragement()
    )
    _strato_send_support_messages_for_self_harm = StratoSendSupportMessagesForSelfHarm()
    _strato_suspend_user_for_child_sexual_exploitation = (
        StratoSuspendUserForChildSexualExploitation()
    )
    _strato_is_test_user = StratoIsTestUser()
    _strato_grok_ptos_delete_labels = StratoGrokPtosDeleteLabels()

    @staticmethod
    def _build_found_metadata(post) -> FoundMetadata:
        return FoundMetadata(
            imageCount=sum(1 for m in post.media if isinstance(m, Image))
            if post.media
            else 0,
            videoCount=sum(1 for m in post.media if isinstance(m, Video))
            if post.media
            else 0,
            cardV2Count=len(post.cardsV2) if post.cardsV2 else 0,
        )

    @classmethod
    def _compute_bool_metadata_from_violations(
        cls, safety_annotations
    ) -> SafetyBoolMetadata:
        is_gore = False
        is_nsfw = False
        is_soft_nsfw = False
        is_spam = False

        if safety_annotations and safety_annotations.violatedPolicies:
            for violation in safety_annotations.violatedPolicies:
                if (
                    violation.category == SafetyPolicyCategory.ViolentMedia
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_gore = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_gore.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.AdultContent
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    == SafetyPolicyType.AdultContentSexualHard
                ):
                    is_nsfw = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_nsfw.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.AdultContent
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    == SafetyPolicyType.AdultContentSexualSoft
                ):
                    is_soft_nsfw = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_soft_nsfw.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.Spam
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_spam = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_spam.count"
                    ).add(1)

                if (
                    violation.category
                    == SafetyPolicyCategory.IllegalAndRegulatedBehaviors
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_spam = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_spam_illegal.count"
                    ).add(1)

        return SafetyBoolMetadata(
            isGore=is_gore, isNsfw=is_nsfw, isSoftNsfw=is_soft_nsfw, isSpam=is_spam
        )

    @classmethod
    def _merge_bool_metadata(
        cls, existing: SafetyBoolMetadata | None, new: SafetyBoolMetadata
    ) -> SafetyBoolMetadata:
        if existing is None:
            return new
        return SafetyBoolMetadata(
            isGore=True if (existing.isGore or new.isGore) else False,
            isNsfw=True if (existing.isNsfw or new.isNsfw) else False,
            isSoftNsfw=True if (existing.isSoftNsfw or new.isSoftNsfw) else False,
            isSpam=True if (existing.isSpam or new.isSpam) else False,
        )

    @classmethod
    async def _apply_labels_for_annotation(
        cls, annotation: SafetyPostAnnotations, post: Post
    ) -> list[str]:
        tweet_id = annotation.tweetId
        violated_policies = annotation.violatedPolicies or []
        applied: list[str] = []

        def _get_author_id() -> int:
            user = post.user
            if not user or user.id is None:
                raise ValueError("post's userId is required for PTOS actioning")
            return int(user.id)

        async def _is_test_user() -> bool:
            author = _get_author_id()
            return await cls._strato_is_test_user.fetch(author)

        if await _is_test_user():
            await cls._strato_apply_label_from_ptos.apply(
                tweet_id, SafetyLabelType.PtosReviewed
            )
            logger.info(
                f"Skipping apply labels for post {tweet_id} except ptos reviewed since user is test user"
            )
            return [SafetyLabelType.PtosReviewed.value]

        author_id = _get_author_id()
        is_high_page_rank, is_grey_badge = await asyncio.gather(
            UserStratoLoader.is_high_page_rank_v2_user(author_id),
            UserStratoLoader.is_grey_badge_user(author_id),
        )
        is_high_page_rank_user = is_high_page_rank or is_grey_badge
        has_media = PostRenderer.has_media(post)

        for vp in violated_policies:
            category = vp.category
            policy = vp.safetyPolicy
            if not policy or not policy.policyType:
                continue

            policy_type = policy.policyType

            if category == SafetyPolicyCategory.ViolentMedia:
                if policy_type != SafetyPolicyType.NoViolation:
                    if has_media:
                        await cls._strato_apply_label_from_ptos.apply(
                            tweet_id, SafetyLabelType.GoreAndViolenceHighPrecision
                        )
                        applied.append(
                            SafetyLabelType.GoreAndViolenceHighPrecision.value
                        )

            if category == SafetyPolicyCategory.AdultContent:
                if policy_type == SafetyPolicyType.AdultContentSexualHard:
                    await cls._strato_apply_label_from_ptos.apply(
                        tweet_id, SafetyLabelType.NsfwHighRecall
                    )
                    applied.append(SafetyLabelType.NsfwHighRecall.value)

                    if has_media:
                        await cls._strato_apply_label_from_ptos.apply(
                            tweet_id, SafetyLabelType.NsfwHighPrecision
                        )
                        applied.append(SafetyLabelType.NsfwHighPrecision.value)

                elif policy_type == SafetyPolicyType.AdultContentSexualSoft:
                    await cls._strato_apply_label_from_ptos.apply(
                        tweet_id, SafetyLabelType.SoftNsfw
                    )
                    applied.append(SafetyLabelType.SoftNsfw.value)

            if category == SafetyPolicyCategory.Spam:
                should_label_for_spam = False
                if policy_type in (
                    SafetyPolicyType.SpamEngagementBaiting,
                    SafetyPolicyType.SpamEngagementFarming,
                ):
                    should_label_for_spam = True
                elif policy_type != SafetyPolicyType.NoViolation:
                    should_label_for_spam = not is_high_page_rank_user

                if should_label_for_spam:
                    await cls._strato_apply_label_from_ptos.apply(
                        tweet_id, SafetyLabelType.SpamHighRecall
                    )
                    applied.append(SafetyLabelType.SpamHighRecall.value)

            if category == SafetyPolicyCategory.IllegalAndRegulatedBehaviors:
                if policy_type != SafetyPolicyType.NoViolation:
                    if not is_high_page_rank_user:
                        await cls._strato_apply_label_from_ptos.apply(
                            tweet_id, SafetyLabelType.SpamHighRecall
                        )
                        applied.append(SafetyLabelType.SpamHighRecall.value)

            if category == SafetyPolicyCategory.HateOrAbuse:
                should_label_for_hate_or_abuse_insults_with_target = (
                    policy_type == SafetyPolicyType.HateOrAbuseInsultsWithTarget
                    and post.language == "en"
                    and not is_high_page_rank_user
                )
                if should_label_for_hate_or_abuse_insults_with_target:
                    await cls._strato_apply_label_from_ptos.apply(
                        tweet_id, SafetyLabelType.FosnrAbuseInsults
                    )
                    applied.append(SafetyLabelType.FosnrAbuseInsults.value)

            if (
                category == SafetyPolicyCategory.SuicideOrSelfHarm
                and policy_type
                in (
                    SafetyPolicyType.SuicideOrSelfHarmEncouraging,
                    SafetyPolicyType.SuicideOrSelfHarmDepiction,
                )
                and not is_high_page_rank_user
            ):
                await cls._strato_bounce_post_for_self_harm_encouragement.execute(
                    tweet_id, author_id
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.bounce.self_harm_encouragement.count"
                ).add(1)

            if (
                category == SafetyPolicyCategory.SuicideOrSelfHarm
                and policy_type == SafetyPolicyType.SuicideOrSelfHarmExpressingDesire
            ):
                await cls._strato_send_support_messages_for_self_harm.execute(
                    tweet_id, author_id
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.support_messages.self_harm.count"
                ).add(1)

            if (
                category == SafetyPolicyCategory.ChildSafety
                and policy_type == SafetyPolicyType.ChildSafetySexualExploitation
                and not is_high_page_rank_user
            ):
                await cls._strato_suspend_user_for_child_sexual_exploitation.execute(
                    tweet_id, author_id
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.suspend.child_sexual_exploitation.count"
                ).add(1)

        await cls._strato_apply_label_from_ptos.apply(
            tweet_id, SafetyLabelType.PtosReviewed
        )
        applied.append(SafetyLabelType.PtosReviewed.value)

        return applied

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.write_safety_post_annotations_result_sink.count").add(1)

        post = ctx.payload.post
        if not post:
            return

        safety_annotations = ctx.state(SafetyPtosState).annotations
        if not safety_annotations:
            return

        post_id = int(post.id)

        existing_result = await cls._strato_direct_mh.fetch(post_id)
        if existing_result:
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.existing_found.count"
            ).add(1)
        else:
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.existing_not_found.count"
            ).add(1)

        if safety_annotations.violatedPolicies:
            safety_annotations.violatedPolicies.sort(
                key=lambda x: x.score or 0, reverse=True
            )

        task_type_suffix = ctx.payload.task_type if ctx.payload.task_type else "unknown"
        identifier = f"{task_type_suffix}"
        timestamp_ms = int(time.time() * 1000)
        found_metadata = cls._build_found_metadata(post)

        new_annotation = SafetyPostAnnotations(
            tweetId=post_id,
            violatedPolicies=[
                SafetyPtosViolatedPolicy(**p.model_dump())
                for p in (safety_annotations.violatedPolicies or [])
            ],
            foundMetadata=found_metadata,
            identifier=identifier,
            timestamp=timestamp_ms,
        )

        annotations_list = (
            list(existing_result.safetyPostAnnotations)
            if existing_result and existing_result.safetyPostAnnotations
            else []
        )
        annotations_list.append(new_annotation)

        new_bool_metadata = cls._compute_bool_metadata_from_violations(
            safety_annotations
        )
        existing_bool_metadata = (
            existing_result.safetyBoolMetadata if existing_result else None
        )
        if post.get_fav_count() > 100 and existing_bool_metadata and new_bool_metadata:
            if not existing_bool_metadata.isNsfw and new_bool_metadata.isNsfw:
                logger.info(
                    f"grokPtosActionWithLabels latently found isNsfw for post {post_id}"
                )
        merged_bool_metadata = cls._merge_bool_metadata(
            existing_bool_metadata, new_bool_metadata
        )

        applied_labels = await cls._apply_labels_for_annotation(
            new_annotation, post=post
        )
        if len(applied_labels) > 0:
            logger.info(
                f"applyLabelFromPtos applied labels {applied_labels} for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.apply_label_from_ptos.count"
            ).add(1)
            for label in applied_labels:
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.apply_label_from_ptos.applied_label.count"
                ).add(1, attributes={"label": label})
        else:
            logger.info(
                f"applyLabelFromPtos did not apply any labels for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.apply_label_from_ptos.empty.count"
            ).add(1)

        ptos_already_nsfw = new_bool_metadata.isNsfw
        safemodel = ctx.state(SafetyPtosState).safemodel_sex_nudity
        if safemodel.positive and not ptos_already_nsfw:
            safemodel_confidence_int = round(safemodel.confidence * 100)
            safemodel_annotation = SafetyPostAnnotations(
                tweetId=post_id,
                violatedPolicies=[
                    SafetyPtosViolatedPolicy(
                        category=SafetyPolicyCategory.AdultContent,
                        score=safemodel_confidence_int,
                        reason="safemodel sex-and-nudity classifier detected adult content",
                        safetyPolicy=SafetyPolicy(
                            policyType=SafetyPolicyType.AdultContentSexualHard,
                            confidenceScore=safemodel_confidence_int,
                            reason="safemodel sex-and-nudity classifier detected adult content",
                        ),
                    ),
                ],
                foundMetadata=found_metadata,
                identifier="safemodel-sex-nudity",
                timestamp=timestamp_ms,
            )
            annotations_list.append(safemodel_annotation)
            merged_bool_metadata = cls._merge_bool_metadata(
                merged_bool_metadata,
                SafetyBoolMetadata(
                    isGore=False, isNsfw=True, isSoftNsfw=False, isSpam=False
                ),
            )

            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.safemodel_enforced.count"
            ).add(1)
            safemodel_applied_labels = await cls._apply_labels_for_annotation(
                safemodel_annotation, post=post
            )
            if len(safemodel_applied_labels) > 0:
                logger.info(
                    f"safemodel enforce: applyLabelFromPtos applied labels {safemodel_applied_labels} for post {post_id}"
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.safemodel_action.count"
                ).add(1)
                for label in safemodel_applied_labels:
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.safemodel_action.applied_label.count"
                    ).add(1, attributes={"label": label})
            else:
                logger.info(
                    f"safemodel enforce: applyLabelFromPtos applied no labels for post {post_id}"
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.safemodel_action.empty.count"
                ).add(1)

        final_result = SafetyPostAnnotationsResult(
            tweetId=post_id,
            safetyPostAnnotations=annotations_list,
            safetyBoolMetadata=merged_bool_metadata,
        )

        delete_labels_result = await cls._strato_grok_ptos_delete_labels.execute(
            final_result
        )
        if delete_labels_result:
            logger.info(
                f"grokPtosDeleteLabels returned '{delete_labels_result}' for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_delete_labels.count"
            ).add(1)
        else:
            logger.info(
                f"grokPtosDeleteLabels returned no result for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_delete_labels.empty.count"
            ).add(1)

        await cls._strato_mh.put(post_id, final_result)
        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.mh.success.count"
        ).add(1)

        await cls._strato_kafka.insert(post_id, final_result)
        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.kafka.success.count"
        ).add(1)

        is_very_high_fav = post.get_fav_count() >= HIGH_FAV_THRESHOLD
        logger.info(
            f"Published safety annotations for post {post_id} and is_very_high_fav {is_very_high_fav} with task identifier={identifier}: isGore={merged_bool_metadata.isGore}, isNsfw={merged_bool_metadata.isNsfw}, isSoftNsfw={merged_bool_metadata.isSoftNsfw}, isSpam={merged_bool_metadata.isSpam}"
        )
        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.success.count"
        ).add(1)
