import logging
import time
from functools import cache

from kafka_cli.producer import KafkaProducer
from monitor.metrics import Metrics
from strato_http.queries.data_types import (
    SafetyPostAnnotations as StratoSafetyPostAnnotations,
)
from strato_http.queries.data_types import (
    SafetyPtosViolatedPolicy as StratoSafetyPtosViolatedPolicy,
)
from strato_http.queries.live_cluster_anchor_verdict_mh import (
    LiveClusterAnchorMhVerdict,
    LiveClusterAnchorVerdictResult,
    StratoLiveClusterAnchorVerdictMh,
)

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import TaskWithPost
from grox.flows.ptos import constants
from grox.flows.ptos.disable_rules import DisableTaskForNonPtosProd
from grox.flows.ptos.state import (
    LiveClusterAnchorInput,
    LiveClusterAnchorKafkaVerdict,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPostAnnotations,
    SafetyPtosState,
    SafetyPtosViolatedPolicy,
)

logger = logging.getLogger(__name__)

_ENFORCE_CATEGORY = SafetyPolicyCategory.Spam


def _is_actionable_spam_policy_violation(vp: SafetyPtosViolatedPolicy) -> bool:
    if vp.category != _ENFORCE_CATEGORY:
        return False
    policy = vp.safetyPolicy
    return bool(
        policy
        and policy.policyType
        and policy.policyType != SafetyPolicyType.NoViolation
    )


def _has_enforce_decision(annotations: SafetyPostAnnotations | None) -> bool:
    if not annotations or not annotations.violatedPolicies:
        return False
    return any(
        _is_actionable_spam_policy_violation(vp) for vp in annotations.violatedPolicies
    )


def _to_strato_safety_post_annotations(
    anchor_post_id: int,
    annotations: SafetyPostAnnotations | None,
    timestamp_ms: int,
) -> StratoSafetyPostAnnotations | None:
    if annotations is None:
        return None
    violated = None
    if annotations.violatedPolicies is not None:
        violated = [
            StratoSafetyPtosViolatedPolicy(**p.model_dump())
            for p in annotations.violatedPolicies
        ]
    return StratoSafetyPostAnnotations(
        tweetId=anchor_post_id,
        violatedPolicies=violated,
        identifier="live-cluster-anchor",
        timestamp=timestamp_ms,
    )


def _build_live_cluster_anchor_verdict(
    anchor_post_id: int,
    cluster_id: int,
    annotations: SafetyPostAnnotations | None,
) -> LiveClusterAnchorMhVerdict:
    timestamp_ms = int(time.time() * 1000)
    return LiveClusterAnchorMhVerdict(
        anchorPostId=anchor_post_id,
        clusterId=cluster_id,
        enforceDecision=_has_enforce_decision(annotations),
        timestampMs=timestamp_ms,
        safetyPostAnnotations=_to_strato_safety_post_annotations(
            anchor_post_id, annotations, timestamp_ms
        ),
    )


def _to_kafka_payload(
    verdict: LiveClusterAnchorMhVerdict,
) -> LiveClusterAnchorKafkaVerdict:
    return LiveClusterAnchorKafkaVerdict(
        anchorPostId=verdict.anchorPostId,
        clusterId=verdict.clusterId,
        enforceDecision=verdict.enforceDecision,
        timestampMs=verdict.timestampMs,
    )


class TaskWriteLiveClusterAnchorVerdictSink(TaskWithPost):
    DISABLE_RULES = [DisableTaskForNonPtosProd]
    _strato_mh = StratoLiveClusterAnchorVerdictMh()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        anchor_input = cls._require_anchor_input(ctx, post)
        annotations = ctx.state(SafetyPtosState).annotations

        verdict = _build_live_cluster_anchor_verdict(
            anchor_post_id=anchor_input.anchorPostId,
            cluster_id=anchor_input.clusterId,
            annotations=annotations,
        )
        Metrics.counter("task.live_cluster_anchor_verdict.count").add(
            1,
            attributes={
                "enforce_decision": str(verdict.enforceDecision).lower(),
            },
        )

        anchor_post_id = int(anchor_input.anchorPostId)
        existing = await cls._strato_mh.fetch(anchor_post_id)
        if existing:
            Metrics.counter(
                "task.live_cluster_anchor_verdict.existing_found.count"
            ).add(1)
            result = existing
        else:
            Metrics.counter(
                "task.live_cluster_anchor_verdict.existing_not_found.count"
            ).add(1)
            result = LiveClusterAnchorVerdictResult(anchorPostId=anchor_post_id)

        result.liveClusterAnchorMhVerdicts.append(verdict)
        await cls._strato_mh.put(anchor_post_id, result)
        Metrics.counter("task.live_cluster_anchor_verdict.mh.success.count").add(1)

        kafka_payload = _to_kafka_payload(verdict).model_dump_json().encode("utf-8")
        await cls._get_producer().send(id=str(post.id), value=kafka_payload)
        Metrics.counter("task.live_cluster_anchor_verdict.emitted.count").add(
            1, attributes={"enforce_decision": str(verdict.enforceDecision).lower()}
        )

        logger.info(
            f"live_cluster_anchor stored+emitted post={post.id} cluster_id={anchor_input.clusterId} "
            f"enforce_decision={verdict.enforceDecision} mh_len={len(result.liveClusterAnchorMhVerdicts)}"
        )

    @classmethod
    def _require_anchor_input(
        cls, ctx: TaskContext, post: Post
    ) -> LiveClusterAnchorInput:
        anchor = ctx.payload.ext(LiveClusterAnchorInput)
        if not isinstance(anchor, LiveClusterAnchorInput):
            Metrics.counter("task.live_cluster_anchor_verdict.failed.count").add(
                1, attributes={"reason": "missing_anchor_input"}
            )
            msg = f"live_cluster_anchor missing required LiveClusterAnchorInput post={post.id}"
            logger.error(msg)
            raise ValueError(msg)

        if str(anchor.anchorPostId) != str(post.id):
            Metrics.counter("task.live_cluster_anchor_verdict.failed.count").add(
                1, attributes={"reason": "anchor_post_id_mismatch"}
            )
            msg = f"live_cluster_anchor anchor_post_id mismatch payload_post={post.id} anchor_post_id={anchor.anchorPostId} cluster_id={anchor.clusterId}"
            logger.error(msg)
            raise ValueError(msg)
        return anchor

    @classmethod
    @cache
    def _get_producer(cls) -> KafkaProducer:
        producer_config = grox_config.get_kafka_producer_topic(
            constants.TOPIC_LIVE_CLUSTER_ANCHOR_VERDICTS
        )
        return KafkaProducer(producer_config)
