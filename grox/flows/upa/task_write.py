import logging

from monitor.metrics import Metrics
from strato_http.queries.data_types import (
    EntityWithMetadata,
    FoundMetadata,
    QualifiedId,
    UnifiedPostAnnotations,
)
from strato_http.queries.unified_post_annotations import (
    StratoUnifiedPostAnnotations,
    StratoUpsertTweetBoolMetadataToUnifiedPostAnnotations,
)

from grox.core.data_loaders.data_types import (
    Image,
    Video,
)
from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskForNonProd
from grox.core.tasks.task import Task
from grox.flows.upa.models import ContentCategoryScore
from grox.flows.upa.state_initial_banger import InitialBangerState
from grox.flows.upa.state_post_safety import PostSafetyState

logger = logging.getLogger(__name__)


class TaskPublishUnifiedPostAnnotationsManhattan(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.publish_unified_post_annotations.count").add(1)
        banger_state = ctx.state(InitialBangerState)
        grok_response = banger_state.result
        if not grok_response:
            logger.info("No unified post annotations to publish")
            return

        post = ctx.payload.post
        if not post:
            return

        if grok_response.tweet_bool_metadata:
            if grok_response.tweet_bool_metadata.isNsfw:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_nsfw_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isGore:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_gore_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isViolent:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_violent_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isSpam:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_spam_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isSoftNsfw:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_soft_nsfw_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isAdult:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_adult_true.count"
                ).add(1)

        if grok_response.tags and len(grok_response.tags) > 0:
            Metrics.counter(
                "task.publish_unified_post_annotations.tags_non_empty.count"
            ).add(1)

        if grok_response.is_image_editable_by_grok:
            Metrics.counter(
                "task.publish_unified_post_annotations.is_image_editable_by_grok_true.count"
            ).add(1)

        if post.media:
            if any(isinstance(m, Video) for m in post.media):
                Metrics.counter(
                    "task.publish_unified_post_annotations.has_video_true.count"
                ).add(1)
            if any(isinstance(m, Image) for m in post.media):
                Metrics.counter(
                    "task.publish_unified_post_annotations.has_image_true.count"
                ).add(1)

        resolved_grok_topics = []
        if grok_response.taxonomy_categories and banger_state.available_topics:
            id_to_name = {}
            name_to_category_id = {}
            for category in banger_state.available_topics:
                id_to_name[category.categoryEntityId] = category.categoryName
                name_to_category_id[category.categoryName] = category.categoryEntityId
                for sub in category.subtopics:
                    id_to_name[sub.topicEntityId] = sub.topicName
                    name_to_category_id[sub.topicName] = category.categoryEntityId

            topic_id_to_best_score = {}
            for grok_topic in grok_response.taxonomy_categories:
                topic_id = grok_topic.id
                if topic_id in id_to_name:
                    topic_name = id_to_name[topic_id]
                    category_id = name_to_category_id[topic_name]

                    resolved_grok_topic = ContentCategoryScore(
                        id=topic_id,
                        name=topic_name,
                        score=grok_topic.score,
                        category_id=category_id,
                    )
                    logger.info(
                        f"Validated grok_topic: ID {topic_id} -> '{topic_name}' (category_id: {category_id})"
                    )
                else:
                    logger.warning(
                        f"Invalid topic ID from Grok: {topic_id} not found in available topics"
                    )
                    Metrics.counter(
                        "task.publish_unified_post_annotations.invalid_grok_topic.count"
                    ).add(1)
                    continue

                if (
                    topic_id not in topic_id_to_best_score
                    or grok_topic.score > topic_id_to_best_score[topic_id].score
                ):
                    topic_id_to_best_score[topic_id] = resolved_grok_topic

            resolved_grok_topics = list(topic_id_to_best_score.values())
        elif grok_response.taxonomy_categories:
            logger.warning("No available topics to validate grok_topics")
            resolved_grok_topics = []

        for topic in resolved_grok_topics:
            sanitized_topic_name_for_metric = (
                topic.name.lower().replace(" ", "_").replace("&", "and")
            )
            Metrics.counter(
                f"task.publish_unified_post_annotations.topic_{sanitized_topic_name_for_metric}.count"
            ).add(1)

        entities = []
        if resolved_grok_topics and len(resolved_grok_topics) > 0:
            Metrics.counter(
                "task.publish_unified_post_annotations.with_grok_topics.count"
            ).add(1)
            entities = [
                EntityWithMetadata(
                    qualifiedId=QualifiedId(domainId=236, entityId=str(grok_topic.id)),
                    score=grok_topic.score,
                    categoryId=QualifiedId(
                        domainId=236, entityId=str(grok_topic.category_id)
                    )
                    if grok_topic.category_id
                    else None,
                )
                for grok_topic in resolved_grok_topics
            ]

        annotations = UnifiedPostAnnotations(
            tweetId=post.id,
            entities=entities,
            tags=[{"tag": tag, "score": 0.0} for tag in (grok_response.tags or [])],
            tweetBoolMetadata=grok_response.tweet_bool_metadata.model_dump()
            if grok_response.tweet_bool_metadata
            else None,
            description=grok_response.summary,
            isImageEditableByGrok=grok_response.is_image_editable_by_grok,
            slopScore=grok_response.slop_score,
            originalOcrText="",
            hasVideo=post.media and any(isinstance(m, Video) for m in post.media),
            hasImage=post.media and any(isinstance(m, Image) for m in post.media),
            qualityScore=1.0,
            hasMinorScore=grok_response.has_minor_score,
            hasCard=post.card is not None,
            foundMetadata=FoundMetadata(
                imageCount=sum(1 for m in post.media if isinstance(m, Image))
                if post.media
                else 0,
                videoCount=sum(1 for m in post.media if isinstance(m, Video))
                if post.media
                else 0,
                cardCount=1 if post.card else 0,
                cardV2Count=len(post.cardsV2) if post.cardsV2 else 0,
            ),
        )

        await StratoUnifiedPostAnnotations().put(int(post.id), annotations)
        Metrics.counter("task.publish_unified_post_annotations.success.count").add(1)


class TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter(
            "task.upsert_tweet_bool_metadata_to_unified_post_annotations.count"
        ).add(1)
        grok_response = ctx.state(PostSafetyState).result
        if not grok_response or not grok_response.tweet_bool_metadata:
            logger.info("No unified post annotations to publish")
            return

        post = ctx.payload.post
        if not post:
            return

        if grok_response.tweet_bool_metadata.isNsfw:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_nsfw_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isGore:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_gore_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isViolent:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_violent_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isSpam:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_spam_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isSoftNsfw:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_soft_nsfw_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isAdult:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_adult_true.count"
            ).add(1)

        await StratoUpsertTweetBoolMetadataToUnifiedPostAnnotations().put(
            int(post.id), grok_response.tweet_bool_metadata.model_dump()
        )
        Metrics.counter(
            "task.upsert_tweet_bool_metadata_to_unified_post_annotations.success.count"
        ).add(1)
