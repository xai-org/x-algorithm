from grox.core.data_loaders.kafka_loader import KafkaPostLoader
from grox.core.generators.stream_generator import StreamTaskGenerator
from grox.core.registry import register
from grox.flows.mm_emb.constants import (
    POST_EMBEDDING_V5_FOR_REPLY_STREAM,
    POST_EMBEDDING_V5_RECOVERY_STREAM,
    POST_EMBEDDING_V5_STREAM,
    POST_EMBEDDING_V8_2_FOR_REPLY_STREAM,
    POST_EMBEDDING_V8_2_RECOVERY_STREAM,
    POST_EMBEDDING_V8_2_STREAM,
    TOPIC_MIN_TRACTION_MULTI_MODAL,
    TOPIC_REQUESTS_WITH_SUMMARY,
    TOPIC_V5_RECOVERY,
    TOPIC_V8_RECOVERY,
)
from grox.flows.mm_emb.plan_post_embedding_v5 import PlanPostEmbeddingV5
from grox.flows.mm_emb.plan_post_embedding_v5_for_reply import (
    PlanPostEmbeddingV5ForReply,
)
from grox.flows.mm_emb.plan_post_embedding_v82 import PlanPostEmbeddingV82
from grox.flows.mm_emb.plan_post_embedding_v82_for_reply import (
    PlanPostEmbeddingV82ForReply,
)


@register
class PostEmbeddingV5StreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V5_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV5.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_REQUESTS_WITH_SUMMARY)


@register
class PostEmbeddingV5RecoveryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V5_RECOVERY_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV5.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_V5_RECOVERY)


@register
class PostEmbeddingV5ForReplyStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V5_FOR_REPLY_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV5ForReply.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MIN_TRACTION_MULTI_MODAL)


@register
class PostEmbeddingV82StreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V8_2_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV82.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_REQUESTS_WITH_SUMMARY)


@register
class PostEmbeddingV82RecoveryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V8_2_RECOVERY_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV82.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_V8_RECOVERY)


@register
class PostEmbeddingV82ForReplyStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_EMBEDDING_V8_2_FOR_REPLY_STREAM
    PLANS_TO_INJECT = {PlanPostEmbeddingV82ForReply.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MIN_TRACTION_MULTI_MODAL)
