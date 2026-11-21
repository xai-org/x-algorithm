from grox.core.data_loaders.kafka_loader import KafkaPostLoader
from grox.core.generators.stream_generator import StreamTaskGenerator
from grox.flows.upa.plan_initial_banger import PlanInitialBanger
from grox.flows.upa.plan_post_safety import PlanPostSafety
from grox.core.registry import register
from grox.flows.upa.constants import (
    POST_MIN_TRACTION_STREAM_FOR_GROX,
    POST_SAFETY_STREAM,
    POST_STREAM_RECOVERY,
    TOPIC_MIN_TRACTION,
    TOPIC_POPULAR,
    TOPIC_RECOVERY,
)


@register
class MinTractionPostStreamForGroxTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_MIN_TRACTION_STREAM_FOR_GROX
    PLANS_TO_INJECT = {PlanInitialBanger.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MIN_TRACTION)


@register
class PostStreamRecoveryTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_STREAM_RECOVERY
    PLANS_TO_INJECT = {PlanInitialBanger.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_RECOVERY)


@register
class PostSafetyStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_SAFETY_STREAM
    PLANS_TO_INJECT = {PlanPostSafety.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_POPULAR)
