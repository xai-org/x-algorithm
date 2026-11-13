from grox.core.data_loaders.kafka_loader import KafkaPostLoader
from grox.core.generators.stream_generator import StreamTaskGenerator
from grox.core.registry import register
from grox.flows.ptos.constants import (
    POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS,
    SAFETY_PTOS_ADULT_CONTENT_LEADING_FRAMES,
    SAFETY_PTOS_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION,
    SAFETY_PTOS_KEY_FRAMES,
    SAFETY_PTOS_MANY_MEDIA,
    SAFETY_PTOS_SPECIAL_VIDEO,
    TOPIC_ADULT_CONTENT_LEADING_FRAMES,
    TOPIC_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION,
    TOPIC_KEY_FRAMES,
    TOPIC_MANY_MEDIA,
    TOPIC_SPECIAL_VIDEO,
    POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS,
    SAFETY_PTOS_BACKFILL,
    SAFETY_PTOS_DELUXE,
    SAFETY_PTOS_LIVE_CLUSTER_ANCHORS,
    SAFETY_PTOS_REALTIME_WITH_SIGNALS,
    SAFETY_PTOS_RECOVERY,
    TOPIC_BACKFILL,
    TOPIC_DELAYED_REPLICATION,
    TOPIC_DELUXE,
    TOPIC_LIVE_CLUSTER_ANCHORS,
    TOPIC_MIN_IMPRESSION,
    TOPIC_MIN_TRACTION,
    TOPIC_RECOVERY,
)
from grox.flows.ptos.kafka_loader import KafkaLiveClusterAnchorLoader
from grox.flows.ptos.plan_safety_ptos import PlanSafetyPtos
from grox.flows.ptos.plan_safety_ptos_adult_content_leading_frames import (
    PlanSafetyPtosAdultContentLeadingFrames,
)
from grox.flows.ptos.plan_safety_ptos_media_injected_adult_infrared_video_spam_detection import (
    PlanSafetyPtosMediaInjectedAdultInfraredVideoSpamDetection,
)
from grox.flows.ptos.plan_safety_ptos_key_frames import PlanSafetyPtosKeyFrames
from grox.flows.ptos.plan_safety_ptos_many_media import PlanSafetyPtosManyMedia
from grox.flows.ptos.plan_safety_ptos_special_video import PlanSafetyPtosSpecialVideo
from grox.flows.ptos.plan_safety_ptos_live_cluster_anchors import (
    PlanSafetyPtosLiveClusterAnchors,
)
from grox.flows.ptos.plan_safety_ptos_realtime_with_signals import (
    PlanSafetyPtosWithRealTimeSignals,
)


@register
class MinTractionPostStreamForGroxPtosTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS
    PLANS_TO_INJECT = {PlanSafetyPtos.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MIN_TRACTION)


@register
class SafetyPtosRecoveryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_RECOVERY
    PLANS_TO_INJECT = {PlanSafetyPtos.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_RECOVERY)


@register
class SafetyPtosDeluxeStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_DELUXE
    PLANS_TO_INJECT = {PlanSafetyPtos.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_DELUXE)


@register
class SafetyPtosBackfillStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_BACKFILL
    PLANS_TO_INJECT = {PlanSafetyPtos.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_BACKFILL)


@register
class SafetyPtosRealtimeWithSignalsStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_REALTIME_WITH_SIGNALS
    PLANS_TO_INJECT = {PlanSafetyPtosWithRealTimeSignals.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_DELAYED_REPLICATION)


@register
class MinImpressionPostStreamForGroxPtosTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS
    PLANS_TO_INJECT = {PlanSafetyPtos.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MIN_IMPRESSION)


@register
class SafetyPtosLiveClusterAnchorsStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_LIVE_CLUSTER_ANCHORS
    PLANS_TO_INJECT = {PlanSafetyPtosLiveClusterAnchors.KEY}

    def _get_loader(self):
        return KafkaLiveClusterAnchorLoader(TOPIC_LIVE_CLUSTER_ANCHORS)


@register
class SafetyPtosSpecialVideoStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_SPECIAL_VIDEO
    PLANS_TO_INJECT = {PlanSafetyPtosSpecialVideo.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_SPECIAL_VIDEO)


@register
class SafetyPtosAdultContentLeadingFramesStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_ADULT_CONTENT_LEADING_FRAMES
    PLANS_TO_INJECT = {PlanSafetyPtosAdultContentLeadingFrames.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_ADULT_CONTENT_LEADING_FRAMES)


@register
class SafetyPtosKeyFramesStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_KEY_FRAMES
    PLANS_TO_INJECT = {PlanSafetyPtosKeyFrames.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_KEY_FRAMES)


@register
class SafetyPtosManyMediaStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_MANY_MEDIA
    PLANS_TO_INJECT = {PlanSafetyPtosManyMedia.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MANY_MEDIA)


@register
class SafetyPtosMediaInjectedAdultInfraredVideoSpamDetectionStreamTaskGenerator(
    StreamTaskGenerator
):
    TASK_GENERATOR_TYPE = SAFETY_PTOS_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION
    PLANS_TO_INJECT = {PlanSafetyPtosMediaInjectedAdultInfraredVideoSpamDetection.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION)
