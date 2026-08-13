from grox.core.data_loaders.kafka_loader import KafkaPostLoader
from grox.core.generators.stream_generator import StreamTaskGenerator
from grox.core.registry import register
from grox.flows.reply_spam.constants import (
    POST_STREAM,
    REPLY_RANKING_RECOVERY,
    TOPIC_REPLY_RANKING_RECOVERY,
    TOPIC_UNIFIED_POSTS,
)
from grox.flows.reply_spam.plan_coordinated_spam import PlanCoordinatedSpam
from grox.flows.reply_spam.plan_reply_ranking import PlanReplyRanking
from grox.flows.reply_spam.plan_spam_comment import PlanSpamComment


@register
class PostStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = POST_STREAM
    PLANS_TO_INJECT = {
        PlanSpamComment.KEY,
        PlanReplyRanking.KEY,
        PlanCoordinatedSpam.KEY,
    }

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_UNIFIED_POSTS)


@register
class ReplyRankingRecoveryTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = REPLY_RANKING_RECOVERY
    PLANS_TO_INJECT = {PlanReplyRanking.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_REPLY_RANKING_RECOVERY)
