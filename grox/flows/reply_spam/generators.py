from grox.core.data_loaders.kafka_loader import KafkaPostLoader
from grox.core.generators.stream_generator import StreamTaskGenerator
from grox.core.registry import register
from grox.flows.reply_spam.constants import (
    REPLY_RANKING,
    REPLY_RANKING_RECOVERY,
    TOPIC_REPLY_RANKING_RECOVERY,
    TOPIC_UNIFIED_POSTS_V3,
)
from grox.flows.reply_spam.plan_coordinated_spam import PlanCoordinatedSpam
from grox.flows.reply_spam.plan_reply_ranking import PlanReplyRanking
from grox.flows.reply_spam.plan_spam_comment import PlanSpamComment


@register
class ReplyRankingTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = REPLY_RANKING
    PLANS_TO_INJECT = {
        PlanReplyRanking.KEY,
        PlanSpamComment.KEY,
        PlanCoordinatedSpam.KEY,
    }

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_UNIFIED_POSTS_V3)


@register
class ReplyRankingRecoveryTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = REPLY_RANKING_RECOVERY
    PLANS_TO_INJECT = {PlanReplyRanking.KEY}

    def _get_loader(self):
        return KafkaPostLoader(TOPIC_REPLY_RANKING_RECOVERY)
