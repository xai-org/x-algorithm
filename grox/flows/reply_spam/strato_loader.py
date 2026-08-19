import logging

from strato_http.queries.data_types import (
    ReplyRankingScore,
    ReplyRankingScoreKafka,
)
from strato_http.queries.reply_ranking_score import StratoReplyRankingScore
from strato_http.queries.reply_ranking_score_kafka_v2 import (
    StratoReplyRankingScoreV2Kafka,
)

logger = logging.getLogger(__name__)


class ReplyRankingScoreStratoLoader:
    strato = StratoReplyRankingScore()
    reply_ranking_v2_kafka_strato = StratoReplyRankingScoreV2Kafka()

    @classmethod
    async def save_reply_ranking_score(
        cls, post_id: str, reply_ranking_score: ReplyRankingScore
    ):
        await cls.strato.put(int(post_id), reply_ranking_score)

    @classmethod
    async def save_reply_ranking_kafka_v2(
        cls, post_id: str, reply_ranking_score_kafka: ReplyRankingScoreKafka
    ):
        await cls.reply_ranking_v2_kafka_strato.insert(
            int(post_id), reply_ranking_score_kafka
        )
