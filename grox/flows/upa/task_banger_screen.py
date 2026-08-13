import logging
import time
from typing import override

from monitor.metrics import Metrics
from strato_http.queries.grok_topics import StratoGrokTopics

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.upa.classifier_banger_initial_screen_gemma import (
    BangerInitialScreenGemmaClassifier,
)
from grox.flows.upa.state_initial_banger import InitialBangerState

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600


class TaskBangerScreen(TaskWithPost):
    gemma_classifier = BangerInitialScreenGemmaClassifier()
    _cached_topics = None
    _cache_timestamp = None

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.banger_initial_screen.total.count").add(1)

        if cls._cached_topics is None or (
            cls._cache_timestamp is not None
            and time.time() - cls._cache_timestamp > CACHE_TTL_SECONDS
        ):
            logger.info("Fetching grok topics for cache")
            Metrics.counter("task.banger_initial_screen.grok_cache.new_load.count").add(
                1
            )
            query = StratoGrokTopics()
            fetched_topics = await query.fetch()
            if fetched_topics:
                cls._cached_topics = fetched_topics
                cls._cache_timestamp = time.time()
                logger.info(f"Cached {len(cls._cached_topics)} categories with topics")
                Metrics.counter(
                    "task.banger_initial_screen.grok_cache.new_load.success.count"
                ).add(1)
            else:
                logger.warning("Failed to fetch grok topics")
                Metrics.counter(
                    "task.banger_initial_screen.grok_cache.new_load.failure.count"
                ).add(1)
        else:
            Metrics.counter(
                "task.banger_initial_screen.grok_cache.cache_hit.count"
            ).add(1)

        if cls._cached_topics and len(cls._cached_topics) > 0:
            Metrics.counter("task.banger_initial_screen.with_cached_topics.count").add(
                1
            )
        else:
            Metrics.counter(
                "task.banger_initial_screen.without_cached_topics.count"
            ).add(1)

        Metrics.counter("task.banger_initial_screen.classifier.count").add(
            1, attributes={"classifier": "gemma"}
        )
        res = await cls.gemma_classifier.classify(post, topics=cls._cached_topics)
        state = ctx.state(InitialBangerState)
        state.result = res
        state.available_topics = cls._cached_topics
