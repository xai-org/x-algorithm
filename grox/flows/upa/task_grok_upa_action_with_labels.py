import logging

from monitor.metrics import Metrics
from strato_http.queries.grok_upa_action_with_labels import (
    StratoGrokUpaActionWithLabels,
)

from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskForNonProd
from grox.core.tasks.task import Task
from grox.flows.upa.state_initial_banger import InitialBangerState
from grox.flows.upa.state_post_safety import PostSafetyState

logger = logging.getLogger(__name__)


class TaskGrokUpaActionWithLabels(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    _strato_grok_upa_action_with_labels = StratoGrokUpaActionWithLabels()

    @classmethod
    def _get_tweet_bool_metadata(cls, ctx: TaskContext):
        banger = ctx.state(InitialBangerState).result
        if banger and banger.tweet_bool_metadata:
            return banger.tweet_bool_metadata
        safety = ctx.state(PostSafetyState).result
        return safety.tweet_bool_metadata if safety else None

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.grok_upa_action_with_labels.count").add(1)

        post = ctx.payload.post
        if not post:
            return

        metadata = cls._get_tweet_bool_metadata(ctx)
        if not metadata:
            return

        tweet_id = int(post.id)
        tweet_bool_metadata = metadata.model_dump()

        action_result = await cls._strato_grok_upa_action_with_labels.execute(
            tweet_id, tweet_bool_metadata
        )
        if action_result and len(action_result.applied_labels) > 0:
            logger.info(
                f"grokUpaActionWithLabels applied labels: debugString='{action_result.debug_string}', appliedLabels={action_result.applied_labels} for post {tweet_id}"
            )
            Metrics.counter("task.grok_upa_action_with_labels.applied.count").add(1)
            for label in action_result.applied_labels:
                Metrics.counter(
                    "task.grok_upa_action_with_labels.applied_label.count"
                ).add(1, attributes={"label": label})
        elif action_result:
            logger.info(
                f"grokUpaActionWithLabels no labels applied: debugString='{action_result.debug_string}' for post {tweet_id}"
            )
            Metrics.counter("task.grok_upa_action_with_labels.empty.count").add(1)
        else:
            logger.info(f"grokUpaActionWithLabels failed for post {tweet_id}")
            Metrics.counter("task.grok_upa_action_with_labels.failed.count").add(1)
