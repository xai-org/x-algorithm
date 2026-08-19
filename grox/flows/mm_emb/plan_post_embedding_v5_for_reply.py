from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_asr import TaskASRTranscription
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.mm_emb.task_filter import TaskPostEmbeddingForReplyFilter
from grox.flows.mm_emb.task_multimodal_post_embedding import (
    TaskMultimodalPostEmbeddingV5,
)
from grox.flows.mm_emb.task_rate_limit import TaskRateLimitEmbeddingV5ForReply
from grox.flows.mm_emb.task_write_mm_embedding_sink import TaskWriteMMEmbeddingSinkV5


@register
class PlanPostEmbeddingV5ForReply(Plan):
    KEY = "mm_emb_v5_for_reply"

    TASKS = {
        "task_post_embedding_rate_limit_v5_for_reply": TaskRateLimitEmbeddingV5ForReply,
        "task_post_embedding_filter_for_reply": TaskPostEmbeddingForReplyFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_asr_transcription": TaskASRTranscription,
        "task_multimodal_post_embedding_v5": TaskMultimodalPostEmbeddingV5,
        "task_write_post_embedding_sink_v5": TaskWriteMMEmbeddingSinkV5,
    }

    TASK_DEPENDENCIES = {
        "task_post_embedding_rate_limit_v5_for_reply": set(),
        "task_post_embedding_filter_for_reply": {
            "task_post_embedding_rate_limit_v5_for_reply"
        },
        "task_media_hydration": {"task_post_embedding_filter_for_reply"},
        "task_asr_transcription": {"task_media_hydration"},
        "task_multimodal_post_embedding_v5": {"task_asr_transcription"},
        "task_write_post_embedding_sink_v5": {"task_multimodal_post_embedding_v5"},
    }
