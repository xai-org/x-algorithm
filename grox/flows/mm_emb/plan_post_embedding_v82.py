from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_asr import TaskASRTranscription
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.mm_emb.task_multimodal_post_embedding import (
    TaskMultimodalPostEmbeddingV82,
)
from grox.flows.mm_emb.task_rate_limit import TaskRateLimitEmbeddingV82
from grox.flows.mm_emb.task_write_mm_embedding_sink import (
    TaskWriteMMEmbeddingSinkV82SkipKafkaForReplies,
)


@register
class PlanPostEmbeddingV82(Plan):
    KEY = "mm_emb_v8_2"

    TASKS = {
        "task_post_embedding_rate_limit_v82": TaskRateLimitEmbeddingV82,
        "task_media_hydration": TaskMediaHydration,
        "task_asr_transcription": TaskASRTranscription,
        "task_multimodal_post_embedding_v82": TaskMultimodalPostEmbeddingV82,
        "task_write_post_embedding_sink_v82": TaskWriteMMEmbeddingSinkV82SkipKafkaForReplies,
    }

    TASK_DEPENDENCIES = {
        "task_post_embedding_rate_limit_v82": set(),
        "task_media_hydration": {"task_post_embedding_rate_limit_v82"},
        "task_asr_transcription": {"task_media_hydration"},
        "task_multimodal_post_embedding_v82": {"task_asr_transcription"},
        "task_write_post_embedding_sink_v82": {"task_multimodal_post_embedding_v82"},
    }
