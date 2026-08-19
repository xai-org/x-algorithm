from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_asr import TaskASRTranscription
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.mm_emb.task_multimodal_post_embedding import (
    TaskMultimodalPostEmbeddingV5,
)
from grox.flows.mm_emb.task_rate_limit import TaskRateLimitEmbeddingV5
from grox.flows.mm_emb.task_write_mm_embedding_sink import (
    TaskWriteMMEmbeddingSinkV5SkipKafkaForReplies,
    TaskWriteMMEmbeddingV5ToAllTopic,
)


@register
class PlanPostEmbeddingV5(Plan):
    KEY = "mm_emb_v5"

    TASKS = {
        "task_post_embedding_rate_limit": TaskRateLimitEmbeddingV5,
        "task_media_hydration": TaskMediaHydration,
        "task_asr_transcription": TaskASRTranscription,
        "task_multimodal_post_embedding_v5": TaskMultimodalPostEmbeddingV5,
        "task_write_post_embedding_sink_v5": TaskWriteMMEmbeddingSinkV5SkipKafkaForReplies,
        "task_write_post_embedding_v5_to_all_topic": TaskWriteMMEmbeddingV5ToAllTopic,
    }

    TASK_DEPENDENCIES = {
        "task_post_embedding_rate_limit": set(),
        "task_media_hydration": {"task_post_embedding_rate_limit"},
        "task_asr_transcription": {"task_media_hydration"},
        "task_multimodal_post_embedding_v5": {"task_asr_transcription"},
        "task_write_post_embedding_sink_v5": {"task_multimodal_post_embedding_v5"},
        "task_write_post_embedding_v5_to_all_topic": {
            "task_multimodal_post_embedding_v5"
        },
    }
