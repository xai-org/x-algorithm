POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS = "post_min_traction_stream_for_grox_ptos"
POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS = "post_min_impression_stream_for_grox_ptos"
SAFETY_PTOS_RECOVERY = "safety_ptos_recovery"
SAFETY_PTOS_DELUXE = "safety_ptos_deluxe"
SAFETY_PTOS_BACKFILL = "safety_ptos_backfill"
SAFETY_PTOS_REALTIME_WITH_SIGNALS = "safety_ptos_realtime_with_signals"
SAFETY_PTOS_LIVE_CLUSTER_ANCHORS = "safety_ptos_live_cluster_anchors"
TOPIC_MIN_TRACTION = (
    "content-understanding-realtime-unified-posts-min-traction-for-grox-ptos"
)
TOPIC_MIN_IMPRESSION = (
    "content-understanding-realtime-unified-posts-min-impression-for-grox-ptos"
)
TOPIC_RECOVERY = "safety-ptos-recovery"
TOPIC_DELUXE = "safety-ptos-deluxe"
TOPIC_BACKFILL = "safety-ptos-backfill"
TOPIC_DELAYED_REPLICATION = (
    "content-understanding-realtime-unified-posts-delayed-replication"
)
TOPIC_LIVE_CLUSTER_ANCHORS = "safety-ptos-live-cluster-anchors"
TOPIC_LIVE_CLUSTER_ANCHOR_VERDICTS = "safety-ptos-live-cluster-anchor-verdicts"
GEMMA = "oai-gemma4-26b"
GEMMA_PTOS_REALTIME = "oai-gemma4-26b-ptos-realtime"

HIGH_FAV_THRESHOLD = 128

SAFETY_PTOS_SPECIAL_VIDEO = "safety_ptos_special_video"
TOPIC_SPECIAL_VIDEO = "safety-ptos-special-video"
SAFETY_PTOS_ADULT_CONTENT_LEADING_FRAMES = "safety_ptos_adult_content_leading_frames"
TOPIC_ADULT_CONTENT_LEADING_FRAMES = "safety-ptos-adult-content-leading-frames"
ADULT_CONTENT_LEADING_FRAMES_CROP_SECONDS = 10.0
ADULT_CONTENT_LEADING_FRAMES_MIN_DURATION_SECONDS = 30.0
ADULT_CONTENT_LEADING_FRAMES_UNCONDITIONAL_FAV = 512
SAFETY_PTOS_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION = (
    "safety_ptos_media_injected_adult_infrared_video_spam_detection"
)
TOPIC_MEDIA_INJECTED_ADULT_INFRARED_VIDEO_SPAM_DETECTION = (
    "safety-ptos-media-injected-adult-infrared-video-spam-detection"
)
MEDIA_INJECTED_REASONING_FAV = 512
SAFETY_PTOS_KEY_FRAMES = "safety_ptos_key_frames"
TOPIC_KEY_FRAMES = "safety-ptos-key-frames"
KEY_FRAMES_UNCONDITIONAL_FAV = 1024
SAFETY_PTOS_MANY_MEDIA = "safety_ptos_many_media"
TOPIC_MANY_MEDIA = "safety-ptos-many-media"
MANY_MEDIA_CHUNK_SIZE = 4
SPECIAL_VIDEO_TASK_TYPES = frozenset(
    {SAFETY_PTOS_SPECIAL_VIDEO, SAFETY_PTOS_ADULT_CONTENT_LEADING_FRAMES}
)
DELUXE_TIER_TASK_TYPES = frozenset(
    {
        SAFETY_PTOS_DELUXE,
        SAFETY_PTOS_SPECIAL_VIDEO,
        SAFETY_PTOS_ADULT_CONTENT_LEADING_FRAMES,
        SAFETY_PTOS_KEY_FRAMES,
        SAFETY_PTOS_MANY_MEDIA,
    }
)
