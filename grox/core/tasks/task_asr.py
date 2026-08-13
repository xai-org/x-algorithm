import logging
import traceback

from monitor.metrics import Metrics
from strato_http.queries.media_asr_transcript import StratoMediaAsrTranscript

from grox.core.data_loaders.asr_processor import ASRProcessor
from grox.core.data_loaders.data_types import Post, Video
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import TaskWithPost

logger = logging.getLogger(__name__)

_transcript_store = StratoMediaAsrTranscript()

_INPUT_ERRORS = {"no_audio_stream", "ffmpeg_timeout"}


def _is_input_error(error: str) -> bool:
    return error in _INPUT_ERRORS or error.startswith("ffmpeg_error:")


def _get_video_url(video: Video) -> tuple[str | None, bool]:
    if video.animatedGifInfo:
        v = video.animatedGifInfo.get_best_variant()
        if v and v.url:
            return v.url, True
    if video.videoInfo:
        v = video.videoInfo.get_best_variant()
        if v and v.url:
            return v.url, False
    if video.url:
        return video.url, False
    return None, False


class TaskASRTranscription(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.asr_transcription.total.count").add(1)

        if not post.media:
            logger.debug(f"Post {post.id} has no media, skipping ASR")
            Metrics.counter("task.asr_transcription.skipped.count").add(
                1, attributes={"reason": "no_media"}
            )
            return

        videos = [m for m in post.media if isinstance(m, Video)]
        if not videos:
            logger.debug(f"Post {post.id} has no video, skipping ASR")
            Metrics.counter("task.asr_transcription.skipped.count").add(
                1, attributes={"reason": "no_video"}
            )
            return

        Metrics.counter("task.asr_transcription.has_video.count").add(1)

        for video in videos:
            video_url, is_animated_gif = _get_video_url(video)
            if not video_url:
                continue

            if is_animated_gif:
                logger.debug(
                    f"Post {post.id} video {video.id} is an animated GIF, skipping ASR (no audio)"
                )
                Metrics.counter("task.asr_transcription.skipped.count").add(
                    1, attributes={"reason": "animated_gif"}
                )
                continue

            media_id = video.id
            Metrics.counter("task.asr_transcription.attempted.count").add(1)

            transcript = (
                await _fetch_cached_transcript(media_id, post.id) if media_id else None
            )
            if transcript is not None:
                if video.convo_video is not None:
                    video.convo_video.asr_transcript = transcript
                continue

            Metrics.counter("asr_proc.total.count").add(1)
            transcript, error = await ASRProcessor.process(
                post.id, video_url, media_id=media_id
            )

            if error:
                if _is_input_error(error):
                    Metrics.counter("task.asr_transcription.skipped.count").add(
                        1, attributes={"reason": error}
                    )
                    logger.debug(
                        f"ASR skipped for post {post.id} video {video.id}: {error}"
                    )
                else:
                    Metrics.counter("task.asr_transcription.failed.count").add(
                        1, attributes={"reason": error}
                    )
                    logger.warning(
                        f"ASR failed for post {post.id} video {video.id}: {error}"
                    )
            elif transcript:
                Metrics.counter("asr_proc.success.count").add(1)
                if video.convo_video is not None:
                    video.convo_video.asr_transcript = transcript
                Metrics.counter("task.asr_transcription.success.count").add(1)
                logger.info(
                    f"ASR completed for post {post.id} video {video.id}, transcript_len={len(transcript)}"
                )
                if media_id:
                    await _save_transcript(media_id, transcript, post.id)
            else:
                Metrics.counter("asr_proc.success.count").add(1)
                Metrics.counter("task.asr_transcription.success.count").add(1)
                Metrics.counter("task.asr_transcription.empty_transcript.count").add(1)
                logger.info(
                    f"ASR returned empty transcript for post {post.id} video {video.id} (no speech)"
                )
                if media_id:
                    await _save_transcript(media_id, "", post.id)


async def _fetch_cached_transcript(media_id: str, post_id: str) -> str | None:
    try:
        transcript = await _transcript_store.fetch(media_id)
        if transcript is not None:
            Metrics.counter("task.asr_transcription.cache_hit.count").add(1)
            logger.info(
                f"ASR cache hit for post {post_id} media {media_id}, transcript_len={len(transcript)}"
            )
            return transcript
        Metrics.counter("task.asr_transcription.cache_miss.count").add(1)
    except Exception:
        logger.warning(
            f"ASR cache fetch failed for media {media_id}: {traceback.format_exc()}"
        )
        Metrics.counter("task.asr_transcription.cache_fetch_error.count").add(1)
    return None


async def _save_transcript(media_id: str, transcript: str, post_id: str) -> None:
    try:
        await _transcript_store.put(media_id, transcript)
        Metrics.counter("task.asr_transcription.cache_save.count").add(1)
        logger.info(f"Saved ASR transcript for post {post_id} media {media_id}")
    except Exception:
        logger.warning(
            f"ASR cache save failed for media {media_id}: {traceback.format_exc()}"
        )
        Metrics.counter("task.asr_transcription.cache_save_error.count").add(1)
