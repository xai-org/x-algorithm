import asyncio
import json
import logging
import time
import traceback
import urllib.error
import urllib.request
import uuid

from grox.core.lib.utils import detect_image_content_type
from grox.core.data_loaders.data_types import Image, Post, Video
from grox.flows.ptos.state import SafetyPolicyCategory, SafetyPtosState
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.flows.ptos.constants import SAFETY_PTOS_DELUXE
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
)

logger = logging.getLogger(__name__)

_SAFEMODEL_URL = "http://safemodel-prod.global.svc.cluster.local/infer"
_SLUG = "sex-and-nudity"
_CHECKPOINT_GCS = "gs://safetool-classifier-assets-prod/classifiers/sex-and-nudity/train_jobs/20b742f8/checkpoint.pt"

_TASK_TIMEOUT_S = 15.0
_HTTP_TIMEOUT_S = 5.0
_MAX_RETRIES = 3
_MAX_FRAMES_PER_VIDEO = 8
_MAX_PAYLOADS_PER_POST = 12
_ADULT_POSITIVE_BUCKETS = {"R", "X"}
_METRIC_PREFIX = "task.safety_ptos_safemodel_sex_nudity"


class TaskSafetyPtosSafemodelSexNudity(TaskWithPost):
    _result_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        try:
            await asyncio.wait_for(cls._run(ctx, post), timeout=_TASK_TIMEOUT_S)
        except asyncio.TimeoutError:
            Metrics.counter(f"{_METRIC_PREFIX}.timeout.count").add(1)
            logger.warning(
                f"Post {post.id}: safemodel timed out after {_TASK_TIMEOUT_S}s"
            )
        except Exception as e:
            Metrics.counter(f"{_METRIC_PREFIX}.error.count").add(
                1, attributes={"reason": "exception"}
            )
            logger.warning(f"Post {post.id}: safemodel failed: {e}")

    @classmethod
    async def _post_is_already_flagged_nsfw(cls, post: Post) -> bool:
        try:
            result = await cls._result_direct_mh.fetch(int(post.id))
        except Exception as e:
            Metrics.counter(f"{_METRIC_PREFIX}.nsfw_lookup_error.count").add(1)
            logger.warning(
                f"Post {post.id}: NSFW MH lookup failed, treating as not flagged: {e}"
            )
            return False
        return bool(
            result and result.safetyBoolMetadata and result.safetyBoolMetadata.isNsfw
        )

    @classmethod
    def _has_adult_content_suspicion(cls, ctx: TaskContext) -> bool:
        annotations = ctx.state(SafetyPtosState).annotations
        if not annotations or not annotations.violatedPolicies:
            return False
        return any(
            v.category == SafetyPolicyCategory.AdultContent
            for v in annotations.violatedPolicies
        )

    @classmethod
    async def _run(cls, ctx: TaskContext, post: Post) -> None:
        is_deluxe = ctx.payload.task_type == SAFETY_PTOS_DELUXE
        flow = "deluxe" if is_deluxe else "standard"

        if not is_deluxe and not cls._has_adult_content_suspicion(ctx):
            Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                1, attributes={"reason": "no_adult_content_suspicion", "flow": flow}
            )
            return

        if is_deluxe and await cls._post_is_already_flagged_nsfw(post):
            Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                1, attributes={"reason": "prior_nsfw", "flow": flow}
            )
            return

        payloads = cls._collect_payloads(post)
        if not payloads:
            Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                1, attributes={"reason": "no_media", "flow": flow}
            )
            return

        has_video = any(kind == "video_frame" for kind, _ in payloads)
        has_video_attr = "true" if has_video else "false"
        n_images = sum(1 for kind, _ in payloads if kind == "image")
        n_video_frames = sum(1 for kind, _ in payloads if kind == "video_frame")

        Metrics.counter(f"{_METRIC_PREFIX}.invoked.count").add(
            1, attributes={"has_video": has_video_attr, "flow": flow}
        )

        post_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                asyncio.to_thread(cls._infer_one_sync, kind, payload_bytes)
                for kind, payload_bytes in payloads
            ],
        )
        post_latency_ms = (time.perf_counter() - post_start) * 1000.0
        Metrics.histogram(f"{_METRIC_PREFIX}.post_latency_ms").record(
            post_latency_ms, attributes={"has_video": has_video_attr, "flow": flow}
        )

        buckets_seen: list[str] = []
        n_errors = 0
        safemodel_positive = False
        max_positive_confidence = 0.0
        for result in results:
            if result is None:
                n_errors += 1
                continue
            prediction, confidence = result
            buckets_seen.append(prediction)
            if prediction in _ADULT_POSITIVE_BUCKETS:
                safemodel_positive = True
                max_positive_confidence = max(max_positive_confidence, confidence)

        if n_errors > 0:
            logger.warning(
                f"Post {post.id}: safemodel had {n_errors} error(s) "
                f"(buckets_seen={buckets_seen}, n_images={n_images}, n_video_frames={n_video_frames}), skipping comparison"
            )
            Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                1, attributes={"reason": "safemodel_errors", "flow": flow}
            )
            return

        logger.info(
            f"Post {post.id} ({flow}): safemodel={'positive' if safemodel_positive else 'negative'} "
            f"(buckets={buckets_seen}, n_errors={n_errors}, n_images={n_images}, n_video_frames={n_video_frames})"
        )

        ctx.state(SafetyPtosState).safemodel_sex_nudity.scored = True
        if safemodel_positive:
            ctx.state(SafetyPtosState).safemodel_sex_nudity.positive = True
            ctx.state(
                SafetyPtosState
            ).safemodel_sex_nudity.confidence = max_positive_confidence

    @classmethod
    def _collect_payloads(cls, post: Post) -> list[tuple[str, bytes]]:
        payloads: list[tuple[str, bytes]] = []
        for media in cls._iter_media(post):
            if isinstance(media, Image):
                if media.convo_image and media.convo_image.content:
                    payloads.append(("image", media.convo_image.content))
            elif isinstance(media, Video):
                if media.convo_video and media.convo_video.frames:
                    for frame in cls._sample_uniform(
                        media.convo_video.frames, _MAX_FRAMES_PER_VIDEO
                    ):
                        payloads.append(("video_frame", frame))
            if len(payloads) >= _MAX_PAYLOADS_PER_POST:
                return payloads[:_MAX_PAYLOADS_PER_POST]
        return payloads

    @staticmethod
    def _iter_media(post: Post):
        if post.media:
            for m in post.media:
                yield m
        if post.quoted_post and post.quoted_post.media:
            for m in post.quoted_post.media:
                yield m

    @staticmethod
    def _sample_uniform(items: list[bytes], k: int) -> list[bytes]:
        n = len(items)
        if n <= k:
            return list(items)
        step = n / k
        return [items[int(i * step)] for i in range(k)]

    @staticmethod
    def _infer_one_sync(kind: str, payload_bytes: bytes) -> tuple[str, float] | None:
        Metrics.counter(f"{_METRIC_PREFIX}.payload_call.count").add(
            1, attributes={"kind": kind}
        )
        start = time.perf_counter()

        content_type = detect_image_content_type(payload_bytes)
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for name, value in [
            ("classifier_slug", _SLUG),
            ("version", "0"),
            ("checkpoint_gcs", _CHECKPOINT_GCS),
        ]:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode(
                    "utf-8"
                )
            )
        file_header = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode("utf-8")
        parts.append(file_header + payload_bytes)
        closing = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = b"\r\n".join(parts) + closing

        last_error_reason = "exception"
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    _SAFEMODEL_URL,
                    data=body,
                    headers={
                        "Content-Type": f"multipart/form-data; boundary={boundary}"
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                last_error_reason = f"http_{e.code}"
                logger.warning(
                    f"safemodel request failed (attempt {attempt + 1}/{_MAX_RETRIES}, kind={kind}): HTTP {e.code} {e.reason}"
                )
            except Exception:
                last_error_reason = "exception"
                logger.warning(
                    f"safemodel request failed (attempt {attempt + 1}/{_MAX_RETRIES}, kind={kind}): {traceback.format_exc()}"
                )
        else:
            Metrics.counter(f"{_METRIC_PREFIX}.error.count").add(
                1, attributes={"reason": last_error_reason}
            )
            return None

        latency_ms = (time.perf_counter() - start) * 1000.0
        Metrics.histogram(f"{_METRIC_PREFIX}.payload_latency_ms").record(
            latency_ms, attributes={"kind": kind}
        )

        prediction = data.get("prediction")
        if not prediction:
            Metrics.counter(f"{_METRIC_PREFIX}.error.count").add(
                1, attributes={"reason": "no_probabilities"}
            )
            return None

        try:
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0
        Metrics.counter(f"{_METRIC_PREFIX}.predicted.count").add(
            1, attributes={"bucket": prediction, "kind": kind}
        )
        return prediction, confidence

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
