import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import time
import traceback
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as MultiprocessingEvent
from queue import Empty

import aiohttp
from cachetools import TTLCache
from monitor.logging import Logging
from monitor.metrics import Metrics
from pydantic import BaseModel

from grox.config.config import grox_config
from grox.core.schedules.init import init_proc

logger = logging.getLogger(__name__)


class _ASRRequest(BaseModel):
    key: str
    post_id: str
    video_url: str
    max_audio_duration_s: float


class _ASRResult(BaseModel):
    key: str
    post_id: str
    transcript: str | None = None
    error: str | None = None


def _extract_wav_from_url(
    video_url: str, max_duration_s: float | None = None
) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-timeout",
            "60000000",
            "-rw_timeout",
            "60000000",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-i",
            video_url,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
        ]
        if max_duration_s is not None and max_duration_s > 0:
            cmd += ["-t", str(max_duration_s)]
        cmd.append(wav_path)
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode != 0:
            if not os.path.exists(wav_path):
                return None
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        with open(wav_path, "rb") as f:
            return f.read()


def _clean_asr(raw: str) -> str:
    if "<asr_text>" in raw:
        raw = raw.split("<asr_text>", 1)[1]
    if "</asr_text>" in raw:
        raw = raw.split("</asr_text>", 1)[0]
    return raw.strip()


class _ASRWorker:
    def __init__(
        self, task_queue: Queue, resp_queue: Queue, shutdown_event: MultiprocessingEvent
    ):
        self._task_queue: Queue[tuple[_ASRRequest, dict[str, str]]] = task_queue
        self._resp_queue: Queue[_ASRResult] = resp_queue
        self._shutdown_event: MultiprocessingEvent = shutdown_event

    @staticmethod
    def _metrics_attributes() -> dict[str, str]:
        return {"pid": str(os.getpid())}

    async def _transcribe(self, request: _ASRRequest) -> str | None:
        asr_config = grox_config.asr

        t_start = time.monotonic()
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(
            None, _extract_wav_from_url, request.video_url, request.max_audio_duration_s
        )
        t_extract = time.monotonic() - t_start
        Metrics.histogram("asr_proc.extract_duration_s").record(
            t_extract, attributes=self._metrics_attributes()
        )
        if wav_bytes is None:
            return None
        Metrics.histogram("asr_proc.audio_bytes").record(
            len(wav_bytes), attributes=self._metrics_attributes()
        )
        logger.debug(
            f"Extracted audio in {t_extract:.2f}s, size={len(wav_bytes)} bytes"
        )

        t_start = time.monotonic()
        b64_audio = base64.b64encode(wav_bytes).decode()
        body = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{b64_audio}"},
                        },
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                }
            ],
            "temperature": asr_config.temperature,
            "max_tokens": asr_config.max_tokens,
        }

        async with self._session.post(
            f"{asr_config.endpoint}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=asr_config.timeout),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                raw_transcript = data["choices"][0]["message"]["content"].strip()
                transcript = _clean_asr(raw_transcript)

                t_transcribe = time.monotonic() - t_start
                Metrics.histogram("asr_proc.transcribe_duration_s").record(
                    t_transcribe, attributes=self._metrics_attributes()
                )
                Metrics.histogram("asr_proc.transcript_chars").record(
                    len(transcript), attributes=self._metrics_attributes()
                )

                if "usage" in data:
                    usage = data["usage"]
                    if "prompt_tokens" in usage:
                        Metrics.histogram("asr_proc.prompt_tokens").record(
                            usage["prompt_tokens"],
                            attributes=self._metrics_attributes(),
                        )
                    if "completion_tokens" in usage:
                        Metrics.histogram("asr_proc.completion_tokens").record(
                            usage["completion_tokens"],
                            attributes=self._metrics_attributes(),
                        )
                    if "total_tokens" in usage:
                        Metrics.histogram("asr_proc.total_tokens").record(
                            usage["total_tokens"], attributes=self._metrics_attributes()
                        )

                return transcript
            else:
                error_text = await resp.text()
                raise Exception(
                    f"ASR request failed with status {resp.status}: {error_text}"
                )

    async def _process(self, request: _ASRRequest, ctx: dict[str, str]) -> None:
        def result(**kwargs) -> _ASRResult:
            return _ASRResult(key=request.key, post_id=request.post_id, **kwargs)

        with Metrics.tracer("asr_proc").start_as_current_span("asr.process"):
            Logging.set_context(**ctx)
            start = time.perf_counter()
            try:
                transcript = await self._transcribe(request)
                if transcript is None:
                    logger.debug(
                        f"Video has no audio stream for post {request.post_id}, skipping ASR"
                    )
                    self._resp_queue.put(result(error="no_audio_stream"))
                else:
                    self._resp_queue.put(result(transcript=transcript))
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"FFmpeg timeout extracting audio for post {request.post_id}"
                )
                self._resp_queue.put(result(error="ffmpeg_timeout"))
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                logger.warning(f"FFmpeg error for post {request.post_id}: {error_msg}")
                self._resp_queue.put(result(error=f"ffmpeg_error: {error_msg}"))
            except asyncio.TimeoutError:
                logger.warning(f"ASR request timed out for post {request.post_id}")
                self._resp_queue.put(result(error="asr_timeout"))
            except Exception as e:
                logger.error(
                    f"ASR processing failed for post {request.post_id}: {traceback.format_exc()}"
                )
                self._resp_queue.put(result(error=str(e)))
            finally:
                end = time.perf_counter()
                Metrics.histogram("asr_proc.duration").record(
                    end - start, attributes=self._metrics_attributes()
                )

    async def _init_run(self) -> None:
        await init_proc("asr_proc")
        self._session = aiohttp.ClientSession()

    async def _run(self) -> None:
        logger.info("starting ASR worker process loop")
        pending: set[asyncio.Task] = set()
        while not self._is_shutdown() or not self._task_queue.empty():
            try:
                request, ctx = self._task_queue.get(block=False)
            except Empty:
                await asyncio.sleep(0.01)
                continue
            try:
                task = asyncio.create_task(self._process(request, ctx))
                pending.add(task)
                task.add_done_callback(pending.discard)
            except Exception:
                logger.error(
                    f"error processing ASR request {request.post_id}: {traceback.format_exc()}"
                )
        if pending:
            logger.info(f"ASR worker draining {len(pending)} in-flight tasks")
            await asyncio.gather(*pending, return_exceptions=True)
        logger.warning("ASR worker process loop done")

    def run(self) -> None:
        async def wrapper():
            await self._init_run()
            try:
                await self._run()
            finally:
                await self._session.close()

        asyncio.run(wrapper())

    def _start_loop(self) -> Process:
        process = Process(target=self.run)
        process.start()
        return process

    def start(self) -> list[Process]:
        return [self._start_loop() for _ in range(grox_config.asr.max_workers)]

    def _is_shutdown(self) -> bool:
        try:
            return self._shutdown_event.is_set()
        except BrokenPipeError:
            logger.error("Broken pipe error, assuming shutdown")
            return True
        except Exception:
            logger.error(
                f"Error checking shutdown event, assuming shutdown: {traceback.format_exc()}"
            )
            return True


class ASRProcessor:
    _task_queue: Queue = Queue()
    _resp_queue: Queue = Queue()
    _shutdown_event = Event()
    _inflights: dict[str, asyncio.Future[str | None]] = {}
    _workers: list[Process] = []
    _initialized = False
    _result_task: asyncio.Task | None = None
    _cache: TTLCache = TTLCache(maxsize=1_000, ttl=300)

    @classmethod
    async def process(
        cls,
        post_id: str,
        video_url: str,
        media_id: str | None = None,
        max_audio_duration_s: float | None = None,
    ) -> tuple[str | None, str | None]:
        if not cls._initialized:
            raise RuntimeError("ASR processor not initialized")

        key = media_id or video_url

        cached = cls._cache.get(key)
        if cached is not None:
            Metrics.counter("asr_proc.cache_hit.count").add(1)
            return cached, None

        if max_audio_duration_s is None:
            max_audio_duration_s = grox_config.asr.max_audio_duration_s

        future = cls._submit(key, post_id, video_url, max_audio_duration_s)
        result: _ASRResult = await future

        if result.transcript is not None:
            cls._cache[key] = result.transcript

        return result.transcript, result.error

    @classmethod
    def _submit(
        cls, key: str, post_id: str, video_url: str, max_audio_duration_s: float
    ) -> asyncio.Future[_ASRResult]:
        if key in cls._inflights:
            return cls._inflights[key]

        request = _ASRRequest(
            key=key,
            post_id=post_id,
            video_url=video_url,
            max_audio_duration_s=max_audio_duration_s,
        )
        cls._task_queue.put((request, Logging.get_context()))

        future: asyncio.Future[_ASRResult] = asyncio.get_running_loop().create_future()
        cls._inflights[key] = future
        return future

    @classmethod
    async def _result_loop(cls) -> None:
        logger.info("ASR processor result loop started")
        while not cls._shutdown_event.is_set() or cls._inflights:
            try:
                result: _ASRResult = cls._resp_queue.get(block=False)
                future = cls._inflights.pop(result.key, None)
                if not future:
                    logger.warning(
                        f"no future found for post {result.post_id} key {result.key}"
                    )
                    continue
                future.set_result(result)
            except Empty:
                await asyncio.sleep(0.01)
            except Exception:
                logger.error(f"Error processing ASR result: {traceback.format_exc()}")
        logger.warning("ASR processor result loop done")

    @classmethod
    def start(cls) -> None:
        if cls._initialized:
            logger.warning("ASR processor already initialized")
            return

        logger.info(
            f"starting ASR processor with {grox_config.asr.max_workers} workers"
        )
        cls._workers = _ASRWorker(
            cls._task_queue, cls._resp_queue, cls._shutdown_event
        ).start()
        cls._result_task = asyncio.create_task(cls._result_loop())
        cls._initialized = True

    @classmethod
    async def stop(cls, timeout: float = 5) -> None:
        logger.warning("stopping ASR processor")
        cls._shutdown_event.set()
        for worker in cls._workers:
            if worker.is_alive():
                worker.join(timeout)
        if cls._result_task and not cls._result_task.done():
            cls._result_task.cancel()
        logger.warning("ASR processor stopped")
