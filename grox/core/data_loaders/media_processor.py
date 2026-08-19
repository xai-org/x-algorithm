import asyncio
import logging
import os
import time
import traceback
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as MultiprocessingEvent
from queue import Empty
from uuid import uuid4

from monitor.logging import Logging
from monitor.metrics import Metrics
from pydantic import BaseModel

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.media_loader import MediaLoader
from grox.core.schedules.init import init_proc

logger = logging.getLogger(__name__)


class _Result(BaseModel):
    post_id: str
    request_id: str
    post: Post | None = None
    error: str | None = None


class _MediaWorker:
    def __init__(
        self, task_queue: Queue, resp_queue: Queue, shutdown_event: MultiprocessingEvent
    ):
        self._task_queue: Queue[tuple[Post, dict[str, str], str]] = task_queue
        self._resp_queue: Queue[_Result] = resp_queue
        self._shutdown_event: MultiprocessingEvent = shutdown_event

    async def _hydrate(self, post: Post):
        await MediaLoader.hydrate_post(post)

    async def _process(self, post: Post, ctx: dict[str, str], request_id: str) -> None:
        attributes = {"pid": str(os.getpid())}
        with Metrics.tracer("media_prod").start_as_current_span("post.process"):
            Logging.set_context(**ctx)
            start = time.perf_counter()
            try:
                Metrics.counter("media_proc.post.total.count").add(
                    1, attributes=attributes
                )
                await self._hydrate(post)
                Metrics.counter("media_proc.post.success.count").add(
                    1, attributes=attributes
                )
                self._resp_queue.put(
                    _Result(post_id=post.id, request_id=request_id, post=post)
                )
            except Exception as e:
                Metrics.counter("media_proc.post.error.count").add(
                    1, attributes=attributes
                )
                self._resp_queue.put(
                    _Result(post_id=post.id, request_id=request_id, error=str(e))
                )
                logger.error(
                    f"error hydrating post {post.id}: {traceback.format_exc()}"
                )
            finally:
                end = time.perf_counter()
                Metrics.histogram("media_proc.post.duration").record(
                    end - start, attributes=attributes
                )

    async def _init_run(self) -> None:
        await init_proc("media_proc")

    async def _run(self) -> None:
        logger.info("starting media worker process loop")
        while not self._is_shutdown() or not self._task_queue.empty():
            try:
                post, ctx, request_id = self._task_queue.get(block=False)
            except Empty:
                await asyncio.sleep(0.01)
                continue
            try:
                asyncio.create_task(self._process(post, ctx, request_id))
            except Exception:
                logger.error(
                    f"error hydrating post {post.id}: {traceback.format_exc()}"
                )
        logger.warning("media worker process loop done")

    def run(self) -> None:
        async def wrapper():
            await self._init_run()
            await self._run()

        asyncio.run(wrapper())

    def _start_loop(self) -> Process:
        process = Process(target=self.run)
        process.start()
        return process

    def start(self) -> list[Process]:
        return [
            self._start_loop() for _ in range(grox_config.media_hydration.max_workers)
        ]

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


class MediaProcessor:
    _task_queue: Queue = Queue()
    _resp_queue: Queue = Queue()
    _shutdown_event = Event()
    _inflights: dict[str, asyncio.Future[Post]] = {}
    _workers: list[Process] = []
    _initialized = False
    _result_task: asyncio.Task | None = None

    @classmethod
    async def process(cls, post: Post) -> None:
        if not cls._initialized:
            raise RuntimeError("media processor not initialized")
        request_id, future = cls._submit(post)
        try:
            hydrated = await future
        except BaseException:
            cls._inflights.pop(request_id, None)
            raise
        post.media = hydrated.media
        post.url_videos = hydrated.url_videos
        post.card = hydrated.card
        post.broadcast_metadata = hydrated.broadcast_metadata
        post.quoted_post = hydrated.quoted_post
        post.ancestors = hydrated.ancestors
        post.screenshot = hydrated.screenshot
        post.cardsV2 = hydrated.cardsV2
        post.article_metadata = hydrated.article_metadata
        post.descendants = hydrated.descendants
        post.list_metadata = hydrated.list_metadata
        post.chat_group_metadata = hydrated.chat_group_metadata

    @classmethod
    def _submit(cls, post: Post) -> tuple[str, asyncio.Future[Post]]:
        request_id = f"{post.id}:{uuid4().hex}"
        cls._task_queue.put((post, Logging.get_context(), request_id))
        future: asyncio.Future[Post] = asyncio.get_running_loop().create_future()
        cls._inflights[request_id] = future
        return request_id, future

    @classmethod
    async def _result_loop(cls) -> None:
        logger.info("media processor result loop started")
        while not cls._shutdown_event.is_set() or cls._inflights:
            try:
                result = cls._resp_queue.get(block=False)
                future = cls._inflights.pop(result.request_id, None)
                if future is None:
                    logger.warning(
                        f"no future found for post {result.post_id}, dropping result"
                    )
                    continue
                if future.done():
                    logger.warning(
                        f"future for post {result.post_id} already done, dropping result"
                    )
                    continue
                if not result.post:
                    future.set_exception(
                        Exception(result.error if result.error else "no post captured")
                    )
                else:
                    future.set_result(result.post)
            except Empty:
                await asyncio.sleep(0.01)
            except Exception:
                logger.error(f"Error processing result: {traceback.format_exc()}")
        logger.warning("media processor result loop done")

    @classmethod
    def start(cls) -> None:
        logger.info("starting media processor")
        cls._workers = _MediaWorker(
            cls._task_queue, cls._resp_queue, cls._shutdown_event
        ).start()
        cls._result_task = asyncio.create_task(cls._result_loop())
        cls._initialized = True

    @classmethod
    async def stop(cls, timeout: float = 5) -> None:
        logger.warning("stopping media processor")
        cls._shutdown_event.set()
        for worker in cls._workers:
            if worker.is_alive():
                worker.join(timeout)
        if cls._result_task and not cls._result_task.done():
            cls._result_task.cancel()
        logger.warning("media processor stopped")
