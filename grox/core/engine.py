import asyncio
import logging
import multiprocessing
import os
import time
import traceback
from multiprocessing import Process
from queue import Empty, Queue
from threading import Event

from monitor.logging import Logging
from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.data_loaders.asr_processor import ASRProcessor
from grox.core.data_loaders.media_processor import MediaProcessor
from grox.core.loading import load_all
from grox.core.plans.plan_master import PlanMaster
from grox.core.processors import start_registered, stop_registered
from grox.core.schedules.context import ScheduleContext
from grox.core.schedules.init import init_metrics, init_proc
from grox.core.schedules.types import TaskPayload, TaskResult

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, context: ScheduleContext):
        self.config = grox_config.engine
        self.context = context
        self._task_queue: Queue[TaskPayload] = self.context["task_queue"]
        self._resp_queue: Queue[TaskResult] = self.context["resp_queue"]
        self._shutdown_event: Event = self.context["shutdown_event"]
        self._process = None

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

    async def _init_run(self):
        await init_proc("engine", defer_metrics=True)
        load_all()
        MediaProcessor.start()
        ASRProcessor.start()
        start_registered()
        init_metrics("engine")

    async def _process_task(self, task: TaskPayload) -> TaskResult:
        logger.debug("engine started processing task")
        start = time.perf_counter()
        res = await PlanMaster.exec(task)
        end = time.perf_counter()
        logger.debug(f"engine finished processing task in {end - start:.2f} seconds")
        Metrics.histogram("engine.task.processing_time").record(end - start)
        return res

    async def _run_task(self, task: TaskPayload):
        start = time.perf_counter()
        with Metrics.tracer("engine").start_as_current_span("task.root"):
            Logging.set_context(task=task.payload_id)
            if task.post:
                Logging.set_context(post=task.post.id)
            if task.user:
                Logging.set_context(user=task.user.id)
            try:
                res = await asyncio.wait_for(
                    self._process_task(task), timeout=self.config.task_timeout
                )
                self._resp_queue.put(res)
                Metrics.counter("engine.task.success.count").add(1)
            except TimeoutError:
                logger.error(
                    f"task timed out after {self.config.task_timeout}s, task_type={task.task_type}"
                )
                self._resp_queue.put(
                    TaskResult(
                        task=task,
                        success=False,
                        error=f"engine task timed out after {self.config.task_timeout}s",
                        task_started_at=start,
                        task_finished_at=time.perf_counter(),
                    )
                )
                Metrics.counter("engine.task.timeout.count").add(1)
                Metrics.counter("engine.task.failed.count").add(1)
            except asyncio.CancelledError:
                logger.error(
                    f"task cancelled while awaiting a dependency, task_type={task.task_type}"
                )
                self._resp_queue.put(
                    TaskResult(
                        task=task,
                        success=False,
                        error="engine task cancelled while awaiting a dependency",
                        task_started_at=start,
                        task_finished_at=time.perf_counter(),
                    )
                )
                Metrics.counter("engine.task.cancelled.count").add(1)
                Metrics.counter("engine.task.failed.count").add(1)
            except Exception as e:
                logger.error(f"failed to process task, error: {traceback.format_exc()}")
                self._resp_queue.put(
                    TaskResult(
                        task=task,
                        success=False,
                        error=str(e),
                        task_started_at=start,
                        task_finished_at=time.perf_counter(),
                    )
                )
                Metrics.counter("engine.task.failed.count").add(1)

    async def _poll_task(self) -> TaskPayload | None:
        logger.debug(f"engine polling task, queue size: {self._task_queue.qsize()}")
        try:
            task = self._task_queue.get(block=False)
            logger.debug(f"engine received task: {task.payload_id}")
            Metrics.counter("engine.task.received.count").add(1)
            return task
        except Empty:
            logger.debug("engine polling task returned None")
            return None
        except BrokenPipeError:
            logger.error("Broken pipe error, shutting down")
            return None
        except Exception:
            logger.error(f"failed to poll task: {traceback.format_exc()}")
            return None

    async def _run(self, started_event: Event):
        await self._init_run()
        started_event.set()
        while not self._is_shutdown() or not self._task_queue.empty():
            task = await self._poll_task()
            if task is None:
                await asyncio.sleep(0.1)
                continue
            asyncio.create_task(self._run_task(task))
        logger.warning("engine stopped")

    def run(self, started_event: Event):
        asyncio.run(self._run(started_event))
        os._exit(0)

    async def start(self):
        logger.info("Starting Grox engine...")
        started_event = multiprocessing.Event()
        self._process = Process(
            target=self.run, args=(started_event,), name="grox-engine"
        )
        self._process.start()
        started_event.wait()
        logger.info("Grox engine started")

    async def stop(self):
        logger.warning("Stopping Grox engine...")
        if self._process and self._process.is_alive():
            self._process.join(self.config.graceful_shutdown_timeout)
        else:
            logger.warning("Engine process is not alive, skipping join")
        await stop_registered()
        await MediaProcessor.stop()
        await ASRProcessor.stop()
        logger.warning("Engine stopped")
