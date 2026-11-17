import asyncio
import logging
import traceback
import multiprocessing
from queue import Empty, Queue
from threading import Event
from multiprocessing import Process

from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.schedules.init import init_proc
from grox.core.schedules.types import TaskResult, TaskPayload
from grox.core.schedules.context import ScheduleContext
from grox.core.loading import load_all
from grox.core.generators.registry import build as build_generator
from grox.core.generators.task_generator import TaskGenerator, PriorityTaskGenerator

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, context: ScheduleContext):
        self.config = grox_config.dispatcher
        self.context = context
        self._task_queue: Queue[TaskPayload] = self.context["task_queue"]
        self._resp_queue: Queue[TaskResult] = self.context["resp_queue"]
        self._shutdown_event: Event = self.context["shutdown_event"]
        self._queue_connection_shutdown_event: Event = self.context[
            "queue_connection_shutdown_event"
        ]
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

    def _is_queue_connection_shutdown(self) -> bool:
        try:
            return self._queue_connection_shutdown_event.is_set()
        except BrokenPipeError:
            logger.error("Broken pipe error, assuming queue connection shutdown")
            return True
        except Exception:
            logger.error(
                f"Error checking shutdown event, assuming queue connection shutdown: {traceback.format_exc()}"
            )
            return True

    async def _init_run(self):
        await init_proc("dispatcher")
        self._in_flights: set[str] = set()
        self._task_generator = (
            self._get_task_generator() if self.config.task_generators else None
        )
        if self._task_generator is not None:
            await self._task_generator.start()

    def _get_task_generator(self) -> TaskGenerator:
        load_all()
        generators: list[tuple[TaskGenerator, int]] = [
            (build_generator(cfg), cfg.weight) for cfg in self.config.task_generators
        ]
        return PriorityTaskGenerator(generators)

    async def _submit_task(self, task_payload: TaskPayload) -> None:
        inflight_gauge = Metrics.gauge("dispatcher.inflight.count")
        self._in_flights.add(task_payload.payload_id)
        inflight_gauge.set(len(self._in_flights))
        Metrics.counter("dispatcher.task.sent.count").add(
            1,
            attributes={
                "task_type": task_payload.task_type
                if task_payload.task_type
                else "none"
            },
        )
        self._task_queue.put(task_payload)
        logger.debug(
            f"Submitted task: {task_payload.payload_id}, queue size: {self._task_queue.qsize()}"
        )

    async def _fill_loop(self):
        if self._task_generator is None:
            logger.info("No task generators configured; dispatcher fill loop idle")
            return
        logger.info("Starting fill loop")
        while not self._is_shutdown():
            try:
                async for task_payload in self._task_generator.poll():
                    if task_payload is None:
                        await asyncio.sleep(0.01)
                        continue
                    while len(self._in_flights) >= self.config.max_in_flight:
                        await asyncio.sleep(0.01)
                    await self._submit_task(task_payload)
                await asyncio.sleep(1)
            except Exception:
                logger.error(
                    f"Error polling from task queues: {traceback.format_exc()}"
                )
                await asyncio.sleep(1)

    async def _poll_result(self) -> TaskResult | None:
        try:
            res = self._resp_queue.get(block=False)
            logger.debug(f"Dispatcher received result: {res.task.payload_id}")
            Metrics.counter("dispatcher.result.received.count").add(1)
            return res
        except Empty:
            return None
        except BrokenPipeError:
            logger.error("Broken pipe error, shutting down")
            return None
        except Exception:
            logger.error(f"failed to poll result: {traceback.format_exc()}")
            return None

    async def _result_loop(self) -> None:
        if self._task_generator is None:
            logger.info("No task generators configured; dispatcher result loop idle")
            return
        logger.info("Starting result loop")
        inflight_gauge = Metrics.gauge("dispatcher.inflight.count")
        while not self._is_shutdown() or self._in_flights:
            try:
                result = await self._poll_result()
                if result is None:
                    await asyncio.sleep(0.01)
                    continue
                task = result.task
                if task.payload_id in self._in_flights:
                    self._in_flights.remove(task.payload_id)
                    inflight_gauge.set(len(self._in_flights))
                if result.success:
                    Metrics.counter("dispatcher.result.success.count").add(1)
                    await self._task_generator.ack(result)
                else:
                    logger.error(
                        f"Task {task.payload_id} failed, error is {result.error}"
                    )
                    origin = self._task_generator.identify_task_origin(result)
                    Metrics.counter("dispatcher.result.failed.final.count").add(
                        1, attributes={"origin": origin or "unknown"}
                    )
                    self._task_generator.on_terminal_failure(result)
                    if origin is None:
                        logger.warning(
                            f"No origin found for task {task.payload_id}, skipping ack"
                        )
                    else:
                        await self._task_generator.ack(result)
            except BrokenPipeError:
                logger.error("Broken pipe error, shutting down")
                break
        logger.warning("Result loop finished")

    async def _wait_for_queue_connection_shutdown(self):
        while not self._is_queue_connection_shutdown():
            await asyncio.sleep(1)
        logger.warning("Shutting down task generators")
        if self._task_generator is not None:
            await self._task_generator.stop()
        logger.warning("Task generators stopped")

    async def _inflight_monitor_loop(self) -> None:
        inflight_gauge = Metrics.gauge("dispatcher.inflight.count")
        while not self._is_shutdown():
            inflight_gauge.set(len(self._in_flights))
            await asyncio.sleep(15)

    async def _run(self, started_event: Event):
        await self._init_run()
        started_event.set()
        await asyncio.gather(
            self._fill_loop(),
            self._result_loop(),
            self._inflight_monitor_loop(),
            self._wait_for_queue_connection_shutdown(),
        )

    def run(self, started_event: Event):
        asyncio.run(self._run(started_event))

    async def start(self):
        logger.info("Starting Grox dispatcher...")
        started_event = multiprocessing.Event()
        self._process = Process(
            target=self.run, args=(started_event,), name="grox-dispatcher"
        )
        self._process.start()
        started_event.wait()
        logger.info("Grox dispatcher started")

    async def stop(self):
        logger.warning("Stopping Grox dispatcher...")
        if self._process and self._process.is_alive():
            self._process.join(self.config.graceful_shutdown_timeout)
        else:
            logger.warning("Dispatcher process is not alive, skipping join")
        logger.warning("Dispatcher stopped")
