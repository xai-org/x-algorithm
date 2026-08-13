import asyncio
import gc
import logging
import random
import time

import setproctitle
from monitor.logging import Logging
from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.schedules.context import prevent_default

logger = logging.getLogger(__name__)


async def init_proc(proc_name: str, defer_metrics: bool = False):
    prevent_default()
    Logging.config(grox_config.logging)
    if not defer_metrics:
        Metrics.init(proc_name, grox_config.metrics)
    setproctitle.setproctitle(proc_name)
    logger.info(f"Changed process title to {proc_name}")
    asyncio.create_task(periodic_gc(proc_name))


def init_metrics(proc_name: str):
    Metrics.init(proc_name, grox_config.metrics)


async def periodic_gc(proc_name: str):
    while True:
        seconds = grox_config.periodic_gc.interval + random.randint(
            0, int(grox_config.periodic_gc.jitter)
        )
        await asyncio.sleep(seconds)
        logger.info(f"Running periodic GC for {proc_name}")
        start = time.perf_counter()
        gc.collect()
        end = time.perf_counter()
        logger.info(f"GC for {proc_name} took {end - start:.2f} seconds")
