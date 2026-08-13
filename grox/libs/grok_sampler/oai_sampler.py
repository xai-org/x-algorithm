import json
import logging
import time
import traceback

from monitor.metrics import Metrics
from openai import AsyncOpenAI

from grok_sampler.config import OaiModelConfig

logger = logging.getLogger(__name__)


class OaiSampler:
    def __init__(self, config: OaiModelConfig):
        self.model = config.model
        self.api_base_url = config.api_base_url
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.client = AsyncOpenAI(
            base_url=config.api_base_url,
            api_key="unused",
            timeout=config.timeout,
        )

    async def sample(self, messages: list[dict], **kwargs) -> str:
        attributes = {"model_name": self.api_base_url}
        Metrics.counter("llm.sample.count").add(1, attributes=attributes)

        logger.info(
            f"OaiSampler: request to model={self.model}, messages={len(messages)}"
        )

        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        json_schema = kwargs.get("json_schema")
        if json_schema:
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": json.loads(json_schema),
                    "strict": True,
                },
            }

        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - start
            Metrics.counter("llm.sample.success.count").add(1, attributes=attributes)
            logger.info(f"OaiSampler: response in {elapsed:.2f}s, {len(content)} chars")
            return content
        except Exception:
            logger.error(
                f"OaiSampler: error calling model={self.model}, elapsed={time.perf_counter() - start:.2f}s: {traceback.format_exc()}"
            )
            Metrics.counter("llm.sample.error.count").add(1, attributes=attributes)
            raise
        finally:
            Metrics.histogram("llm.sample.duration").record(
                time.perf_counter() - start, attributes=attributes
            )
