import logging
import os
import time
import traceback
from asyncio import get_running_loop
from dataclasses import dataclass, field

import grpc.aio
from monitor.metrics import Metrics
from xai_sdk import AsyncClient
from xai_sdk.proto import chat_pb2
from xai_sdk.tools import web_search

from grok_sampler.config import EapiModelConfig

logger = logging.getLogger(__name__)


def get_loop_id() -> int:
    return id(get_running_loop())


def _rpc_request_id(error: Exception) -> str | None:
    if not isinstance(error, grpc.aio.AioRpcError):
        return None
    for metadata in (error.initial_metadata(), error.trailing_metadata()):
        if not metadata:
            continue
        for key, value in metadata:
            if key == "x-request-id":
                return value
    return None


@dataclass
class WebSearchDetails:
    used: bool = False
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    server_side_tool_usage: dict | None = None


@dataclass
class SampleDetails:
    trace: str | None = None
    web_search: WebSearchDetails = field(default_factory=WebSearchDetails)


class EapiSampler:
    _cached_clients: dict[tuple[int, str, str], AsyncClient] = {}

    def __init__(self, eapi_config: EapiModelConfig):
        self.model: str = eapi_config.model
        self.temperature: float = eapi_config.temperature
        self.timeout: int = eapi_config.timeout
        self.max_tokens = eapi_config.max_resp_len
        self.api_key = eapi_config.api_key
        self.api_host = eapi_config.api_host
        self.log_reasoning_trace: bool = eapi_config.log_reasoning_trace
        self.enable_search: bool = eapi_config.enable_search
        self.reasoning_effort: str | None = eapi_config.reasoning_effort

        api_key = self.api_key or os.getenv("XAI_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required: set EapiModelConfig.api_key or the XAI_API_KEY environment variable."
            )

        self.client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
            "channel_options": [("grpc.enable_retries", 0)],
        }

        if self.api_host:
            self.client_kwargs["api_host"] = self.api_host

    def _get_client(self) -> AsyncClient:
        loop_id = get_loop_id()
        key = (loop_id, self.model, self.api_host)
        if key in self._cached_clients:
            client = self._cached_clients[key]
            return client
        logger.info(
            f"Creating new AsyncClient for {self.model} and {self.api_host} (loop_id={loop_id})"
        )
        client = AsyncClient(**self.client_kwargs)
        self._cached_clients[key] = client
        return client

    def _prompt_len(self, prompt: list[chat_pb2.Message]) -> int:
        def _len(content: chat_pb2.Content) -> int:
            if content.text:
                return len(content.text)
            if content.image_url and content.image_url.image_url:
                return len(content.image_url.image_url)
            return 0

        return sum(_len(content) for message in prompt for content in message.content)

    def _log_request(self, prompt: list[chat_pb2.Message]):
        def _to_log_string(content: chat_pb2.Content) -> str:
            if content.text:
                return content.text
            if content.image_url and content.image_url.image_url:
                return f"[[[[image_bytes({len(content.image_url.image_url)})]]]]"
            return ""

        msg = "\n".join(
            [
                _to_log_string(content)
                for message in prompt
                for content in message.content
            ]
        )
        logger.info(f"inputs: {msg}")

    async def _sample_raw(self, prompt: list[chat_pb2.Message], **kwargs):
        conversation_id = kwargs.get("conversation_id")
        search_allowed_domains = kwargs.get("search_allowed_domains")
        attributes = {"model_name": self.model}
        Metrics.counter("llm.sample.count").add(1, attributes=attributes)
        if kwargs.get("log_prompt", False):
            self._log_request(prompt)

        start = time.perf_counter()

        client = self._get_client()
        try:
            create_kwargs: dict = dict(
                model=self.model,
                conversation_id=conversation_id,
                messages=prompt,
                temperature=self.temperature,
                top_p=0.95,
                max_tokens=self.max_tokens,
            )
            if self.reasoning_effort is not None:
                create_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.enable_search and search_allowed_domains:
                create_kwargs["tools"] = [
                    web_search(allowed_domains=search_allowed_domains)
                ]
            chat = client.chat.create(**create_kwargs)
            response = await chat.sample()
            Metrics.counter("llm.sample.success.count").add(1, attributes=attributes)
            if (
                self.log_reasoning_trace
                and hasattr(response, "reasoning_content")
                and response.reasoning_content
            ):
                logger.info(f"thinking trace: {response.reasoning_content}")
            return response
        except Exception as e:
            logger.error(
                f"Error in sampling EAPI request to model {self.model} and api host {self.api_host}, conversation_id={conversation_id}, x_request_id={_rpc_request_id(e)}, with prompt_len {self._prompt_len(prompt)} with elapsed time {time.perf_counter() - start}: {traceback.format_exc()}"
            )
            Metrics.counter("llm.sample.error.count").add(1, attributes=attributes)
            raise
        finally:
            Metrics.histogram("llm.sample.duration").record(
                time.perf_counter() - start, attributes=attributes
            )

    async def sample(self, prompt: list[chat_pb2.Message], **kwargs) -> str:
        response = await self._sample_raw(prompt, **kwargs)
        return response.content

    def _extract_details(self, response) -> SampleDetails:
        trace = getattr(response, "reasoning_content", None) or None
        ws = WebSearchDetails()

        if hasattr(response, "tool_calls") and response.tool_calls:
            ws.used = True
            for tc in response.tool_calls:
                ws.tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        if hasattr(response, "tool_outputs") and response.tool_outputs:
            ws.used = True
            for to in response.tool_outputs:
                content = (
                    to.message.content
                    if hasattr(to.message, "content")
                    else str(to.message)
                )
                ws.tool_outputs.append(content)

        if hasattr(response, "citations") and response.citations:
            ws.used = True
            ws.citations = list(response.citations)

        if (
            hasattr(response, "server_side_tool_usage")
            and response.server_side_tool_usage
        ):
            ws.used = True
            ws.server_side_tool_usage = dict(response.server_side_tool_usage)

        return SampleDetails(trace=trace, web_search=ws)

    async def sample_with_details(
        self, prompt: list[chat_pb2.Message], **kwargs
    ) -> tuple[str, SampleDetails]:
        response = await self._sample_raw(prompt, **kwargs)
        return response.content, self._extract_details(response)
