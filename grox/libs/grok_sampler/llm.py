import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Generic, TypeVar

from grpc_cli.grpc_client import GrpcClient, aclosing_grpc
from monitor.logging import Logging
from monitor.metrics import Metrics
from protos.sampler.sampler_pb2 import (
    PromptInput,
    SampleTextRequest,
    SampleTextResponse,
)
from protos.sampler.sampler_pb2_grpc import SamplerStub

from grok_sampler.config import GrokModelConfig

T = TypeVar("T")
logger = logging.getLogger(__name__)
SEPARATOR = "<|separator|>"
PROMPT_LENGTH_BUCKETS = (
    10.0,
    50.0,
    100.0,
    500.0,
    1000.0,
    5000.0,
    10000.0,
    20000.0,
    50000.0,
    100000.0,
    150000.0,
    float("inf"),
)


class LiteLLM(ABC, Generic[T]):
    def __init__(self, model_config: GrokModelConfig):
        self.model_config = model_config
        self.request_metadata = [
            ("x-model-version", model_config.model_name),
        ]

        if model_config.bearer_token:
            self.request_metadata.append(
                ("authorization", f"Bearer {model_config.bearer_token}")
            )

        if model_config.priority_header:
            self.request_metadata.append(("x-jwt-prio", model_config.priority_header))

    def _get_client(self) -> SamplerStub:
        return SamplerStub(
            GrpcClient(
                ca_root_dir="/etc/certs/sampler-proxy-root-cas",
                default_ca_name="prod.internal.x.ai",
            ).make(self.model_config.endpoint)
        )

    @abstractmethod
    async def _get_sample_request(
        self, query: T, separator: str, **kwargs
    ) -> SampleTextRequest:
        raise NotImplementedError()

    async def _sample_streaming(self, query: T, **kwargs) -> AsyncGenerator[str, None]:
        keep_separator = kwargs.get("keep_separator", False)
        separator = kwargs.get("separator", SEPARATOR)
        log_prompt = kwargs.get("log_prompt", False)
        request = await self._get_sample_request(query, separator, **kwargs)
        prompt_len = self._prompt_len(request)
        logger.info(f"Started sampling request, prompt_len={prompt_len}")
        if log_prompt:
            self._log_request(request)
        attributes = {"model_name": self.model_config.model_name}
        Metrics.counter("llm.sample.count").add(1, attributes=attributes)
        Metrics.histogram(
            "llm.sample.prompt_len",
            explicit_bucket_boundaries_advisory=PROMPT_LENGTH_BUCKETS,
        ).record(prompt_len, attributes=attributes)
        resp = ""
        tokens_received = 0
        start = time.perf_counter()
        try:
            async for tok in self._sample_streaming_raw(request, **kwargs):
                if tokens_received == 0:
                    ttft = time.perf_counter() - start
                    Metrics.histogram("llm.sample.ttft").record(
                        ttft, attributes=attributes
                    )
                    logger.info(f"Time to first token: {ttft:.3f}")
                tokens_received += 1
                Metrics.counter("llm.sample.token.count").add(1, attributes=attributes)
                if not keep_separator:
                    tok.text = tok.text.replace(separator, "")
                resp += tok.text
                yield tok.text
            logger.info(
                f"Finished sampling request in {time.perf_counter() - start:.2f} seconds"
            )
            Metrics.counter("llm.sample.success.count").add(1, attributes=attributes)
        except Exception:
            logger.error(f"Error in sampling request: {traceback.format_exc()}")
            Metrics.counter("llm.sample.error.count").add(1, attributes=attributes)
            raise
        finally:
            Metrics.histogram("llm.sample.tokens").record(
                tokens_received, attributes=attributes
            )
            Metrics.histogram("llm.sample.duration").record(
                time.perf_counter() - start, attributes=attributes
            )

    def _log_request(self, request: SampleTextRequest):
        def _format_request(request: SampleTextRequest) -> str:
            if request.prompt:
                return f"prompt: {request.prompt}"
            if request.inputs:
                msg = "\n".join(
                    [
                        input.text
                        if input.text
                        else f"[[[[image_bytes({len(input.image)})]]]]"
                        for input in request.inputs
                    ]
                )
                return f"inputs: {msg}"
            return ""

        logger.info(f"{_format_request(request)}")

    async def _sample_streaming_raw(
        self, request: SampleTextRequest, **kwargs
    ) -> AsyncGenerator[SampleTextResponse, None]:
        request_metadata = self.request_metadata.copy()
        conversation_id = kwargs.get("conversation_id")
        if conversation_id:
            request_metadata.append(("x-conversation-id", conversation_id))
        async with aclosing_grpc(
            self._get_client().SampleText(
                request, metadata=request_metadata, timeout=self.model_config.timeout
            )
        ) as gen:
            async for elem in gen:
                yield elem

    def _prompt_len(self, req: SampleTextRequest) -> int:
        def _len(input: PromptInput) -> int:
            if input.text:
                return len(input.text)
            if input.image:
                return len(input.image)
            return 0

        if req.prompt:
            return len(req.prompt)
        if req.inputs:
            return sum(_len(input) for input in req.inputs)
        return 0

    async def sample(self, query: T, **kwargs) -> str:
        try:
            res = ""
            tokens_received = 0
            with Metrics.tracer("llm").start_as_current_span("sample"):
                Logging.set_context(model_name=self.model_config.model_name)
                async for tok in self._sample_streaming(query, **kwargs):
                    res += tok
                    tokens_received += 1
            logger.info(f"Sample completed: {tokens_received} tokens, {len(res)} chars")
            return res
        except Exception as e:
            logger.error(
                f"Error in sampling request after {tokens_received} tokens: {traceback.format_exc()} {e}"
            )
            raise

    async def sample_streaming(self, query: T, **kwargs) -> AsyncGenerator[str, None]:
        res = ""
        try:
            with Metrics.tracer("llm").start_as_current_span("sample_streaming"):
                Logging.set_context(model_name=self.model_config.model_name)
                async with aclosing(self._sample_streaming(query, **kwargs)) as toks:
                    async for tok in toks:
                        yield tok
                        res += tok
        except Exception:
            logger.error(
                f"Error in streaming sampling request: {traceback.format_exc()}"
            )
            raise
