import asyncio
import base64
import logging

import httpx
import numpy as np
from aiolimiter import AsyncLimiter
from pydantic import BaseModel

logger = logging.getLogger(__name__)


IMAGE_PAD = "<|vision_start|><|image_pad|><|vision_end|>"


class ChatTemplate:
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def __init__(self, system_prompt: str = "Represent the user's input."):
        self.system_prompt = system_prompt

    def format(self, user_text: str) -> str:
        return f"{self.IM_START}system\n{self.system_prompt}{self.IM_END}\n{self.IM_START}user\n{user_text}{self.IM_END}\n{self.IM_START}assistant\n"

    def format_query(self, query: str) -> str:
        return self.format(query)

    def format_document(self, document: str) -> str:
        return self.format(document)

    def format_with_image_pads(self, user_text: str, num_images: int) -> str:
        content = ""
        for _ in range(num_images):
            content += IMAGE_PAD + "\n"
        content += user_text
        return f"{self.IM_START}system\n{self.system_prompt}{self.IM_END}\n{self.IM_START}user\n{content}{self.IM_END}\n{self.IM_START}assistant\n"

    def format_raw(self, user_text: str) -> str:
        return f"{self.IM_START}system\n{self.system_prompt}{self.IM_END}\n{self.IM_START}user\n{user_text}{self.IM_END}\n{self.IM_START}assistant\n"


class EmbeddingModelConfig(BaseModel):
    model_name: str
    endpoint: str
    text_max_len: int = 1024
    timeout_seconds: float | None = None


class XaiEmbeddingClientHttp:
    def __init__(
        self,
        http_server: str | None = None,
        max_concurrent_requests: int = 10000,
        max_qps: int = 10000,
        config: EmbeddingModelConfig | None = None,
        system_prompt: str | None = None,
        chat_template: ChatTemplate | None = None,
    ):
        if config:
            self.http_server = config.endpoint
            self.text_max_len = config.text_max_len
            self.timeout_seconds = config.timeout_seconds or 120.0
            self.model_name = config.model_name
        else:
            if http_server is None:
                raise ValueError("http_server must be provided if config is not")
            self.http_server = http_server
            self.text_max_len = 1024
            self.timeout_seconds = 120.0
            self.model_name = "embedding"

        if chat_template is not None:
            self.chat_template = chat_template
        else:
            self.chat_template = ChatTemplate(
                system_prompt or "Represent the user's input."
            )
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._rate_limiter = AsyncLimiter(max_qps)

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_concurrent_requests,
                max_keepalive_connections=max_concurrent_requests,
            ),
        )

        logger.info(
            f"XaiEmbeddingClientHttp initialized for {self.http_server} with {max_qps} QPS limit, timeout={self.timeout_seconds}s"
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def __aenter__(self) -> "XaiEmbeddingClientHttp":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def encode_async(
        self,
        text: str,
        images: list[bytes] | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        async with self._semaphore, self._rate_limiter:
            num_images = len(images) if images else 0
            prompt = self.chat_template.format_with_image_pads(text, num_images)

            return await self._send_encode_request(prompt, images, timeout)

    async def encode_with_embedded_pads_async(
        self,
        text_with_pads: str,
        images: list[bytes] | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        import time

        wait_start = time.perf_counter()
        async with self._semaphore, self._rate_limiter:
            wait_duration_ms = (time.perf_counter() - wait_start) * 1000

            prompt = self.chat_template.format_raw(text_with_pads)

            result = await self._send_encode_request(prompt, images, timeout)

            if wait_duration_ms > 100:
                logger.warning(
                    f"encode_with_embedded_pads_async: waited {wait_duration_ms:.1f}ms for semaphore/rate_limiter, "
                    f"images={len(images) if images else 0}"
                )

            return result

    @staticmethod
    def _image_data_uri(image: bytes) -> str:
        mime = "image/png" if image[:8].startswith(b"\x89PNG") else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(image).decode()}"

    async def embed_openai_async(
        self,
        text: str,
        image: bytes | None = None,
        images: list[bytes] | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        import time

        img_list: list[bytes] | None = (
            images if images else ([image] if image is not None else None)
        )

        wait_start = time.perf_counter()
        async with self._semaphore, self._rate_limiter:
            wait_duration_ms = (time.perf_counter() - wait_start) * 1000

            if img_list:
                uris = [self._image_data_uri(b) for b in img_list]
                api_input: dict = {
                    "text": text,
                    "image": uris if len(uris) > 1 else uris[0],
                }
            else:
                api_input = {"text": text}

            payload = {"input": [api_input], "model": self.model_name}

            base = (
                self.http_server
                if self.http_server.startswith("http")
                else f"http://{self.http_server}"
            )
            url = f"{base}/v1/embeddings"

            request_start = time.perf_counter()
            request_timeout = timeout or self.timeout_seconds
            response = await self._http_client.post(
                url, json=payload, timeout=httpx.Timeout(request_timeout)
            )
            response.raise_for_status()
            request_duration_ms = (time.perf_counter() - request_start) * 1000

            result = response.json()
            try:
                embedding = result["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"Cannot parse /v1/embeddings response: {result!r}"
                ) from exc

            n_img = len(img_list) if img_list else 0
            if wait_duration_ms > 100:
                logger.warning(
                    f"embed_openai_async: waited {wait_duration_ms:.1f}ms for semaphore/rate_limiter, n_img={n_img}"
                )
            logger.info(
                f"embed_openai_async: http={request_duration_ms:.1f}ms, n_img={n_img}, text_len={len(text)}, dim={len(embedding)}"
            )

            return np.array(embedding, dtype=np.float32)

    async def _send_encode_request(
        self,
        prompt: str,
        images: list[bytes] | None,
        timeout: float | None,
    ) -> np.ndarray:
        import time

        payload_start = time.perf_counter()
        payload: dict = {"text": prompt, "sampling_params": {"max_new_tokens": 1}}
        if images:
            payload["image_data"] = [
                f"data:image/png;base64,{base64.b64encode(img).decode()}"
                for img in images
            ]
        payload_duration_ms = (time.perf_counter() - payload_start) * 1000

        base = (
            self.http_server
            if self.http_server.startswith("http")
            else f"http://{self.http_server}"
        )
        url = f"{base}/encode"

        request_start = time.perf_counter()
        request_timeout = timeout or self.timeout_seconds
        response = await self._http_client.post(
            url, json=payload, timeout=httpx.Timeout(request_timeout)
        )
        response.raise_for_status()
        request_duration_ms = (time.perf_counter() - request_start) * 1000

        parse_start = time.perf_counter()
        result = response.json()

        if isinstance(result, dict) and "embedding" in result:
            embedding = list(result["embedding"])
        elif isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "embedding" in first:
                embedding = list(first["embedding"])
            else:
                embedding = list(first)
        elif isinstance(result, dict):
            for key in ("embeddings",):
                if key in result:
                    emb = result[key]
                    if isinstance(emb, list) and emb:
                        embedding = (
                            list(emb[0]) if isinstance(emb[0], list) else list(emb)
                        )
                        break
            else:
                raise ValueError(
                    f"Cannot parse /encode response: {type(result)}, keys={list(result.keys())}"
                )
        else:
            raise ValueError(f"Cannot parse /encode response: {type(result)}")

        parse_duration_ms = (time.perf_counter() - parse_start) * 1000

        total_duration_ms = (
            payload_duration_ms + request_duration_ms + parse_duration_ms
        )
        logger.info(
            f"_send_encode_request: total={total_duration_ms:.1f}ms "
            f"(payload={payload_duration_ms:.1f}ms, http={request_duration_ms:.1f}ms, parse={parse_duration_ms:.1f}ms), "
            f"images={len(images) if images else 0}, prompt_len={len(prompt)}"
        )

        return np.array(embedding, dtype=np.float32)


HttpEmbeddingClient = XaiEmbeddingClientHttp
