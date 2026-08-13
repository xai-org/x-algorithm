import logging

from protos.sampler.sampler_pb2 import SampleSettings, SampleTextRequest

from grok_sampler.llm import LiteLLM

logger = logging.getLogger(__name__)


class TextSampler(LiteLLM[str]):
    async def _get_sample_request(
        self, query: str, separator: str, **kwargs
    ) -> SampleTextRequest:
        conversation_id = kwargs.get("conversation_id")
        max_len = kwargs.get("max_len", self.model_config.max_resp_len)
        output_logits = kwargs.get("output_logits", False)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        nucleus_p = kwargs.get("nucleus_p", 0.95)
        temperature = kwargs.get("temperature", self.model_config.temperature)
        rng_seed = kwargs.get("rng_seed")
        priority = kwargs.get("priority", self.model_config.priority)
        return SampleTextRequest(
            prompt=query,
            output_probs=output_logits,
            return_tokens=output_logits,
            conversation_id=conversation_id,
            frequency_penalty=frequency_penalty,
            settings=SampleSettings(
                max_len=max_len,
                stop_strings=[separator],
                rng_seed=rng_seed,
                nucleus_p=nucleus_p,
                temperature=temperature,
            ),
            priority=priority,
        )
