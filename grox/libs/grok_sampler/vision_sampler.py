import logging

from protos.sampler.sampler_pb2 import PromptInput, SampleSettings, SampleTextRequest

from grok_sampler.llm import LiteLLM

logger = logging.getLogger(__name__)


class VisionSampler(LiteLLM[list[str | bytes]]):
    async def _get_sample_request(
        self, query: list[str | bytes], separator: str, **kwargs
    ) -> SampleTextRequest:
        conversation_id = kwargs.get("conversation_id")
        max_resp_len = kwargs.get("max_resp_len", self.model_config.max_resp_len)
        output_logits = kwargs.get("output_logits", False)
        nucleus_p = kwargs.get("nucleus_p", 0.95)
        temperature = kwargs.get("temperature", self.model_config.temperature)
        rng_seed = kwargs.get("rng_seed")
        json_schema = kwargs.get("json_schema")
        structural_tag = kwargs.get("structural_tag")
        structural_pattern = kwargs.get("structural_pattern")
        structural_pattern_v2 = kwargs.get("structural_pattern_v2")
        ebnf = kwargs.get("ebnf")
        priority = kwargs.get("priority", self.model_config.priority)

        inputs: list[PromptInput] = []
        for message in query:
            if isinstance(message, bytes):
                inputs.append(PromptInput(image=message))
            elif isinstance(message, str):
                inputs.append(PromptInput(text=message))
            else:
                logger.error(f"unexpected message: {message}")
                raise ValueError(f"Invalid message type: {type(message)=}")

        return SampleTextRequest(
            inputs=inputs,
            output_probs=output_logits,
            return_tokens=output_logits,
            conversation_id=conversation_id,
            settings=SampleSettings(
                max_len=max_resp_len,
                stop_strings=[separator],
                rng_seed=rng_seed,
                nucleus_p=nucleus_p,
                temperature=temperature,
            ),
            json_schema=json_schema,
            structural_tag=structural_tag,
            structural_pattern=structural_pattern,
            structural_pattern_v2=structural_pattern_v2,
            ebnf=ebnf,
            priority=priority,
        )
