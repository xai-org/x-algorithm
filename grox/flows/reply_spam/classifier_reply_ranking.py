import json
import logging
import re
import uuid

import json_repair
from grok_sampler.config import GrokModelConfig
from grok_sampler.vision_sampler import VisionSampler
from monitor.metrics import Metrics
from pydantic import ValidationError

from grox.config.config import ModelName, grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import (
    NO_THINKING_PROMPT,
    SEPARATOR,
    THINKING_CONTROL_END,
    THINKING_CONTROL_START,
    Conversation,
    Message,
    Role,
)
from grox.core.lm.thread import ThreadRenderer
from grox.flows.reply_spam.prompts import reply_scoring_system_prompt
from grox.flows.reply_spam.state_reply_ranking import ReplyScoreResult

logger = logging.getLogger(__name__)


class ReplyScorer:
    model_name = "GROK"

    def __init__(self):
        vlm_config = grox_config.get_model(ModelName.GROK_4_MINI_CRITICAL)
        self.vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))

        vlm_fallback_config = grox_config.get_model(
            ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL
        )
        self.vlm_fallback = VisionSampler(
            GrokModelConfig(**vlm_fallback_config.model_dump())
        )

    async def score(self, post: Post) -> ReplyScoreResult:
        convo = await self._to_convo(post)
        result = await self._sample(convo, post)
        parsed = await self._parse(result)
        Metrics.histogram(
            "ranked_replies_scores",
            explicit_bucket_boundaries_advisory=[0.0, 1.0, 2.0, 3.0],
        ).record(parsed.score)

        return parsed

    async def _to_convo(self, post: Post, non_reasoning: bool = False) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        system_prompt = reply_scoring_system_prompt(100_000)
        if non_reasoning:
            system_prompt = NO_THINKING_PROMPT + system_prompt
        convo.messages.append(Message(role=Role.SYSTEM, content=[system_prompt]))
        convo.messages.append(
            ThreadRenderer.render(
                post,
                role=Role.HUMAN,
                separator=THINKING_CONTROL_START + SEPARATOR,
                include_signals=True,
            )
        )
        if non_reasoning:
            convo.messages.append(
                Message(
                    role=Role.ASSISTANT, content=[THINKING_CONTROL_END], separator=""
                )
            )
        else:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def _sample(self, convo: Conversation, post: Post) -> str:
        output = await self.vlm.sample(
            convo.interleave(),
            conversation_id=convo.conversation_id,
            json_schema=json.dumps(ReplyScoreResult.model_json_schema()),
        )
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not (match and "score" in match.group(0)):
            fallback_convo = await self._to_convo(post, non_reasoning=True)
            output = await self.vlm_fallback.sample(
                fallback_convo.interleave(),
                conversation_id=fallback_convo.conversation_id,
                json_schema=json.dumps(ReplyScoreResult.model_json_schema()),
            )
        return output

    async def _clean_output(self, output: str) -> str:
        if output.endswith("<|eos|>"):
            output = output.removesuffix("<|eos|>")
        output = output.strip()
        if output.startswith("```json"):
            output = output[7:]
        elif output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        output = output.strip()
        return output

    async def _parse(self, output: str) -> ReplyScoreResult:
        score = None
        reason = ""

        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match and "score" in match.group(0):
            raw_result = match.group(0).strip()
        else:
            raw_result = output

        cleaned_result = await self._clean_output(raw_result)

        try:
            result = ReplyScoreResult.model_validate_json(cleaned_result)
            score = result.score
            reason = result.reason
        except (ValidationError, ValueError):
            try:
                repaired = json_repair.repair_json(cleaned_result, return_objects=True)
                if isinstance(repaired, dict) and "score" in repaired:
                    result = ReplyScoreResult.model_validate(repaired)
                    score = result.score
                    reason = result.reason
                    Metrics.counter("task.reply_ranker.json_repaired.count").add(1)
            except Exception:
                pass

            if score is None:
                score_match = re.search(r'"score":\s*([\d.]+)', cleaned_result)
                if score_match:
                    try:
                        score = float(score_match.group(1).strip())
                    except ValueError:
                        score = None
                        Metrics.counter("task.reply_ranker.invalid.count").add(
                            1,
                            attributes={
                                "filter": "reply_ranking",
                                "reason": "invalid_score_format",
                            },
                        )
            if not reason:
                reason_match = re.search(
                    r'"reason":\s*"((?:[^"\\]|\\.)*)"', cleaned_result, re.DOTALL
                )
                if reason_match:
                    reason = reason_match.group(1)

        if not score and score != 0:
            logger.error(f"Invalid output format: {output}")
            Metrics.counter("task.reply_ranker.invalid.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "invalid_format"}
            )
            raise ValueError(f"Invalid output: {output}")

        if not (0 <= score <= 3):
            logger.error(f"Score {score} outside the 0-3 rubric: {output}")
            Metrics.counter("task.reply_ranker.invalid.count").add(
                1,
                attributes={"filter": "reply_ranking", "reason": "score_out_of_range"},
            )
            raise ValueError(f"Score {score} outside the 0-3 rubric")

        return ReplyScoreResult(score=score, reason=reason)
