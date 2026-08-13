import asyncio
import logging
import random
import re
import traceback
import uuid
from datetime import datetime

from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen
from grok_sampler.config import EapiModelConfig, GrokModelConfig
from grok_sampler.eapi_sampler import EapiSampler
from grok_sampler.oai_sampler import OaiSampler
from grok_sampler.vision_sampler import VisionSampler
from monitor.metrics import Metrics

from grox.config.config import ModelName, grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import (
    NO_THINKING_PROMPT,
    THINKING_CONTROL_END,
    THINKING_CONTROL_START,
    Conversation,
    Message,
    Role,
)
from grox.core.lm.post import PostRenderer
from grox.core.lm.thread import ThreadRenderer
from grox.core.lm.user import UserRenderer
from grox.flows.ptos.constants import GEMMA, HIGH_FAV_THRESHOLD
from grox.flows.ptos.mode import SafetyPtosMode
from grox.flows.ptos.prompts import (
    adult_content_policy_prompt,
    child_safety_policy_prompt,
    hate_or_abuse_policy_prompt,
    illegal_and_regulated_behaviors_policy_prompt,
    safety_ptos_prompt,
    spam_policy_prompt,
    suicide_or_self_harm_policy_prompt,
    violent_media_policy_prompt,
    violent_speech_policy_prompt,
)
from grox.flows.ptos.state import (
    SafetyPolicy,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPostAnnotations,
    SafetyPtosViolatedPolicy,
)

logger = logging.getLogger(__name__)

_THINKING_RESTRICTION_LINES = {
    line.strip() for line in NO_THINKING_PROMPT.splitlines() if line.strip()
}


def _strip_thinking_restrictions(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(
        line for line in lines if line.strip() not in _THINKING_RESTRICTION_LINES
    ).lstrip("\n")


def _fav_bucket(fav_count: int) -> str:
    if fav_count <= 0:
        return "0"
    return str(2 ** (fav_count.bit_length() - 1))


_EAPI_4_3_X_ALGO_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_rate_threshold=0.5,
    window_size=120.0,
    min_calls_in_window=10,
    recovery_timeout=600.0,
    half_open_max_calls=5,
    excluded_exceptions=(asyncio.CancelledError,),
)
_EAPI_4_5_INTERNAL_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_rate_threshold=0.5,
    window_size=600.0,
    min_calls_in_window=10,
    recovery_timeout=600.0,
    half_open_max_calls=5,
    excluded_exceptions=(asyncio.CancelledError,),
)
_eapi_4_3_x_algo_breaker = CircuitBreaker(
    ModelName.EAPI_GROK_4_3_X_ALGO, _EAPI_4_3_X_ALGO_BREAKER_CONFIG
)
_eapi_4_5_internal_breaker = CircuitBreaker(
    ModelName.EAPI_GROK_4_5_INTERNAL, _EAPI_4_5_INTERNAL_BREAKER_CONFIG
)


class SafetyPtosCategoryClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(
        self,
        model_name: str,
        mode: SafetyPtosMode,
        use_gemma: bool = False,
    ):
        self.mode = mode
        self.deluxe = mode.is_deluxe
        self.use_gemma = use_gemma
        vlm_config = grox_config.get_model(model_name)
        self.llm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))

        oai_config = grox_config.get_oai_model(GEMMA)
        self.oai_gemma4 = OaiSampler(oai_config)
        self.use_oai_gemma4_dial = 1.0

    def build_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        content = safety_ptos_prompt()

        if self.deluxe:
            content = _strip_thinking_restrictions(content)
        convo.messages.append(Message(role=Role.SYSTEM, content=[content]))

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)

        if self.deluxe:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        else:
            convo.messages.append(
                Message(
                    role=Role.ASSISTANT, content=[THINKING_CONTROL_END], separator=""
                )
            )

        return convo

    async def classify_post(self, post: Post) -> SafetyPostAnnotations:
        convo = await self._to_convo(post)
        result = await self._sample(convo)
        mode = self.mode.value
        logger.info(
            f"safety ptos category classifier ({mode}) result for {post.id}: {result}"
        )

        match = self.result_pattern.search(result)
        if match:
            json_str = match.group(2).strip()
            return SafetyPostAnnotations.model_validate_json(json_str)
        else:
            raise ValueError(
                f"Invalid output for safety ptos category classifier ({mode}): {result}"
            )

    async def _to_convo(self, post: Post) -> Conversation:
        return self.build_convo(post)

    async def _sample(self, convo: Conversation) -> str:
        if self.use_gemma and random.random() < self.use_oai_gemma4_dial:
            try:
                return await self.oai_gemma4.sample(
                    convo.to_openai_messages(), conversation_id=convo.conversation_id
                )
            except Exception:
                logger.error(
                    f"OaiSampler (gemma4) failed, falling back to grok: {traceback.format_exc()}"
                )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )


class SafetyPtosChildSafetyPolicyClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self, gemma_model_name: str = GEMMA):
        self.oai_gemma4 = OaiSampler(grox_config.get_oai_model(gemma_model_name))
        eapi_cfg = grox_config.get_eapi_model(ModelName.EAPI_GROK_4_5_INTERNAL)
        self.eapi_4_5_internal = EapiSampler(EapiModelConfig(**eapi_cfg.model_dump()))

    def build_convo(self, post: Post) -> Conversation:
        post_creation_time = (
            post.created_at if post.created_at else datetime.now()
        ).strftime("%Y-%m-%d")
        content = _strip_thinking_restrictions(
            child_safety_policy_prompt(post_creation_time)
        )

        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(Message(role=Role.SYSTEM, content=[content]))

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} for the specific safety policy violation category: {SafetyPolicyCategory.ChildSafety.value}"
        )
        user_msg.content.append(
            f"\n\nProvide the requested JSON object for the specific safety policy type.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)
        convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    @staticmethod
    def _no_violation(reason: str) -> SafetyPolicy:
        return SafetyPolicy(policyType=SafetyPolicyType.NoViolation, reason=reason)

    def _parse_policy(self, raw: str) -> SafetyPolicy | None:
        match = self.result_pattern.search(raw)
        if not match:
            return None
        try:
            return SafetyPolicy.model_validate_json(match.group(2).strip())
        except Exception:
            return None

    async def classify_policy(self, post: Post) -> SafetyPolicy:
        convo = self.build_convo(post)
        try:
            raw = await self.oai_gemma4.sample(
                convo.to_openai_messages(), conversation_id=convo.conversation_id
            )
        except Exception:
            logger.error(
                f"child_safety policy gemma sample failed post={post.id} conversation_id={convo.conversation_id}: {traceback.format_exc()}"
            )
            Metrics.counter("safety_ptos.child_safety_policy.count").add(
                1, attributes={"outcome": "gemma_error"}
            )
            return self._no_violation("child_safety_gemma_sample_error")

        policy = self._parse_policy(raw)
        if policy is None:
            logger.error(
                f"child_safety policy gemma unparseable post={post.id} conversation_id={convo.conversation_id} raw={raw[:500]!r}"
            )
            Metrics.counter("safety_ptos.child_safety_policy.count").add(
                1, attributes={"outcome": "gemma_unparseable"}
            )
            return self._no_violation("child_safety_gemma_parse_error")

        if policy.policyType == SafetyPolicyType.NoViolation:
            return policy
        return await self._cross_model_validate_with_4_5(convo, policy, post_id=post.id)

    async def _cross_model_validate_with_4_5(
        self, convo: Conversation, policy: SafetyPolicy, post_id: str
    ) -> SafetyPolicy:
        metric = "safety_ptos.cross_model_validate_with_grok_4_5"
        try:
            async with _eapi_4_5_internal_breaker.guard():
                raw = await self.eapi_4_5_internal.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
            confirm = self._parse_policy(raw)
            if confirm is None:
                logger.error(
                    f"child_safety cross_validate_4_5 unparseable post={post_id} primary={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id} raw={raw[:500]!r}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "unparseable"})
                return self._no_violation("child_safety_grok_4_5_parse_error")
            if confirm.policyType == policy.policyType:
                logger.info(
                    f"child_safety cross_validate_4_5 agreed post={post_id} policy={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "agreed"})
                return policy
            logger.info(
                f"child_safety cross_validate_4_5 disagreed post={post_id} primary={policy.policyType.value} "
                f"confirm={confirm.policyType.value} conversation_id={convo.conversation_id}, clearing to NoViolation"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "disagreed"})
            return self._no_violation("child_safety_grok_4_5_disagreed")
        except Exception:
            logger.error(
                f"child_safety cross_validate_4_5 failed post={post_id} primary={policy.policyType.value} "
                f"conversation_id={convo.conversation_id}: {traceback.format_exc()}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "error"})
            return self._no_violation("child_safety_grok_4_5_sample_error")


class SafetyPtosPolicyClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(
        self,
        model_name: str,
        mode: SafetyPtosMode,
        use_gemma: bool = True,
        gemma_model_name: str = GEMMA,
    ):
        self.mode = mode
        self.deluxe = mode.is_deluxe
        self.use_gemma = use_gemma

        vlm_config = grox_config.get_model(model_name)
        self.llm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))

        oai_config = grox_config.get_oai_model(gemma_model_name)
        self.oai_gemma4 = OaiSampler(oai_config)
        self.use_oai_gemma4_dial = 1.0

        if self.deluxe:
            eapi_config_4_3_x_algo = grox_config.get_eapi_model(
                ModelName.EAPI_GROK_4_3_X_ALGO
            )
            self.eapi_4_3_x_algo = EapiSampler(
                EapiModelConfig(**eapi_config_4_3_x_algo.model_dump())
            )

            eapi_config_4_5_internal = grox_config.get_eapi_model(
                ModelName.EAPI_GROK_4_5_INTERNAL
            )
            self.eapi_4_5_internal = EapiSampler(
                EapiModelConfig(**eapi_config_4_5_internal.model_dump())
            )

    @staticmethod
    def _get_policy_prompt(violation: SafetyPtosViolatedPolicy, post: Post) -> str:
        if violation.category == SafetyPolicyCategory.ViolentMedia:
            return violent_media_policy_prompt()
        elif violation.category == SafetyPolicyCategory.AdultContent:
            return adult_content_policy_prompt()
        elif violation.category == SafetyPolicyCategory.Spam:
            return spam_policy_prompt()
        elif violation.category == SafetyPolicyCategory.IllegalAndRegulatedBehaviors:
            return illegal_and_regulated_behaviors_policy_prompt()
        elif violation.category == SafetyPolicyCategory.HateOrAbuse:
            return hate_or_abuse_policy_prompt()
        elif violation.category == SafetyPolicyCategory.ViolentSpeech:
            return violent_speech_policy_prompt()
        elif violation.category == SafetyPolicyCategory.SuicideOrSelfHarm:
            return suicide_or_self_harm_policy_prompt()
        else:
            raise ValueError(
                f"No policy prompt available for category: {violation.category.value}"
            )

    def build_convo(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> Conversation:
        content = self._get_policy_prompt(violation, post)
        use_reasoning = (
            self.deluxe or violation.category in self.USE_REASONING_CATEGORIES
        )
        if use_reasoning:
            content = _strip_thinking_restrictions(content)

        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(Message(role=Role.SYSTEM, content=[content]))

        if violation.category in self.USE_THREAD_RENDERER_CATEGORIES and post.ancestors:
            user_msg = ThreadRenderer.render(post, role=Role.USER, separator="\n\n")
        else:
            user_msg = Message(role=Role.USER, content=[])
            if self._include_user_profile(violation.category):
                user_msg.content.extend(UserRenderer.render(post.user))
            user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} for the specific safety policy violation category: {violation.category.value}"
        )
        user_msg.content.append(
            f"\n\nProvide the requested JSON object for the specific safety policy type.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)

        if use_reasoning:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        else:
            convo.messages.append(
                Message(
                    role=Role.ASSISTANT, content=[THINKING_CONTROL_END], separator=""
                )
            )

        return convo

    SUPPORTED_CATEGORIES = {
        SafetyPolicyCategory.ViolentMedia,
        SafetyPolicyCategory.AdultContent,
        SafetyPolicyCategory.Spam,
        SafetyPolicyCategory.IllegalAndRegulatedBehaviors,
        SafetyPolicyCategory.HateOrAbuse,
        SafetyPolicyCategory.ViolentSpeech,
        SafetyPolicyCategory.SuicideOrSelfHarm,
    }

    DELUXE_4_3_CATEGORIES = {
        SafetyPolicyCategory.AdultContent,
        SafetyPolicyCategory.ViolentMedia,
    }

    USE_REASONING_CATEGORIES = {
        SafetyPolicyCategory.ViolentSpeech,
        SafetyPolicyCategory.SuicideOrSelfHarm,
    }

    USE_GEMMA_CATEGORIES = {
        SafetyPolicyCategory.Spam,
    }

    USE_THREAD_RENDERER_CATEGORIES = {
        SafetyPolicyCategory.ViolentSpeech,
        SafetyPolicyCategory.SuicideOrSelfHarm,
    }

    def _include_user_profile(self, category: SafetyPolicyCategory) -> bool:
        if (
            self.mode == SafetyPtosMode.LIVE_CLUSTER_ANCHORS
            and category == SafetyPolicyCategory.Spam
        ):
            return False
        return True

    async def classify_policy_for_violation(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> SafetyPolicy | None:
        if violation.category not in self.SUPPORTED_CATEGORIES:
            return None

        fav_count = post.get_fav_count() if post is not None else 0
        convo = await self._to_convo(post, violation)

        fav_bucket = _fav_bucket(fav_count)
        if (
            fav_count >= HIGH_FAV_THRESHOLD
            and self.deluxe
            and violation.category in self.DELUXE_4_3_CATEGORIES
        ):
            if fav_count >= 1024:
                mode = "deluxe-4.5-internal"
                result = await self._sample_4_5_internal(convo)
            else:
                mode = "deluxe-4.3"
                result = await self._sample_4_3(convo)
        else:
            mode = self.mode.value
            sample_for_gemma = violation.category in self.USE_GEMMA_CATEGORIES
            result = await self._sample(convo, sample_for_gemma=sample_for_gemma)
        Metrics.counter("safety_ptos.policy_requests.count").add(
            1, attributes={"mode": mode, "fav_bucket": fav_bucket}
        )

        logger.info(
            f"safety ptos policy classifier ({mode}) result for post {post.id}, violation {violation.category}: {result}"
        )

        match = self.result_pattern.search(result)
        if match:
            json_str = match.group(2).strip()
            policy = SafetyPolicy.model_validate_json(json_str)
            Metrics.counter("safety_ptos.policy_types.count").add(
                1,
                attributes={
                    "mode": mode,
                    "fav_bucket": fav_bucket,
                    "policy_type": policy.policyType.name,
                },
            )
            return policy
        else:
            raise ValueError(
                f"Invalid output for safety ptos policy ({mode}): {result}"
            )

    async def _to_convo(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> Conversation:
        return self.build_convo(post, violation)

    async def _sample_4_3(self, convo: Conversation) -> str:
        breaker, sampler = _eapi_4_3_x_algo_breaker, self.eapi_4_3_x_algo
        try:
            async with breaker.guard():
                return await sampler.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
        except CircuitBreakerOpen as e:
            Metrics.counter("safety_ptos.eapi_4_3_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "breaker_open"}
            )
            logger.warning(
                f"4.3 circuit breaker '{e.name}' open (recovery in {e.remaining_seconds:.0f}s), falling back to 4.1"
            )
        except Exception:
            Metrics.counter("safety_ptos.eapi_4_3_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "error"}
            )
            logger.error(
                f"Failed to call 4.3 reasoning, conversation_id={convo.conversation_id}, error: {traceback.format_exc()}"
            )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _sample_4_5_internal(self, convo: Conversation) -> str:
        breaker, sampler = _eapi_4_5_internal_breaker, self.eapi_4_5_internal
        try:
            async with breaker.guard():
                return await sampler.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
        except CircuitBreakerOpen as e:
            Metrics.counter("safety_ptos.eapi_4_5_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "breaker_open"}
            )
            logger.warning(
                f"4.5 circuit breaker '{e.name}' open (recovery in {e.remaining_seconds:.0f}s), falling back to 4.1"
            )
        except Exception:
            Metrics.counter("safety_ptos.eapi_4_5_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "error"}
            )
            logger.error(
                f"Failed to call 4.5-internal reasoning, conversation_id={convo.conversation_id}, error: {traceback.format_exc()}"
            )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _sample(self, convo: Conversation, sample_for_gemma: bool = False) -> str:
        if (
            sample_for_gemma
            and not self.deluxe
            and self.use_gemma
            and random.random() < self.use_oai_gemma4_dial
        ):
            try:
                return await self.oai_gemma4.sample(
                    convo.to_openai_messages(), conversation_id=convo.conversation_id
                )
            except Exception:
                logger.error(
                    f"OaiSampler (gemma4) failed for spam policy, falling back to grok: {traceback.format_exc()}"
                )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )
