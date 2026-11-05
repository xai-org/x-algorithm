import logging
import random
import re
import asyncio
import traceback
import uuid
from datetime import datetime

from grox.core.lm.convo import (
    NO_THINKING_PROMPT,
    THINKING_CONTROL_END,
    THINKING_CONTROL_START,
    Role,
    Message,
    Conversation,
)
from grox.core.lm.post import PostRenderer
from grox.core.lm.thread import ThreadRenderer
from grox.core.lm.user import UserRenderer
from grox.config.config import ModelName, grox_config
from grox.flows.ptos.mode import SafetyPtosMode
from grok_sampler.config import GrokModelConfig, EapiModelConfig
from grox.flows.ptos.prompts import (
    adult_content_policy_prompt,
    age_gating_harmful_text_policy_prompt,
    child_safety_policy_prompt,
    hate_or_abuse_policy_prompt,
    illegal_and_regulated_behaviors_policy_prompt,
    safety_ptos_prompt,
    spam_policy_prompt,
    suicide_or_self_harm_policy_prompt,
    violent_media_policy_prompt,
    violent_speech_policy_prompt,
)
from grok_sampler.vision_sampler import VisionSampler
from grok_sampler.eapi_sampler import EapiSampler
from grok_sampler.oai_sampler import OaiSampler
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen
from monitor.metrics import Metrics
from grox.core.data_loaders.data_types import Post
from grox.flows.ptos.constants import GEMMA, HIGH_FAV_THRESHOLD
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
_EAPI_4_6_INTERNAL_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_rate_threshold=0.5,
    window_size=600.0,
    min_calls_in_window=10,
    recovery_timeout=600.0,
    half_open_max_calls=5,
    excluded_exceptions=(asyncio.CancelledError,),
)
_EAPI_4_5_X_ALGO_BREAKER_CONFIG = CircuitBreakerConfig(
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
_eapi_4_5_x_algo_breaker = CircuitBreaker(
    ModelName.EAPI_GROK_4_5_X_ALGO, _EAPI_4_5_X_ALGO_BREAKER_CONFIG
)
_eapi_4_6_internal_breaker = CircuitBreaker(
    ModelName.EAPI_GROK_4_6_INTERNAL, _EAPI_4_6_INTERNAL_BREAKER_CONFIG
)


class SafetyPtosCategoryClassifier:
    result_pattern = re.compile(r"(.*)<json(?!\w)[>}\]]?(.*)</json>", re.DOTALL)

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


def _policy_no_violation(reason: str) -> SafetyPolicy:
    return SafetyPolicy(policyType=SafetyPolicyType.NoViolation, reason=reason)


def _policy_with_type(
    policy_type: SafetyPolicyType, source: SafetyPolicy, reason: str
) -> SafetyPolicy:
    return SafetyPolicy(
        policyType=policy_type, confidenceScore=source.confidenceScore, reason=reason
    )


class SafetyPtosPolicyCrossValidator:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        eapi_4_5 = grox_config.get_eapi_model(ModelName.EAPI_GROK_4_5_X_ALGO)
        self.eapi_4_5_x_algo = EapiSampler(EapiModelConfig(**eapi_4_5.model_dump()))
        eapi_4_1 = grox_config.get_eapi_model(ModelName.EAPI_GROK_4_1_FAST_X_ALGO)
        self.eapi_4_1_x_algo = EapiSampler(EapiModelConfig(**eapi_4_1.model_dump()))

    def _parse_policy(self, raw: str) -> SafetyPolicy | None:
        match = self.result_pattern.search(raw)
        if not match:
            return None
        try:
            return SafetyPolicy.model_validate_json(match.group(2).strip())
        except Exception:
            return None

    def _build_policy_convo(
        self, post: Post, category: SafetyPolicyCategory, system_prompt: str
    ) -> Conversation:
        content = _strip_thinking_restrictions(system_prompt)
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(Message(role=Role.SYSTEM, content=[content]))
        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} for the specific safety policy violation category: {category.value}"
        )
        user_msg.content.append(
            f"\n\nProvide the requested JSON object for the specific safety policy type.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)
        convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def validate(
        self, category: SafetyPolicyCategory, post: Post, policy: SafetyPolicy | None
    ) -> SafetyPolicy | None:
        if policy is None or policy.policyType == SafetyPolicyType.NoViolation:
            return policy
        if category == SafetyPolicyCategory.ChildSafety:
            return await self._validate_child_safety(post, policy)
        if category == SafetyPolicyCategory.ViolentMedia:
            return await self._validate_violent_media(post, policy)
        if category == SafetyPolicyCategory.IllegalAndRegulatedBehaviors:
            return await self._validate_illegal_and_regulated_behaviors(post, policy)
        return policy

    async def _validate_child_safety(
        self, post: Post, policy: SafetyPolicy
    ) -> SafetyPolicy:
        metric = "safety_ptos.child_safety_cross_model_validate_with_grok_4_5"
        post_creation_time = (
            post.created_at if post.created_at else datetime.now()
        ).strftime("%Y-%m-%d")
        convo = self._build_policy_convo(
            post,
            SafetyPolicyCategory.ChildSafety,
            child_safety_policy_prompt(post_creation_time),
        )
        try:
            async with _eapi_4_5_x_algo_breaker.guard():
                raw = await self.eapi_4_5_x_algo.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
            confirm = self._parse_policy(raw)
            if confirm is None:
                logger.error(
                    f"child_safety cross_validate unparseable post={post.id} primary={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id} raw={raw[:500]!r}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "unparseable"})
                return _policy_no_violation("child_safety_grok_4_5_parse_error")
            if confirm.policyType == policy.policyType:
                logger.info(
                    f"child_safety cross_validate agreed post={post.id} policy={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "agreed"})
                return policy
            logger.info(
                f"child_safety cross_validate disagreed post={post.id} primary={policy.policyType.value} "
                f"confirm={confirm.policyType.value} conversation_id={convo.conversation_id}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "disagreed"})
            return _policy_no_violation("child_safety_grok_4_5_disagreed")
        except Exception:
            logger.error(
                f"child_safety cross_validate failed post={post.id} primary={policy.policyType.value} "
                f"conversation_id={convo.conversation_id}: {traceback.format_exc()}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "error"})
            return _policy_no_violation("child_safety_grok_4_5_sample_error")

    async def _validate_violent_media(
        self, post: Post, policy: SafetyPolicy
    ) -> SafetyPolicy:
        metric = "safety_ptos.violent_media_cross_model_validate_with_grok_4_5"
        convo = self._build_policy_convo(
            post, SafetyPolicyCategory.ViolentMedia, violent_media_policy_prompt()
        )
        try:
            async with _eapi_4_5_x_algo_breaker.guard():
                raw = await self.eapi_4_5_x_algo.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
            confirm = self._parse_policy(raw)
            if confirm is None:
                logger.error(
                    f"violent_media cross_validate unparseable post={post.id} primary={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id} raw={raw[:500]!r}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "unparseable"})
                return _policy_with_type(
                    SafetyPolicyType.ViolentMediaGraphicMedia,
                    policy,
                    f"violent_media_grok_4_5_parse_error: downgraded from {policy.policyType.value}",
                )
            if confirm.policyType == policy.policyType:
                logger.info(
                    f"violent_media cross_validate agreed post={post.id} policy={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "agreed"})
                return policy
            if confirm.policyType == SafetyPolicyType.NoViolation:
                logger.info(
                    f"violent_media cross_validate no_violation post={post.id} primary={policy.policyType.value} "
                    f"conversation_id={convo.conversation_id}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "no_violation"})
                return _policy_no_violation("violent_media_grok_4_5_no_violation")
            logger.info(
                f"violent_media cross_validate type_mismatch post={post.id} primary={policy.policyType.value} "
                f"confirm={confirm.policyType.value} conversation_id={convo.conversation_id}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "type_mismatch"})
            return _policy_with_type(
                SafetyPolicyType.ViolentMediaGraphicMedia,
                policy,
                f"violent_media_grok_4_5_type_mismatch: downgraded from {policy.policyType.value}, cv={confirm.policyType.value}",
            )
        except Exception:
            logger.error(
                f"violent_media cross_validate failed post={post.id} primary={policy.policyType.value} "
                f"conversation_id={convo.conversation_id}: {traceback.format_exc()}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "error"})
            return _policy_with_type(
                SafetyPolicyType.ViolentMediaGraphicMedia,
                policy,
                f"violent_media_grok_4_5_sample_error: downgraded from {policy.policyType.value}",
            )

    async def _validate_illegal_and_regulated_behaviors(
        self, post: Post, policy: SafetyPolicy
    ) -> SafetyPolicy:
        metric = "safety_ptos.illegal_and_regulated_behaviors_cross_model_validate_with_grok_4_1"
        convo = self._build_policy_convo(
            post,
            SafetyPolicyCategory.IllegalAndRegulatedBehaviors,
            illegal_and_regulated_behaviors_policy_prompt(),
        )
        try:
            raw = await self.eapi_4_1_x_algo.sample(
                convo.interleaveToEapi(), conversation_id=convo.conversation_id
            )
            confirm = self._parse_policy(raw)
            if confirm is None:
                logger.error(
                    f"illegal_and_regulated_behaviors cross_validate unparseable post={post.id} "
                    f"primary={policy.policyType.value} conversation_id={convo.conversation_id} raw={raw[:500]!r}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "unparseable"})
                return _policy_no_violation(
                    "illegal_and_regulated_behaviors_grok_4_1_parse_error"
                )
            if confirm.policyType == policy.policyType:
                logger.info(
                    f"illegal_and_regulated_behaviors cross_validate agreed post={post.id} "
                    f"policy={policy.policyType.value} conversation_id={convo.conversation_id}"
                )
                Metrics.counter(metric).add(1, attributes={"outcome": "agreed"})
                return _policy_with_type(
                    policy.policyType, policy, confirm.reason or policy.reason
                )
            logger.info(
                f"illegal_and_regulated_behaviors cross_validate disagreed post={post.id} "
                f"primary={policy.policyType.value} confirm={confirm.policyType.value} "
                f"conversation_id={convo.conversation_id}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "disagreed"})
            return _policy_no_violation(
                confirm.reason
                or f"illegal_and_regulated_behaviors_grok_4_1_disagreed: cv={confirm.policyType.value}"
            )
        except Exception:
            logger.error(
                f"illegal_and_regulated_behaviors cross_validate failed post={post.id} "
                f"primary={policy.policyType.value} conversation_id={convo.conversation_id}: {traceback.format_exc()}"
            )
            Metrics.counter(metric).add(1, attributes={"outcome": "error"})
            return _policy_no_violation(
                "illegal_and_regulated_behaviors_grok_4_1_sample_error"
            )


class SafetyPtosChildSafetyPolicyClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self, gemma_model_name: str = GEMMA):
        self.oai_gemma4 = OaiSampler(grox_config.get_oai_model(gemma_model_name))

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
            return _policy_no_violation("child_safety_gemma_sample_error")

        policy = self._parse_policy(raw)
        if policy is None:
            logger.error(
                f"child_safety policy gemma unparseable post={post.id} conversation_id={convo.conversation_id} raw={raw[:500]!r}"
            )
            Metrics.counter("safety_ptos.child_safety_policy.count").add(
                1, attributes={"outcome": "gemma_unparseable"}
            )
            return _policy_no_violation("child_safety_gemma_parse_error")
        return policy


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

        if self.deluxe:
            eapi_config_4_3_x_algo = grox_config.get_eapi_model(
                ModelName.EAPI_GROK_4_3_X_ALGO
            )
            self.eapi_4_3_x_algo = EapiSampler(
                EapiModelConfig(**eapi_config_4_3_x_algo.model_dump())
            )

            eapi_config_4_6_internal = grox_config.get_eapi_model(
                ModelName.EAPI_GROK_4_6_INTERNAL
            )
            self.eapi_4_6_internal = EapiSampler(
                EapiModelConfig(**eapi_config_4_6_internal.model_dump())
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
        elif violation.category == SafetyPolicyCategory.AgeGatingHarmfulText:
            return age_gating_harmful_text_policy_prompt()
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
            user_msg.content.extend(
                PostRenderer.render(
                    post,
                    include_reply_to=True,
                    include_media_tags=self._include_media_tags(violation.category),
                )
            )
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
        SafetyPolicyCategory.AgeGatingHarmfulText,
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
        SafetyPolicyCategory.IllegalAndRegulatedBehaviors,
        SafetyPolicyCategory.AgeGatingHarmfulText,
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

    @staticmethod
    def _include_media_tags(category: SafetyPolicyCategory) -> bool:
        return category == SafetyPolicyCategory.Spam

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
                mode = "deluxe-4.6-internal"
                result = await self._sample_4_6_internal(convo)
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

    async def _sample_4_6_internal(self, convo: Conversation) -> str:
        breaker, sampler = _eapi_4_6_internal_breaker, self.eapi_4_6_internal
        try:
            async with breaker.guard():
                return await sampler.sample(
                    convo.interleaveToEapi(), conversation_id=convo.conversation_id
                )
        except CircuitBreakerOpen as e:
            Metrics.counter("safety_ptos.eapi_4_6_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "breaker_open"}
            )
            logger.warning(
                f"4.6 circuit breaker '{e.name}' open (recovery in {e.remaining_seconds:.0f}s), falling back to 4.1"
            )
        except Exception:
            Metrics.counter("safety_ptos.eapi_4_6_fallback.count").add(
                1, attributes={"endpoint": breaker.name, "reason": "error"}
            )
            logger.error(
                f"Failed to call 4.6-internal reasoning, conversation_id={convo.conversation_id}, error: {traceback.format_exc()}"
            )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _sample(self, convo: Conversation, sample_for_gemma: bool = False) -> str:
        if sample_for_gemma:
            try:
                return await self.oai_gemma4.sample(
                    convo.to_openai_messages(), conversation_id=convo.conversation_id
                )
            except Exception:
                logger.error(
                    f"OaiSampler (gemma4) failed for policy, falling back to grok: {traceback.format_exc()}"
                )
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )


class SafetyPtosAdultContentCrossValidationJudge:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        eapi_cfg = grox_config.get_eapi_model(ModelName.EAPI_GROK_4_5_X_ALGO)
        self.llm = EapiSampler(EapiModelConfig(**eapi_cfg.model_dump()))

    def build_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(
                role=Role.SYSTEM,
                content=[_strip_thinking_restrictions(adult_content_policy_prompt())],
            )
        )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} for the specific safety policy violation category: {SafetyPolicyCategory.AdultContent.value}"
        )
        user_msg.content.append(
            f"\n\nProvide the requested JSON object for the specific safety policy type.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)
        convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def judge(self, post: Post) -> SafetyPolicy:
        convo = self.build_convo(post)
        async with _eapi_4_5_x_algo_breaker.guard():
            raw = await self.llm.sample(
                convo.interleaveToEapi(), conversation_id=convo.conversation_id
            )
        match = self.result_pattern.search(raw)
        if not match:
            raise ValueError(
                f"Invalid output for adult content cross validation judge: {raw[:500]!r}"
            )
        return SafetyPolicy.model_validate_json(match.group(2).strip())
