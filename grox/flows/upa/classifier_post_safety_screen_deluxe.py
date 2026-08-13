import logging
import re
import uuid

from grok_sampler.config import GrokModelConfig
from grok_sampler.vision_sampler import VisionSampler

from grox.config.config import ModelName, grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import THINKING_CONTROL_START, Conversation, Message, Role
from grox.core.lm.post import PostRenderer
from grox.core.lm.user import UserRenderer
from grox.flows.upa.prompts import post_safety_deluxe_prompt
from grox.flows.upa.state_post_safety import PostSafetyScreenResult

logger = logging.getLogger(__name__)


class PostSafetyDeluxeClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        vlm_config = grox_config.get_model(ModelName.GROK_4_1_FAST_HEDGEHOG_CRITICAL)
        self.llm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))

    @property
    def model_name(self) -> str:
        return self.llm.model_config.model_name

    @staticmethod
    def build_convo(post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(role=Role.SYSTEM, content=[post_safety_deluxe_prompt()])
        )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)

        convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def classify(self, post: Post) -> PostSafetyScreenResult:
        convo = self.build_convo(post)
        output = await self._sample(convo)
        return self._parse(post, output)

    async def _sample(self, convo: Conversation) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    def _parse(self, post: Post, output: str) -> PostSafetyScreenResult:
        is_high_fav = (
            post.get_fav_count()
            >= grox_config.media_hydration.deluxe_fav_count_threshold
        )
        match = self.result_pattern.search(output)
        if not match:
            raise ValueError(f"Invalid output: {output}")

        reasoning = match.group(1).strip()
        logger.info(
            f"Post Safety Screen reasoning for post {post.id} is_high_fav {is_high_fav} : {reasoning}"
        )
        result = PostSafetyScreenResult.model_validate_json(match.group(2).strip())
        logger.info(
            f"Post Safety Screen result for post {post.id} is_high_fav {is_high_fav} : {result}"
        )
        return result
