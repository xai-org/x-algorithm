import logging
import re
import traceback
import uuid

from grok_sampler.oai_sampler import OaiSampler
from pydantic import BaseModel

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import (
    THINKING_CONTROL_END,
    THINKING_CONTROL_START,
    Conversation,
    Message,
    Role,
)
from grox.core.lm.post import PostRenderer
from grox.core.lm.user import UserRenderer
from grox.flows.upa.constants import GEMMA_UPA
from grox.flows.upa.models import ContentCategoryScore, TweetBoolMetadata
from grox.flows.upa.prompts import banger_mini_vlm_screen_score_gemma_prompt
from grox.flows.upa.state_initial_banger import BangerScreenResult

logger = logging.getLogger(__name__)


def topic_lookup_by_name(topics: list | None) -> dict[str, tuple[int, str]] | None:
    if not topics:
        return None
    lookup: dict[str, tuple[int, str]] = {}
    for category in topics:
        lookup[category.categoryName.strip().casefold()] = (
            category.categoryEntityId,
            category.categoryName,
        )
        for sub in category.subtopics:
            lookup[sub.topicName.strip().casefold()] = (
                sub.topicEntityId,
                sub.topicName,
            )
    return lookup


class BangerInitialScreenParsedOutput(BaseModel):
    description: str
    tags: list[str]
    taxonomy_categories: list[dict] | None = None
    tweet_bool_metadata: TweetBoolMetadata | None = None
    is_image_editable_by_grok: bool | None = None
    slop_score: int | None = None
    has_minor_score: float | None = None


class BangerInitialScreenGemmaClassifier:
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        oai_config = grox_config.get_oai_model(GEMMA_UPA)
        self.oai_gemma4 = OaiSampler(oai_config)

    @property
    def model_name(self) -> str:
        return self.oai_gemma4.model

    @staticmethod
    def build_convo(post: Post, topics: list | None = None) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(
                role=Role.SYSTEM,
                content=[banger_mini_vlm_screen_score_gemma_prompt(topics)],
            )
        )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post.{THINKING_CONTROL_START}"
        )
        convo.messages.append(user_msg)

        convo.messages.append(
            Message(role=Role.ASSISTANT, content=[THINKING_CONTROL_END], separator="")
        )
        return convo

    async def classify(
        self, post: Post, topics: list | None = None
    ) -> BangerScreenResult:
        try:
            convo = self.build_convo(post, topics=topics)
            output = await self._sample(convo)
            return self._parse(post, output, topics)
        except Exception:
            logger.error(
                f"[{self.__class__.__name__}] error processing content classify request: {post.id} {traceback.format_exc()}"
            )
            raise

    async def _sample(self, convo: Conversation) -> str:
        return await self.oai_gemma4.sample(
            convo.to_openai_messages(), conversation_id=convo.conversation_id
        )

    def _parse(
        self, post: Post, output: str, topics: list | None
    ) -> BangerScreenResult:
        match = self.result_pattern.search(output)
        if not match:
            raise ValueError(f"Invalid output: {output}")

        reasoning = match.group(1).strip()
        logger.info(
            f"Banger initial screen result reasoning for post {post.id}: {reasoning}"
        )
        parsed = BangerInitialScreenParsedOutput.model_validate_json(
            match.group(2).strip()
        )
        logger.info(f"Banger initial screen result for post {post.id}: {parsed}")
        taxonomy_categories = []
        if parsed.taxonomy_categories:
            lookup = topic_lookup_by_name(topics)
            if lookup is None:
                logger.warning(
                    f"No input topics to resolve taxonomy names for post {post.id}; dropping {len(parsed.taxonomy_categories)} entries"
                )
            else:
                for tc in parsed.taxonomy_categories:
                    tc_name = str(tc["name"]).strip()
                    resolved = lookup.get(tc_name.casefold())
                    if resolved is None:
                        raise ValueError(
                            f"Taxonomy name not in input topics for post {post.id}: name={tc_name!r}"
                        )
                    entity_id, canonical_name = resolved
                    taxonomy_categories.append(
                        ContentCategoryScore(
                            id=entity_id,
                            name=canonical_name,
                            score=tc["score"],
                            category_id=None,
                        )
                    )

        return BangerScreenResult(
            summary=parsed.description,
            tags=parsed.tags,
            taxonomy_categories=taxonomy_categories,
            tweet_bool_metadata=parsed.tweet_bool_metadata,
            is_image_editable_by_grok=parsed.is_image_editable_by_grok,
            slop_score=parsed.slop_score,
            has_minor_score=parsed.has_minor_score,
        )
