import uuid

from grok_sampler.oai_sampler import OaiSampler

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import Conversation, Message, Role
from grox.core.lm.thread import ThreadRenderer
from grox.flows.reply_spam.classifier_reply_ranking import ReplyScorer
from grox.flows.reply_spam.constants import GEMMA_2
from grox.flows.reply_spam.prompts import reply_scoring_system_simple_prompt


class SimpleReplyScorer(ReplyScorer):
    def __init__(self):
        oai_config = grox_config.get_oai_model(GEMMA_2)
        self.oai_gemma4 = OaiSampler(oai_config)

    async def _to_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        system_prompt = reply_scoring_system_simple_prompt(100_000)
        convo.messages.append(Message(role=Role.SYSTEM, content=[system_prompt]))
        convo.messages.append(
            ThreadRenderer.render(post, role=Role.HUMAN, include_signals=True)
        )
        return convo

    async def _sample(self, convo: Conversation, post: Post) -> str:
        return await self.oai_gemma4.sample(
            convo.to_openai_messages(), conversation_id=convo.conversation_id
        )
