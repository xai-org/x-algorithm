import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from grox.core.data_loaders.data_types import Post, User
from grox.core.schedules.types import ExtendableModel

logger = logging.getLogger(__name__)


class MessageQueuePayload(ExtendableModel):
    mid: str
    post: Post | None = None
    user: User | None = None
    embedding: list[float] | None = None


class MessageQueueLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    def poll(self) -> AsyncGenerator[MessageQueuePayload | None, None]:
        pass

    @abstractmethod
    async def ack(self, mid: str, success: bool = True):
        pass
