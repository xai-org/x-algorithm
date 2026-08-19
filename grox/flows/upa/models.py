
from pydantic import BaseModel


class ContentCategoryScore(BaseModel):
    id: int
    name: str
    score: float
    category_id: int | None = None


class TweetBoolMetadata(BaseModel):
    isHighQuality: bool | None = None
    isNsfw: bool | None = None
    isGore: bool | None = None
    isViolent: bool | None = None
    isSpam: bool | None = None
    isSoftNsfw: bool | None = None
    isAdult: bool | None = None
