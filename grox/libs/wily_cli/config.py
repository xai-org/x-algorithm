import getpass

from pydantic import BaseModel, Field


class WilyConfig(BaseModel):
    zone: str = "atla"
    allow_alternative_zones: bool = Field(default=True)
    role: str = Field(default_factory=getpass.getuser)
    jitter: float = Field(default=0.25)
    client_name: str = Field(default="wily-cli")
    timeout: int = Field(default=30)
    verify: bool = Field(default=True)
