from pydantic import BaseModel


class GrokModelConfig(BaseModel):
    model_name: str
    endpoint: str
    timeout: int
    context_length: int = 131072
    max_resp_len: int = 8092
    temperature: float = 0.7
    bearer_token: str = ""
    priority_header: str = ""
    priority: int = 15


class OaiModelConfig(BaseModel):
    model: str
    api_base_url: str
    timeout: int = 300
    temperature: float = 0.0001
    max_tokens: int = 8092


class EapiModelConfig(BaseModel):
    api_key: str | None = None
    model: str
    timeout: int = 300
    temperature: float = 0.0001
    priority: int = 15
    max_resp_len: int = 8092
    api_host: str = ""
    log_reasoning_trace: bool = False
    enable_search: bool = False
    open_search: bool = False
    x_search: bool = False
    reasoning_effort: str | None = None
