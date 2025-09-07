from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"llm_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"]

    base_url: str
    api_key: str

    model_name: str = "gpt-4o"
    max_tokens: int = 16000
    timeout: float = 30.0

    entry_point: str = "/v1/chat/completions"
