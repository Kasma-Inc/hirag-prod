from typing import Literal, Optional

from pydantic import ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"llm_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"] = Field(
        "openai", description="The type of the LLM service."
    )

    base_url: str = Field(
        description="The base URL of the LLM service."
        " If service type is openai, it will be overwritten by openai_base_url."
    )
    api_key: SecretStr = Field(
        description="The API key of the LLM service."
        " If service type is openai, it will be overwritten by openai_api_key."
    )

    entry_point: str = Field(
        "/v1/chat/completions", description="The entry point of the LLM service."
    )

    model_name: str = Field("gpt-4o-mini", description="The LLM model name.")

    max_tokens: int = Field(
        16000,
        description="The maximum number of tokens that can be generated in the chat completion.",
    )

    timeout: float = Field(
        30.0, description="The timeout in seconds for the LLM requests."
    )

    rate_limit: int = Field(60, description="The max number of requests per unit time.")
    rate_limit_time_unit: Literal["second", "minute", "hour"] = Field(
        "minute", description="The time unit for the rate limit."
    )
    rate_limit_min_interval_seconds: float = Field(
        0.1,
        description="The min interval in seconds between requests to the embedding service.",
    )

    # TODO(tatiana): remove after Models table is refactored
    openai_base_url: Optional[str] = Field(
        None,
        description="The base URL of the OpenAI embedding service."
        " Useful only when service type is openai.",
    )
    openai_api_key: Optional[SecretStr] = Field(
        None,
        description="The API key of the OpenAI embedding service."
        " Useful only when service type is openai.",
    )
