from typing import Literal, Optional

from pydantic import ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class TranslatorConfig(BaseSettings):
    """Translator configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"translator_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"] = Field(
        "local",
        description="The type of the translator service.",
    )

    # Translator settings
    base_url: str = Field(
        description="The base URL of the translator service."
        " If service type is openai, it will be overwritten by openai_base_url."
    )
    api_key: SecretStr = Field(
        description="The API key of the translator service."
        " If service type is openai, it will be overwritten by openai_api_key."
    )
    entry_point: str = Field(
        "/v1/chat/completions", description="The entry point of the translator service."
    )
    model_name: Optional[str] = Field(
        "Hunyuan-MT-7B",
        description="The model name of the translator service."
        " Required when service type is local.",
    )

    # Additional translator settings
    timeout: Optional[float] = Field(
        3600.0, description="The timeout in seconds for the translator requests."
    )
    max_tokens: Optional[int] = Field(
        None,
        description="The maximum number of tokens that can be generated in the chat completion.",
    )
    temperature: Optional[float] = None

    # Rate limits
    rate_limit: int = Field(60, description="The max number of requests per unit time.")
    rate_limit_time_unit: Literal["second", "minute", "hour"] = Field(
        "minute", description="The time unit for the rate limit."
    )
    rate_limit_min_interval_seconds: float = Field(
        0.1,
        description="The min interval in seconds between requests to the translator service.",
    )

    # TODO(tatiana): remove after Models table is refactored
    openai_base_url: Optional[str] = Field(
        None,
        description="The base URL of the OpenAI translator service."
        " Useful only when service type is openai.",
    )
    openai_api_key: Optional[SecretStr] = Field(
        None,
        description="The API key of the OpenAI translator service."
        " Useful only when service type is openai.",
    )

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "TranslatorConfig":
        if self.service_type == "local":
            if not self.model_name:
                env_name = self.model_fields["model_name"].alias
                raise ValueError(f"{env_name} is required when service_type is local")

        if self.service_type == "openai":
            if not self.openai_base_url:
                env_name = self.model_fields["openai_base_url"].alias
                raise ValueError(f"{env_name} is required when service_type is openai")
            if not self.openai_api_key:
                env_name = self.model_fields["openai_api_key"].alias
                raise ValueError(f"{env_name} is required when service_type is openai")

        return self
