from typing import Literal, Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Embedding configuration"""

    model_config = SettingsConfigDict(
        alias_generator=lambda x: f"embedding_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"] = Field(
        "local",
        description="The type of the embedding service.",
    )

    base_url: str = Field(
        description="The base URL of the embedding service."
        " If service type is openai, it will be overwritten by openai_base_url."
    )
    api_key: SecretStr = Field(
        description="The API key of the embedding service."
        " If service type is openai, it will be overwritten by openai_api_key."
    )

    entry_point: str = Field(
        "/v1/embeddings", description="The entry point of the embedding service."
    )
    model_name: Optional[str] = Field(
        "Qwen3-Embedding-8B",
        description="The name of the local embedding model. Required when service type is local.",
    )
    model_path: Optional[str] = Field(
        "Qwen3-Embedding-8B",
        description="The path of the local embedding model. Required when service type is local.",
    )
    default_batch_size: int = Field(
        1000,
        description="The default batch size for local embedding service. "
        "Useful only when service type is local.",
    )

    # Rate limits
    rate_limit: int = Field(
        6000, description="The max number of requests per unit time."
    )
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

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "EmbeddingConfig":
        if self.service_type == "local":
            if not self.model_name:
                env_name = self.model_config["alias_generator"]("model_name")
                raise ValueError(f"{env_name} is required when service_type is local")
            if not self.model_path:
                env_name = self.model_config["alias_generator"]("model_path")
                raise ValueError(f"{env_name} is required when service_type is local")

        if self.service_type == "openai":
            if not self.openai_base_url:
                env_name = self.model_fields["openai_base_url"].alias
                raise ValueError(f"{env_name} is required when service_type is openai")
            if not self.openai_api_key:
                env_name = self.model_fields["openai_api_key"].alias
                raise ValueError(f"{env_name} is required when service_type is openai")

            # Only keep the openai_base_url and openai_api_key if service_type is openai
            self.base_url = self.openai_base_url
            self.api_key = self.openai_api_key

        return self

    @model_validator(mode="before")
    @classmethod
    def validate_service_type_config(cls, data):
        if isinstance(data, dict):
            if "service_type" in data:
                if data["service_type"] == "openai":
                    data["base_url"] = data.get("openai_base_url")
                    data["api_key"] = data.get("openai_api_key")
        return data
