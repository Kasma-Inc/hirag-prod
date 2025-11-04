from typing import Literal

from pydantic import ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings


class RerankConfig(BaseSettings):
    """Reranker configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"reranker_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    # Reranker type selection
    reranker_type: Literal["local"] = Field(
        "local",
        alias="RERANKER_SERVICE_TYPE",
        description="The type of the reranker service.",
    )

    # Local reranker settings
    base_url: str = Field(description="The base URL of the reranker service.")
    api_key: SecretStr = Field(description="The API key of the reranker service.")
    entry_point: str = Field(
        "/rerank", description="The entry point of the reranker service."
    )
    model_name: str = Field(
        "Qwen3-Reranker-4B", description="The model name of the reranker service."
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
        description="The min interval in seconds between requests to the reranker service.",
    )
