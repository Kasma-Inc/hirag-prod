from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class RerankConfig(BaseSettings):
    """Reranker configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: x.upper(),
        populate_by_name=True,
        extra="ignore",
    )

    # Reranker type selection
    reranker_type: Literal["api", "local"] = "local"

    # API reranker settings
    # TODO: Add API reranker settings

    # Local reranker settings
    local_reranker_model_base_url: Optional[str] = None
    local_reranker_model_name: str = "Qwen3-Reranker-8B"
    local_reranker_model_entry_point: str = "/rerank"
    local_reranker_model_authorization: Optional[str] = None

    @model_validator(mode="after")
    def validate_config_based_on_type(self) -> "RerankConfig":
        if self.reranker_type == "api":
            raise ValueError("API reranker is not supported temporarily")
        elif self.reranker_type == "local":
            if not self.local_reranker_model_base_url:
                raise ValueError(
                    "LOCAL_RERANKER_MODEL_BASE_URL is required when RERANKER_TYPE is local"
                )
            if not self.local_reranker_model_authorization:
                raise ValueError(
                    "LOCAL_RERANKER_MODEL_AUTHORIZATION is required when RERANKER_TYPE is local"
                )

        return self
