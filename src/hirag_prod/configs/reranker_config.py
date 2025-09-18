from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class RerankConfig(BaseSettings):
    """Reranker configuration"""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    # Reranker type selection
    reranker_type: Literal["api", "local"] = "api"

    # API reranker (Voyage AI) settings
    voyage_api_key: Optional[str] = None
    voyage_reranker_model_name: str = "rerank-2"
    voyage_reranker_model_base_url: str = "https://api.voyageai.com/v1/rerank"

    # Local reranker settings
    local_reranker_model_base_url: Optional[str] = None
    local_reranker_model_name: str = "Qwen3-Reranker-8B"
    local_reranker_model_entry_point: str = "/rerank"
    local_reranker_model_authorization: Optional[str] = None

    @model_validator(mode="after")
    def validate_config_based_on_type(self) -> "RerankConfig":
        if self.reranker_type == "api":
            if not self.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is required when RERANKER_TYPE is api")
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
