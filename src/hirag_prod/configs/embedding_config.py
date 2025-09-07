from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Embedding configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"embedding_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"]

    base_url: str
    api_key: str

    entry_point: str = "/v1/embeddings"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    default_batch_size: int = 200  # Default batch size for local embedding service

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "EmbeddingConfig":
        if self.service_type == "local":
            if not self.model_name:
                raise ValueError(
                    f"{self.model_config['alias_generator']('model_name')} is required when service_type is local"
                )
            if not self.model_path:
                raise ValueError(
                    f"{self.model_config['alias_generator']('model_path')} is required when service_type is local"
                )

        return self
