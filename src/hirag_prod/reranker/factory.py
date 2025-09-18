from typing import Optional

from hirag_prod.configs.reranker_config import RerankConfig

from .api_reranker import ApiReranker
from .base import Reranker
from .local_reranker import LocalReranker


def create_reranker(
    reranker_config: Optional[RerankConfig] = None, reranker_type: Optional[str] = None
) -> Reranker:
    # Fallback to environment-based config if no config provided (for backward compatibility)
    if reranker_config is None:
        reranker_config = RerankConfig()

    # Allow override of type if explicitly provided
    if reranker_type is not None:
        reranker_config.reranker_type = reranker_type.lower()

    if reranker_config.reranker_type == "api":
        return ApiReranker(
            reranker_config.voyage_api_key,
            reranker_config.voyage_reranker_model_base_url,
            reranker_config.voyage_reranker_model_name,
        )
    elif reranker_config.reranker_type == "local":
        return LocalReranker(
            reranker_config.local_reranker_model_base_url,
            reranker_config.local_reranker_model_name,
            reranker_config.local_reranker_model_entry_point,
            reranker_config.local_reranker_model_authorization,
        )
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_config.reranker_type}")
