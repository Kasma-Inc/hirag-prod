from typing import Optional

from hirag_prod.configs.functions import get_reranker_configs
from hirag_prod.configs.reranker_config import RerankConfig
from hirag_prod.reranker.base import Reranker
from hirag_prod.reranker.local_reranker import LocalReranker
from hirag_prod.resources.functions import get_default_model_id


def create_reranker(
    reranker_config: Optional[RerankConfig] = None, reranker_type: Optional[str] = None
) -> Reranker:
    # Fallback to environment-based config if no config provided (for backward compatibility)
    if reranker_config is None:
        reranker_config = get_reranker_configs()[get_default_model_id("reranker")]

    if reranker_config.db_table["type"] == "local":
        return LocalReranker(
            reranker_config.base_url,
            reranker_config.model_name,
            reranker_config.entry_point,
            reranker_config.api_key.get_secret_value(),
        )
    raise ValueError(f"Unsupported reranker type: {reranker_config.db_table['type']}")
