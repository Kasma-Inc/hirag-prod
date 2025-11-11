from typing import Optional

from hirag_prod.configs.functions import get_reranker_config
from hirag_prod.configs.reranker_config import RerankConfig
from hirag_prod.reranker.aliyun_reranker import AliyunReranker
from hirag_prod.reranker.base import Reranker
from hirag_prod.reranker.local_reranker import LocalReranker


def create_reranker(
    reranker_config: Optional[RerankConfig] = None, reranker_type: Optional[str] = None
) -> Reranker:
    # Fallback to environment-based config if no config provided (for backward compatibility)
    if reranker_config is None:
        reranker_config = get_reranker_config()

    # Allow override of type if explicitly provided
    if reranker_type is not None:
        reranker_config.reranker_type = reranker_type.lower()

    if reranker_config.reranker_type == "local":
        return LocalReranker(
            reranker_config.base_url,
            reranker_config.model_name,
            reranker_config.entry_point,
            reranker_config.api_key.get_secret_value(),
        )
    elif reranker_config.reranker_type == "aliyun":
        return AliyunReranker(
            api_key=reranker_config.api_key.get_secret_value(),
            base_url=reranker_config.base_url,
            model=reranker_config.model_name,
        )
    raise ValueError(f"Unsupported reranker type: {reranker_config.reranker_type}")
