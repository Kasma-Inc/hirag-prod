import os
from typing import Optional

from .api_reranker import ApiReranker
from .base import Reranker
from .local_reranker import LocalReranker


def create_reranker(reranker_type: Optional[str] = None) -> Reranker:
    reranker_type = (reranker_type or os.getenv("RERANKER_TYPE", "api")).lower()

    if reranker_type == "api":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY required")
        return ApiReranker(api_key, "https://api.voyageai.com/v1/rerank", "rerank-2")
    elif reranker_type == "local":
        base_url = os.getenv("RERANKER_MODEL_BASE_URL")
        model_name = os.getenv("RERANKER_MODEL_NAME", "Qwen3-Reranker-8B")
        entry_point = os.getenv("RERANKER_MODEL_ENTRY_POINT", "/rerank")
        auth_token = os.getenv("RERANKER_MODEL_AUTHORIZATION")
        if not base_url:
            raise ValueError("RERANKER_MODEL_BASE_URL required")
        if not auth_token:
            raise ValueError("RERANKER_MODEL_AUTHORIZATION required")
        return LocalReranker(base_url, model_name, entry_point, auth_token)
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}")
