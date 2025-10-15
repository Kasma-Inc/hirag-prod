from typing import List

import httpx

from hirag_prod.configs.functions import get_envs, get_shared_variables
from hirag_prod.rate_limiter import RateLimiter
from hirag_prod.reranker.base import Reranker

rate_limiter = RateLimiter()


class ApiReranker(Reranker):
    def __init__(self, api_key: str, endpoint: str, model: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    @rate_limiter.limit(
        "reranker",
        "RERANKER_RATE_LIMIT_MIN_INTERVAL_SECONDS",
        "RERANKER_RATE_LIMIT",
        "RERANKER_RATE_LIMIT_TIME_UNIT",
    )
    async def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        """Async API call to avoid blocking the event loop"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "query": query,
            "documents": documents,
            "model": self.model,
        }

        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)

            if response.status_code != 200:
                error_text = response.text
                raise Exception(
                    f"Reranker API error {response.status_code}: {error_text}"
                )

            result = response.json()
            if get_envs().ENABLE_TOKEN_COUNT:
                get_shared_variables().input_token_count_dict[
                    "reranker"
                ].value += result.get("usage", {}).get("total_tokens", 0)
            return result.get("data", [])
