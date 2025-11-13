from typing import List

import httpx

from hirag_prod.configs.functions import get_envs, get_shared_variables
from hirag_prod.rate_limiter import RateLimiter
from hirag_prod.reranker.base import Reranker
from hirag_prod.tracing import traced
from hirag_prod.usage import ModelIdentifier, ModelProvider, ModelUsage, UsageCollector

rate_limiter = RateLimiter()


class AliyunReranker(Reranker):
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @rate_limiter.limit(
        "reranker",
        "RERANKER_RATE_LIMIT_MIN_INTERVAL_SECONDS",
        "RERANKER_RATE_LIMIT",
        "RERANKER_RATE_LIMIT_TIME_UNIT",
    )
    @traced(record_args=[])
    async def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        """Async API call to avoid blocking the event loop"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {"return_documents": False},
        }

        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)

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
            UsageCollector.add_usage(
                ModelIdentifier(
                    id=self.model,
                    provider=ModelProvider.ALIYUN.value,
                ),
                ModelUsage(
                    prompt_tokens=result.get("usage", {}).get("total_tokens", 0),
                    completion_tokens=0,  # aliyun doesn't return completion tokens
                ),
            )
            res = result.get("output", {}).get("results", [])
            res = [{**r, "text": documents[r.get("index")]} for r in res]
            return res
