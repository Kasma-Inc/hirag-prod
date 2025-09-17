from typing import Dict, List, Union

import httpx

from .base import Reranker


class ApiReranker(Reranker):
    def __init__(self, api_key: str, endpoint: str, model: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    async def rerank(
        self, query: Union[str, List[str]], items: List[Dict], topn: int
    ) -> List[Dict]:
        if not items or topn <= 0:
            return []

        topn = min(topn, len(items))
        documents = [item.get("text", "") for item in items]

        # Handle single query case
        if isinstance(query, str):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": query,
                        "documents": documents,
                        "model": self.model,
                        "top_k": topn,
                    },
                )
                response.raise_for_status()

                results = response.json().get("data", [])
                reranked = []
                for r in results:
                    idx = r.get("index")
                    if idx is not None and 0 <= idx < len(items):
                        item = items[idx].copy()
                        item["relevance_score"] = r.get("relevance_score", 0.0)
                        reranked.append(item)
                return reranked

        # Handle list of queries case - find max relevance score among all queries
        else:
            # Initialize scores for each document
            max_scores = {}

            async with httpx.AsyncClient() as client:
                # Process each query and track maximum scores
                for single_query in query:
                    response = await client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "query": single_query,
                            "documents": documents,
                            "model": self.model,
                            "top_k": len(items),  # Get scores for all items
                        },
                    )
                    response.raise_for_status()

                    results = response.json().get("data", [])
                    for r in results:
                        idx = r.get("index")
                        if idx is not None and 0 <= idx < len(items):
                            score = r.get("relevance_score", 0.0)
                            # Keep the maximum score for each document
                            if idx not in max_scores or score > max_scores[idx]:
                                max_scores[idx] = score

            # Create final reranked list with max scores
            reranked = []
            for idx, score in max_scores.items():
                item = items[idx].copy()
                item["relevance_score"] = score
                reranked.append(item)

            # Sort by score descending and return top n
            reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
            return reranked[:topn]
