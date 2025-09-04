"""Local deployment reranker implementation"""

import logging
from typing import Dict, List

import httpx

from .base import Reranker


class LocalReranker(Reranker):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        entry_point: str,
        auth_token: str,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.entry_point = entry_point
        self.auth_token = auth_token
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    async def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        """Async API call to avoid blocking the event loop"""
        headers = {
            "Content-Type": "application/json",
            "Model-Name": self.model_name,
            "Entry-Point": self.entry_point,
            "Authorization": (
                self.auth_token
                if self.auth_token.startswith("Bearer ")
                else f"Bearer {self.auth_token}"
            ),
        }

        # templates for the Qwen3-Reranker-8B
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        document_template = "<Document>: {doc}{suffix}"
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        formatted_query = query_template.format(
            prefix=prefix, instruction=instruction, query=query
        )
        formatted_documents = [
            document_template.format(doc=doc, suffix=suffix) for doc in documents
        ]

        payload = {
            "query": formatted_query,
            "documents": formatted_documents,
        }

        url = f"{self.base_url}{self.entry_point}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                error_text = response.text
                raise Exception(
                    f"Reranker API error {response.status_code}: {error_text}"
                )

            result = response.json()
            return result.get("results", [])

    async def rerank(self, query: str, items: List[Dict], topn: int) -> List[Dict]:
        if not items or topn <= 0:
            return []

        topn = min(topn, len(items))
        docs = [item.get("text", "") for item in items]
        results = await self._call_api(query, docs)

        reranked = []
        for r in results[:topn]:
            idx = r.get("index")
            if idx is not None and 0 <= idx < len(items):
                item = items[idx].copy()
                item["score"] = r.get("relevance_score", 0.0)
                reranked.append(item)
        return reranked
