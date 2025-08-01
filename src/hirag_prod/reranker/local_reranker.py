"""Local deployment reranker implementation"""

import logging
import os
from typing import List, Optional

import pyarrow as pa
import requests
from lancedb.rerankers import Reranker


class LocalReranker(Reranker):
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        entry_point: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 30,
        column: str = "text",
        top_n: Optional[int] = None,
        return_score: str = "relevance",
    ):
        super().__init__(return_score)

        self.base_url = (base_url or os.getenv("RERANKER_MODEL_BASE_URL")).rstrip("/")
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME",
            "Qwen3-Reranker-8B",  # Default to use Qwen3-Reranker-8B
        )
        self.entry_point = entry_point or os.getenv(
            "RERANKER_MODEL_Entry_Point", "/rerank"
        )
        self.auth_token = auth_token or os.getenv("RERANKER_MODEL_Authorization")
        self.timeout = timeout
        self.column = column
        self.top_n = top_n
        self.logger = logging.getLogger(__name__)

        if not self.base_url:
            raise ValueError("RERANKER_MODEL_BASE_URL environment variable is required")
        if not self.auth_token:
            raise ValueError(
                "RERANKER_MODEL_Authorization environment variable is required"
            )

    def _call_api_sync(self, query: str, documents: List[str]) -> List[dict]:
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

        response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        if response.status_code != 200:
            error_text = response.text
            raise Exception(f"Reranker API error {response.status_code}: {error_text}")

        result = response.json()
        return result.get("results", [])

    def _rerank(self, result_set: pa.Table, query: str) -> pa.Table:
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set

        docs = result_set[self.column].to_pylist()

        results = self._call_api_sync(query, docs)

        if not results:
            raise Exception("Reranker API returned no results")

        # Extract indices and scores, sorted by relevance
        indices = []
        scores = []

        for result in results[: self.top_n] if self.top_n else results:
            idx = result["index"]
            score = result["relevance_score"]
            if idx < len(docs):
                indices.append(idx)
                scores.append(score)

        if not indices:
            return result_set.slice(0, 0)

        # Reorder results by relevance
        result_set = result_set.take(indices)
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )

        self.logger.info(
            f"Successfully reranked {len(docs)} documents to {len(result_set)} results"
        )
        return result_set

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        """Rerank hybrid search results"""
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._rerank(combined_results, query)

        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)

        return combined_results

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """Rerank vector search results"""
        vector_results = self._rerank(vector_results, query)

        if self.score == "relevance" and "_distance" in vector_results.column_names:
            vector_results = vector_results.drop_columns(["_distance"])

        return vector_results

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """Rerank FTS search results"""
        fts_results = self._rerank(fts_results, query)

        if self.score == "relevance" and "_score" in fts_results.column_names:
            fts_results = fts_results.drop_columns(["_score"])

        return fts_results
