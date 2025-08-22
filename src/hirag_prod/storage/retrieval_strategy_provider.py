#! /usr/bin/env python3
import logging
import os
from typing import Any, Dict, List, Union

from lancedb.query import AsyncQuery, LanceQueryBuilder
from lancedb.rerankers import VoyageAIReranker

from hirag_prod.reranker import LocalReranker


class BaseRetrievalStrategyProvider:
    """Implement this class"""

    default_topk = 10
    default_topn = 5

    def rerank_catalog_query(
        self,
        query: Union[LanceQueryBuilder, AsyncQuery],
        text: str,  # pylint: disable=unused-argument
    ):
        return query

    async def rerank_chunk_query(
        self,
        query: Any,
        text: str,
        topn: int | None = None,  # pylint: disable=unused-argument
    ):
        return query

    def format_catalog_search_result_to_llm(
        self, input_data: List[Dict[str, Any]]
    ) -> str:
        return str(input_data)

    def format_chunk_search_result_to_llm(
        self, input_data: List[Dict[str, Any]]
    ) -> str:
        return str(input_data)


class RetrievalStrategyProvider(BaseRetrievalStrategyProvider):
    """Provides parameters for the retrieval strategy & process the retrieval results for LLM."""

    def rerank_catalog_query(
        self, query: Union[LanceQueryBuilder, AsyncQuery], text: str
    ):
        # TODO(tatiana): add rerank logic
        logging.info("TODO: add rerank logic for %s", text)
        return query

    async def rerank_chunk_query(self, query: Any, text: str, topn: int | None):
        """
        Rerank chunk query. Supports:
          - LanceDB query objects (with .rerank)
          - Python list results from PGVector
        """
        # If it's a LanceDB query object, use its reranker
        if hasattr(query, "rerank"):
            reranker_type = os.getenv("RERANKER_TYPE", "api")
            if reranker_type == "local":
                reranker = LocalReranker(top_n=topn or self.default_topn)
                return query.rerank(reranker=reranker, query_string=text)
            else:
                reranker = VoyageAIReranker(
                    api_key=os.getenv("VOYAGE_API_KEY"),
                    model_name=os.getenv("API_RERANKER_MODEL", "rerank-2"),
                    top_n=topn or self.default_topn,
                    return_score="relevance",
                )
                return query.rerank(reranker=reranker, query_string=text)

        # If it's already a list of dicts (PGVector path), keep sorted order and just trim to topn
        if isinstance(query, list):
            if isinstance(topn, int) and topn > 0:
                return query[:topn]
            return query

        # Fallback: return as-is
        return query

    def format_catalog_search_result_to_llm(
        self, input_data: List[Dict[str, Any]]
    ) -> str:
        # TODO(tatiana): need to format the data in a way that is easy to read by the LLM
        return str(input_data)

    def format_chunk_search_result_to_llm(
        self, input_data: List[Dict[str, Any]]
    ) -> str:
        # TODO(tatiana): need to format the data in a way that is easy to read by the LLM
        return str(input_data)
