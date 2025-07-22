#! /usr/bin/env python3
import logging
import os
from typing import Any, Dict, List, Union

from lancedb.query import AsyncQuery, AsyncVectorQuery, LanceQueryBuilder
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

    def rerank_chunk_query(
        self, query: AsyncQuery, text: str  # pylint: disable=unused-argument
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

    def rerank_chunk_query(self, query: AsyncVectorQuery, text: str, topn: int):
        """
        Rerank chunk query using either API-based or local reranker
        """
        reranker_type = os.getenv("RERANKER_TYPE", "api")

        if reranker_type == "local":
            # Use deployed reranker
            reranker = LocalReranker(top_n=topn)
            reranked_query = query.rerank(reranker=reranker, query_string=text)
            return reranked_query
        else:
            # Use API-based reranker (VoyageAI)
            reranker = VoyageAIReranker(
                api_key=os.getenv("VOYAGE_API_KEY"),
                model_name=os.getenv("API_RERANKER_MODEL", "rerank-2"),
                top_n=topn,
                return_score="relevance",
            )
            reranked_query = query.rerank(reranker=reranker, query_string=text)
            return reranked_query

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
