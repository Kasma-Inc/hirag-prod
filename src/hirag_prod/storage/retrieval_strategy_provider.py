#! /usr/bin/env python3
import logging
from typing import Any, Dict, List, Union

from lancedb.query import AsyncQuery, AsyncVectorQuery, LanceQueryBuilder
from lancedb.rerankers import VoyageAIReranker

from hirag_prod.configs.functions import get_init_config, get_reranker_config
from hirag_prod.reranker import LocalReranker


class BaseRetrievalStrategyProvider:
    """Implement this class"""

    default_topk = get_init_config().default_query_top_k
    default_topn = get_init_config().default_query_top_n

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
        reranker_config = get_reranker_config()
        reranker_type = reranker_config.reranker_type

        if reranker_type == "local":
            # Use deployed reranker
            reranker = LocalReranker(top_n=topn)
            reranked_query = query.rerank(reranker=reranker, query_string=text)
            return reranked_query
        else:
            # Use API-based reranker (VoyageAI)

            reranker = VoyageAIReranker(
                api_key=reranker_config.voyage_api_key,
                model_name=reranker_config.voyage_reranker_model_name,
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
