from abc import ABC, abstractmethod
from datetime import datetime
from logging import getLogger
from typing import Dict, List, Set, Tuple, Union

from hirag_prod.reranker.utils import detect_language
from hirag_prod.resources.functions import get_chinese_convertor

logger = getLogger(__name__)


class Reranker(ABC):

    @abstractmethod
    async def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        pass

    async def _chinese_to_simplified(self, text: str) -> str:
        return get_chinese_convertor("t2s").convert(text)

    def _has_language(self, languages: Set[str], target: Union[Set[str], str]) -> bool:
        if isinstance(target, set):
            return not target.isdisjoint(languages)
        return target in languages

    async def _process_query(
        self, query: List[str], rerank_with_time: bool
    ) -> List[Tuple[str, Set[str]]]:

        def add_timestamp_to_query(text: str) -> str:
            return f"{text}\n\n[Timestamp: {datetime.now().isoformat()}]"

        unique_queries = set()
        return_queries = []

        for q in query:
            language_detected = detect_language(q)
            if self._has_language(language_detected, "CHINESE"):
                q = await self._chinese_to_simplified(q)
            if rerank_with_time:
                q = add_timestamp_to_query(q)

            if q in unique_queries:
                continue

            unique_queries.add(q)
            return_queries.append((q, language_detected))

        return return_queries

    async def _prepare_documents(
        self, items: List[Dict], key: str, rerank_with_time: bool
    ) -> List[Tuple[str, Set[str]]]:
        docs = []

        for item in items:
            text = item.get(key, "")
            language_detected = detect_language(text)
            if self._has_language(language_detected, "CHINESE"):
                text = await self._chinese_to_simplified(text)
            if rerank_with_time:
                extracted_timestamp = item.get("extractedTimestamp", None)
                if extracted_timestamp:
                    text = f"{text}\n\n[Timestamp: {extracted_timestamp}]"
                else:
                    text = f"{text}\n\n[Timestamp: {datetime.min.isoformat()}]"

            docs.append((text, language_detected))

        return docs

    async def rerank(
        self,
        query: Union[str, List[str]],
        items: List[Dict],
        key: str = "text",
        rerank_with_time: bool = False,
    ) -> List[Dict]:
        if not items:
            return []

        if isinstance(query, str):
            query = [query]

        query = await self._process_query(query, rerank_with_time)
        docs = await self._prepare_documents(items, key, rerank_with_time)

        # Final reranked items with maximum scores
        max_scores = {}

        # Process each query and track maximum scores
        for query_idx, single_query in enumerate(query):
            sq, sq_lang = single_query

            # Create mapping from filtered docs to original item indices
            docs_to_rerank = []
            filtered_to_original_idx = []

            for original_idx, (doc, lang) in enumerate(docs):
                if query_idx == 0 or self._has_language(lang, sq_lang):
                    docs_to_rerank.append(doc)
                    filtered_to_original_idx.append(original_idx)

            if not docs_to_rerank:
                logger.warning(
                    f"No documents containing language {sq_lang}, skipping rerank."
                )
                continue

            if query_idx == 0:
                logger.info(
                    f"Reranking {len(docs_to_rerank)} documents with original query"
                )
            else:
                logger.info(
                    f"Reranking {len(docs_to_rerank)} documents containing language {sq_lang}"
                )

            results = await self._call_api(sq, docs_to_rerank)

            # Map results back to original indices and track max scores
            for r in results:
                filtered_idx = r.get("index")
                if filtered_idx is not None and 0 <= filtered_idx < len(
                    filtered_to_original_idx
                ):
                    original_idx = filtered_to_original_idx[filtered_idx]
                    score = r.get("relevance_score", 0.0)
                    # Keep the maximum score for each document
                    if (
                        original_idx not in max_scores
                        or score > max_scores[original_idx]
                    ):
                        max_scores[original_idx] = score

        reranked = []
        for idx, score in max_scores.items():
            item = items[idx].copy()
            item["relevance_score"] = score
            reranked.append(item)

        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked
