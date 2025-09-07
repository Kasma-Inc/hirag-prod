import asyncio
import logging
from typing import Any, Dict, List, Optional

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_hi_rag_config
from hirag_prod.storage.storage_manager import StorageManager

logger = logging.getLogger("HiRAG")


class QueryService:
    """Query service"""

    def __init__(self, storage: StorageManager):
        self.storage = storage

    async def query_chunks(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Query chunks via unified storage"""
        return await self.storage.query_chunks(*args, **kwargs)

    async def recall_chunks(self, *args, **kwargs) -> Dict[str, Any]:
        """Recall chunks and return both raw results and extracted chunk_ids.

        Returns:
            Dict with keys:
                - "chunks": raw chunk search results
                - "chunk_ids": list of document_key values
        """
        chunks = await self.query_chunks(*args, **kwargs)
        chunk_ids = [c.get("documentKey") for c in chunks if c.get("documentKey")]
        return {"chunks": chunks, "chunk_ids": chunk_ids}

    async def query_triplets(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Query relations using unified storage"""
        return await self.storage.query_triplets(*args, **kwargs)

    async def recall_triplets(self, *args, **kwargs) -> Dict[str, Any]:
        """Recall triplets and return both raw results and aggregated entity_ids.

        Returns:
            Dict with keys:
                - "relations": raw triplet search results
                - "entity_ids": unique list of entity ids appearing as source/target
        """
        relations = await self.query_triplets(*args, **kwargs)
        entity_id_set = set()
        for rel in relations:
            src = rel.get("source")
            tgt = rel.get("target")
            if src:
                entity_id_set.add(src)
            if tgt:
                entity_id_set.add(tgt)
        return {"relations": relations, "entity_ids": list(entity_id_set)}

    async def query_chunk_embeddings(
        self, workspace_id: str, knowledge_base_id: str, chunk_ids: List[str]
    ) -> Dict[str, Any]:
        """Query chunk embeddings"""
        if not chunk_ids:
            return {}

        res = {}
        try:
            rows = await self.storage.query_by_keys(
                chunk_ids=chunk_ids,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                columns_to_select=["documentKey", "vector"],
            )
            for row in rows:
                key = row.get("documentKey")
                if key and row.get("vector") is not None:
                    res[key] = row.get("vector")
                elif key:
                    logger.warning(f"Chunk {key} has no vector data")
                    res[key] = None
        except Exception as e:
            log_error_info(logging.ERROR, "Failed to query chunk embeddings", e)
            return {}
        return res

    async def get_chunks_by_ids(
        self,
        chunk_ids: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        columns_to_select: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch chunk rows by document_key list, preserving input order where possible."""
        if not chunk_ids:
            return []
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "fileName",
                "private",
                "updatedAt",
                "documentKey",
            ]
        rows = await self.storage.query_by_keys(
            chunk_ids=chunk_ids,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            columns_to_select=columns_to_select,
        )
        # Build map for stable ordering
        by_id = {row.get("documentKey"): row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def dual_recall_with_pagerank(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        topk: Optional[int] = None,
        topn: Optional[int] = None,
        link_top_k: Optional[int] = None,
        passage_node_weight: Optional[float] = None,
        damping: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Two-path retrieval + PageRank fusion.

        - Recall chunks to form passage reset weights
        - Recall triplets to form phrase (entity) reset weights with frequency penalty
        - Build reset = phrase_weights + passage_weights and run Personalized PageRank
        - If no facts, fall back to DPR order (query rerank order)
        """

        topk = topk or get_hi_rag_config().default_query_top_k
        topn = topn or get_hi_rag_config().default_query_top_n
        link_top_k = link_top_k or get_hi_rag_config().default_link_top_k
        passage_node_weight = (
            passage_node_weight or get_hi_rag_config().default_passage_node_weight
        )

        damping = damping or get_hi_rag_config().default_pagerank_damping

        # Path 1: chunk recall (rerank happens in VDB query)
        chunk_recall = await self.recall_chunks(
            query,
            topk=topk,
            topn=topn,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )
        query_chunks = chunk_recall["chunks"]
        query_chunk_ids = chunk_recall["chunk_ids"]

        # Path 2: triplet recall -> entity seeds
        triplet_recall = await self.recall_triplets(
            query=query,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            topk=topk,
            topn=topn,
        )
        query_triplets = triplet_recall["relations"]
        query_entity_ids = triplet_recall["entity_ids"]

        # Build passage weights from chunk ranks (approximate DPR)
        passage_weights: Dict[str, float] = {}
        if chunk_recall["chunk_ids"]:
            # Inverse-rank weights then min-max normalize
            raw_weights = []
            for rank, cid in enumerate(query_chunk_ids):
                if cid:
                    w = 1.0 / (rank + 1)
                    passage_weights[cid] = w
                    raw_weights.append(w)
            if raw_weights:
                min_w, max_w = min(raw_weights), max(raw_weights)
                scale = (max_w - min_w) if (max_w - min_w) > 0 else 1.0
                for cid in list(passage_weights.keys()):
                    passage_weights[cid] = (
                        (passage_weights[cid] - min_w) / scale
                    ) * passage_node_weight

        # Build phrase weights from relations with frequency penalty and averaging
        phrase_weights: Dict[str, float] = {}
        if triplet_recall["relations"]:
            occurrence_counts: Dict[str, int] = {}
            # Accumulate inverse-rank weights to both source and target entities
            for rank, rel in enumerate(query_triplets):
                base_w = 1.0 / (rank + 1)
                for ent_id in [rel.get("source"), rel.get("target")]:
                    if not ent_id:
                        continue
                    phrase_weights[ent_id] = phrase_weights.get(ent_id, 0.0) + base_w
                    occurrence_counts[ent_id] = occurrence_counts.get(ent_id, 0) + 1

            async def _fetch_entity_chunk_count(ent: str) -> int:
                try:
                    node = await self.storage.gdb.query_node(ent)
                    chunk_ids: List[str] = []
                    if hasattr(node, "metadata"):
                        if hasattr(node.metadata, "chunk_ids"):
                            chunk_ids = node.metadata.chunk_ids

                    return len(chunk_ids) if isinstance(chunk_ids, list) else 0
                except Exception as e:
                    log_error_info(
                        logging.ERROR, "Failed to fetch entity chunk count", e
                    )
                    return 0

            counts = await asyncio.gather(
                *[_fetch_entity_chunk_count(eid) for eid in query_entity_ids]
            )
            ent_to_chunk_count = {
                eid: cnt for eid, cnt in zip(query_entity_ids, counts)
            }

            for ent_id in query_entity_ids:
                freq_penalty = ent_to_chunk_count.get(ent_id, 0)
                denom = (
                    float(freq_penalty) if freq_penalty and freq_penalty > 0 else 1.0
                )
                phrase_weights[ent_id] = (phrase_weights[ent_id] / denom) / float(
                    occurrence_counts.get(ent_id, 1)
                )

            # Keep only top link_top_k entities
            sorted_entities = sorted(
                phrase_weights.items(), key=lambda x: x[1], reverse=True
            )[: max(1, link_top_k)]
            phrase_weights = dict(sorted_entities)

        # If no fact signal, return DPR order directly
        if not phrase_weights:
            return {
                "pagerank": [],
                "query_top": query_chunks,
            }

        # Combine phrase and passage weights
        reset_weights: Dict[str, float] = {}
        for k, v in passage_weights.items():
            if v > 0:
                reset_weights[k] = reset_weights.get(k, 0.0) + v
        for k, v in phrase_weights.items():
            if v > 0:
                reset_weights[k] = reset_weights.get(k, 0.0) + v

        # Personalized PageRank over graph using reset vector
        pr_ranked = await self.storage.gdb.pagerank_top_chunks_with_reset(
            reset_weights=reset_weights,
            topk=topk,
            alpha=damping,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )
        pr_ids = [cid for cid, _ in pr_ranked]

        pr_rows = await self.get_chunks_by_ids(
            pr_ids, workspace_id=workspace_id, knowledge_base_id=knowledge_base_id
        )
        pr_score_map = {cid: score for cid, score in pr_ranked}
        for row in pr_rows:
            row["pagerank_score"] = pr_score_map.get(row.get("documentKey"), 0.0)

        return {
            "pagerank": pr_rows,
            "query_top": query_chunks,
        }

    async def query(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """Query Strategy (default: dual_recall_with_pagerank)"""
        result = await self.dual_recall_with_pagerank(
            query=query,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )
        result["chunks"] = (
            result.get("pagerank")
            if result.get("pagerank")
            else result.get("query_top", [])
        )
        return result
