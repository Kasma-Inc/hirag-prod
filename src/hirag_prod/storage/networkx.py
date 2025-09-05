import asyncio
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import networkx as nx

from hirag_prod._utils import _limited_gather_with_factory, log_error_info
from hirag_prod.configs.functions import get_hi_rag_config
from hirag_prod.schema import Entity, Relation
from hirag_prod.storage.base_gdb import BaseGDB
from hirag_prod.summarization import BaseSummarizer, TrancatedAggregateSummarizer


@dataclass
class NetworkXGDB(BaseGDB):
    path: str
    graph: nx.DiGraph
    llm_func: Callable
    llm_model_name: str
    summarizer: Optional[BaseSummarizer]

    @classmethod
    def create(
        cls,
        path: str,
        llm_func: Callable,
        llm_model_name: str = "gpt-4o-mini",
        summarizer: Optional[BaseSummarizer] = None,
    ):
        if not os.path.exists(path):
            graph = nx.DiGraph()
        else:
            graph = cls.load(path)
        if summarizer is None:
            summarizer = TrancatedAggregateSummarizer(
                extract_func=llm_func, llm_model_name=llm_model_name
            )
        return cls(
            path=path,
            graph=graph,
            llm_func=llm_func,
            llm_model_name=llm_model_name,
            summarizer=summarizer,
        )

    async def _upsert_node(
        self, node: Entity, record_description: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Upsert a node into the graph.

        This method adds a new node to the graph if it doesn't exist, or updates an existing node.
        For concurrent upsertion, we use the following strategy:
        If the node not in the graph, add it. Use the database's transaction atomic to
        ensure the consistency of the graph.
        If the node in the graph, we record the description which we use to update the current node.
        If the record_description is the same as the description in the graph, we update the node, otherwise
        return the description in the graph, to generate the new description.

        Args:
            node (Entity): The entity node to be inserted or updated
            record_description (Optional[List[str]]): Description to compare with existing node's description

        Returns:
            Optional[List[str]]: If the node exists and has a different description, returns the existing description.
                            Otherwise returns None.

        """
        if node.id not in self.graph.nodes:
            try:
                self.graph.add_nodes_from(
                    [
                        (
                            node.id,
                            {
                                **node.metadata.__dict__,
                                "entity_name": node.page_content,
                            },
                        )
                    ]
                )
                return None
            except Exception as e:
                log_error_info(
                    logging.ERROR, "Failed to upsert node", e, raise_error=True
                )
        else:
            node_in_db = self.graph.nodes[node.id]
            # Handle both old string format and new list format for backwards compatibility
            latest_description = node_in_db.get("description", [])
            if isinstance(latest_description, str):
                latest_description = [latest_description]
            elif latest_description is None:
                latest_description = []

            current_description = node.metadata.description
            if record_description == latest_description:
                self.graph.nodes[node.id].update(
                    {**node.metadata.__dict__, "entity_name": node.page_content}
                )
                return None
            elif record_description is None:
                if current_description == latest_description:
                    # update an existing node
                    # skip the merge process
                    return None
                else:
                    # require to merge with the latest description
                    return latest_description
            else:
                # require to merge with the latest description
                return latest_description

    async def _merge_node(self, node: Entity, latest_description: List[str]) -> Entity:
        # Directly merge description lists without summarization
        current_descriptions = node.metadata.description
        merged_descriptions = latest_description + current_descriptions
        # Remove duplicates while preserving order
        seen = set()
        unique_descriptions = []
        for desc in merged_descriptions:
            if desc not in seen:
                seen.add(desc)
                unique_descriptions.append(desc)
        node.metadata.description = unique_descriptions
        return node

    async def upsert_node(self, node: Entity):
        record_description = None
        while True:
            latest_description = await self._upsert_node(node, record_description)
            if latest_description is None:
                break
            else:
                node = await self._merge_node(node, latest_description)
                record_description = latest_description

    async def upsert_nodes(self, nodes: List[Entity], concurrency: int | None = None):
        if concurrency is None:
            coros = [self.upsert_node(node) for node in nodes]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    import logging

                    logging.warning(f"[upsert_nodes] Task failed: {r}")
        else:
            factories = [lambda node=node: self.upsert_node(node) for node in nodes]
            await _limited_gather_with_factory(factories, concurrency)

    async def upsert_relation(self, relation: Relation):
        try:
            props = relation.properties or {}
            workspace_id = props.get("workspace_id")
            knowledge_base_id = props.get("knowledge_base_id")
            chunk_id = props.get("chunk_id")

            def is_chunk(node_id: str) -> bool:
                return str(node_id).startswith("chunk-")

            def ensure_node(node_id: str, name_hint: Optional[str] = None):
                if node_id not in self.graph.nodes:
                    self.graph.add_node(node_id)
                if workspace_id is not None:
                    self.graph.nodes[node_id]["workspace_id"] = workspace_id
                if knowledge_base_id is not None:
                    self.graph.nodes[node_id]["knowledge_base_id"] = knowledge_base_id

                if not is_chunk(node_id):
                    if name_hint:
                        self.graph.nodes[node_id]["entity_name"] = name_hint
                    self.graph.nodes[node_id].setdefault("entity_type", "entity")
                    self.graph.nodes[node_id].setdefault("chunk_ids", [])
                    if (
                        chunk_id
                        and chunk_id not in self.graph.nodes[node_id]["chunk_ids"]
                    ):
                        self.graph.nodes[node_id]["chunk_ids"].append(chunk_id)

            source_name = None if is_chunk(relation.source) else props.get("source")
            target_name = None if is_chunk(relation.target) else props.get("target")

            ensure_node(relation.source, name_hint=source_name)
            ensure_node(relation.target, name_hint=target_name)

            self.graph.add_edge(relation.source, relation.target, **props)

        except Exception as e:
            log_error_info(
                logging.ERROR, "Failed to upsert relation", e, raise_error=True
            )

    async def query_node(self, node_id: str) -> Entity:
        node = self.graph.nodes[node_id]
        entity_name = node.get("entity_name", "")
        metadata = {k: v for k, v in node.items() if k != "entity_name"}

        # Ensure all required fields exist with default values
        # Ensure description is always a list
        if "description" in metadata:
            if isinstance(metadata["description"], str):
                metadata["description"] = [metadata["description"]]
            elif metadata["description"] is None:
                metadata["description"] = []
        else:
            metadata["description"] = []

        # Ensure entity_type exists
        if "entity_type" not in metadata:
            metadata["entity_type"] = "UNKNOWN"

        # Ensure chunk_ids exists
        if "chunk_ids" not in metadata:
            metadata["chunk_ids"] = []

        return Entity(
            id=node_id,
            page_content=entity_name,
            metadata=metadata,
        )

    async def query_edge(self, edge_id: str) -> Relation:
        edge = self.graph.edges[edge_id]
        return Relation(
            source=edge_id[0],
            target=edge_id[1],
            properties=edge,
        )

    async def query_one_hop(self, node_id: str) -> (List[Entity], List[Relation]):  # type: ignore
        neighbors = list(self.graph.neighbors(node_id))
        edges = list(self.graph.edges(node_id))
        neighbor_results = await asyncio.gather(
            *[self.query_node(neighbor) for neighbor in neighbors],
            return_exceptions=True,
        )
        edge_results = await asyncio.gather(
            *[self.query_edge(edge) for edge in edges], return_exceptions=True
        )

        # Filter out failed results and only return successful ones
        successful_neighbors = []
        for r in neighbor_results:
            if isinstance(r, Exception):
                import logging

                logging.warning(f"[query_one_hop] Neighbor task failed: {r}")
            else:
                successful_neighbors.append(r)

        successful_edges = []
        for r in edge_results:
            if isinstance(r, Exception):
                import logging

                logging.warning(f"[query_one_hop] Edge task failed: {r}")
            else:
                successful_edges.append(r)

        return successful_neighbors, successful_edges

    async def pagerank_top_chunks_with_reset(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[tuple[str, float]]:
        """Compute personalized PageRank with a provided reset/personalization vector.

        Args:
            reset_weights: Mapping from node id to non-negative weight. Will be normalized.
            topk: Number of top chunk nodes to return.
            alpha: PageRank damping factor.

        Returns:
            List of tuples (chunk_id, score) sorted by descending score.
        """

        topk = topk or get_hi_rag_config().default_query_top_k
        alpha = alpha or get_hi_rag_config().default_pagerank_damping

        if topk <= 0:
            return []

        # Use undirected view to ensure bidirectional influence
        G = self.graph.to_undirected()
        if G.number_of_nodes() == 0:
            return []

        nodes_to_keep = [
            n
            for n, data in G.nodes(data=True)
            if (data.get("workspace_id") == workspace_id)
            and (data.get("knowledge_base_id") == knowledge_base_id)
        ]
        G = G.subgraph(nodes_to_keep).copy()
        if G.number_of_nodes() == 0:
            return []

        # Sanitize and normalize personalization vector
        personalization: dict[str, float] = {}
        total_mass = 0.0
        for node, weight in (reset_weights or {}).items():
            if node in G:
                try:
                    w = float(weight)
                except Exception as e:
                    log_error_info(
                        logging.WARNING, "Failed to convert weight to float", e
                    )
                    continue
                if w > 0 and not (w != w):  # exclude NaN and non-positive
                    personalization[node] = w
                    total_mass += w

        if total_mass <= 0:
            return []

        # Normalize to sum to 1
        for node in personalization:
            personalization[node] /= total_mass

        pr = nx.pagerank(
            G, alpha=alpha, personalization=personalization, weight="weight"
        )

        # Filter to chunk nodes only (by id prefix)
        chunk_scores = [
            (node, score)
            for node, score in pr.items()
            if str(node).startswith("chunk-")
        ]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:topk]

    async def dump(self):
        if os.path.dirname(self.path) != "":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    async def clean_up(self):
        pass
