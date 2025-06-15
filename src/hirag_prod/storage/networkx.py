import asyncio
import os
import pickle
from dataclasses import dataclass
from typing import Callable, List, Optional

import networkx as nx

from hirag_prod._utils import _limited_gather
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
            graph = nx.Graph()
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
        self, node: Entity, record_description: Optional[str] = None
    ) -> Optional[str]:
        """
        Upsert a node into the graph.

        Thiy method adds a new node to the graph if it doesn't exist, or updates an existing node.
        For concurrent upsertion, we use the following strategy:
        If the node not in the graph, add it. Use the database's transaction atomic to
        ensure the consistency of the graph.
        If the node in the graph, we record the description which we use to update the current node.
        If the record_description is the same as the description in the graph, we update the node, otherwise
        return the description in the graph, to generate the new description.

        Args:
            node (Entity): The entity node to be inserted or updated
            record_description (Optional[str]): Description to compare with existing node's description

        Returns:
            Optional[str]: If the node exists and has a different description, returns the existing description.
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
                return
            except Exception as e:
                # TODO: handle the exception
                raise e
        else:
            node_in_db = self.graph.nodes[node.id]
            latest_description = node_in_db["description"]
            assert latest_description is not None
            if record_description == latest_description:
                self.graph.nodes[node.id].update(
                    {**node.metadata.__dict__, "entity_name": node.page_content}
                )
                return
            elif record_description is None:
                record_description = node.metadata.description
                if record_description == latest_description:
                    # update an existing node
                    # skip the merge process
                    return
                else:
                    # require to merge with the latest description
                    return latest_description
            else:
                # require to merge with the latest description
                return latest_description

    async def _merge_node(self, node: Entity, latest_description: str) -> Entity:
        description_list = [node.metadata.description]
        description = await self.summarizer.summarize_entity(
            node.page_content, description_list
        )
        node.metadata.description = description
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
        coros = [self.upsert_node(node) for node in nodes]
        if concurrency is None:
            await asyncio.gather(*coros)
        else:
            await _limited_gather(coros, concurrency)

    async def upsert_relation(self, relation: Relation):
        try:
            # TODO: Handle the case where the source or target is not in the graph
            # await self.upsert_node(relation.source)
            # await self.upsert_node(relation.target)
            self.graph.add_edge(relation.source, relation.target, **relation.properties)
        except Exception as e:
            # TODO: handle the exception
            raise e

    async def query_node(self, node_id: str) -> Entity:
        node = self.graph.nodes[node_id]
        return Entity(
            id=node_id,
            page_content=node["entity_name"],
            metadata={k: v for k, v in node.items() if k != "entity_name"},
        )

    async def query_edge(self, edge_id: str) -> Relation:
        edge = self.graph.edges[edge_id]
        return Relation(
            source=edge_id[0],
            target=edge_id[1],
            properties=edge,
        )

    async def query_one_hop(self, node_id: str) -> (List[Entity], List[Relation]):
        neighbors = list(self.graph.neighbors(node_id))
        edges = list(self.graph.edges(node_id))
        return await asyncio.gather(
            *[self.query_node(neighbor) for neighbor in neighbors]
        ), await asyncio.gather(*[self.query_edge(edge) for edge in edges])

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
