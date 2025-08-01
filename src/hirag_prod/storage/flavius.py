import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from flavius import DataType, GraphDatabase, TimeStamp

from hirag_prod._utils import _limited_gather_with_factory
from hirag_prod.schema import Entity, Relation
from hirag_prod.storage.base_gdb import BaseGDB
from hirag_prod.summarization import BaseSummarizer, TrancatedAggregateSummarizer


@dataclass
class FlaviusGDB(BaseGDB):
    path: str
    driver: GraphDatabase
    namespace: str
    graph_name: str
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
        flavius_url: str = "http://localhost:30000",
    ):
        # Create driver and verify connectivity
        driver = GraphDatabase.driver(flavius_url)
        driver.verify_connectivity()

        # Create unique namespace and graph name based on path and timestamp
        import time
        import random
        timestamp = int(time.time() * 1000)  # Use milliseconds for more precision
        random_suffix = random.randint(1000, 9999)
        namespace = f"ns_{timestamp}_{random_suffix}"
        graph_name = "g"

        # Create namespace and graph
        driver.create_namespace(namespace)
        driver.create_graph(graph_name, namespace=namespace)

        # Create vertex table for Entity
        driver.create_vertex_table(
            "Entity",
            [
                ("id", DataType.VARCHAR, False),  # Primary key
                ("entity_name", DataType.VARCHAR),
                ("entity_type", DataType.VARCHAR),
                ("description", DataType.VARCHAR),  # JSON string for list
                ("chunk_ids", DataType.VARCHAR),  # JSON string for list
            ],
            "id",  # primary key
            namespace=namespace,
            graph=graph_name,
        )

        # Create edge table for Relation
        driver.create_edge_table(
            "Relation",
            source_vertex="Entity",
            target_vertex="Entity",
            columns=[
                ("relation_type", DataType.VARCHAR),
                ("properties", DataType.VARCHAR),  # JSON string for dict
            ],
            directed=True,
            namespace=namespace,
            graph=graph_name,
        )

        if summarizer is None:
            summarizer = TrancatedAggregateSummarizer(
                extract_func=llm_func, llm_model_name=llm_model_name
            )

        return cls(
            path=path,
            driver=driver,
            namespace=namespace,
            graph_name=graph_name,
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
        import json

        # Check if node exists
        records, keys = self.driver.execute_query(
            "MATCH (n:Entity) WHERE n.id = $id RETURN n.id, n.entity_name, n.entity_type, n.description, n.chunk_ids",
            namespace=self.namespace,
            graph=self.graph_name,
            parameters={"id": node.id},
        )

        if not records:
            # Node doesn't exist, insert it
            try:
                description_json = json.dumps(node.metadata.description or [])
                chunk_ids_json = json.dumps(node.metadata.chunk_ids or [])
                
                self.driver.execute_query(
                    "INSERT INTO Entity VALUES ($id, $name, $type, $desc, $chunks)",
                    namespace=self.namespace,
                    graph=self.graph_name,
                    parameters={
                        "id": node.id,
                        "name": node.page_content,
                        "type": getattr(node.metadata, "entity_type", "UNKNOWN"),
                        "desc": description_json,
                        "chunks": chunk_ids_json,
                    },
                )
                return None
            except Exception as e:
                raise e
        else:
            # Node exists, check description
            existing_node = records[0]
            latest_description = json.loads(existing_node.get("n.description", "[]"))
            
            current_description = node.metadata.description or []
            if record_description == latest_description:
                # Update existing node using Cypher SET syntax
                description_json = json.dumps(current_description)
                chunk_ids_json = json.dumps(node.metadata.chunk_ids or [])
                
                self.driver.execute_query(
                    "MATCH (a:Entity) WHERE a.id = $id SET a.entity_name = $name, a.entity_type = $type, a.description = $desc, a.chunk_ids = $chunks",
                    namespace=self.namespace,
                    graph=self.graph_name,
                    parameters={
                        "id": node.id,
                        "name": node.page_content,
                        "type": getattr(node.metadata, "entity_type", "UNKNOWN"),
                        "desc": description_json,
                        "chunks": chunk_ids_json,
                    },
                )
                return None
            elif record_description is None:
                if current_description == latest_description:
                    return None
                else:
                    # require to merge with the latest description
                    return latest_description
            else:
                return latest_description

    async def _merge_node(self, node: Entity, latest_description: List[str]) -> Entity:
        # Directly merge description lists without summarization
        current_descriptions = node.metadata.description or []
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
            import json
            
            # Check if source and target nodes exist
            source_records, _ = self.driver.execute_query(
                "MATCH (n:Entity) WHERE n.id = $id RETURN n.id",
                namespace=self.namespace,
                graph=self.graph_name,
                parameters={"id": relation.source},
            )
            
            target_records, _ = self.driver.execute_query(
                "MATCH (n:Entity) WHERE n.id = $id RETURN n.id",
                namespace=self.namespace,
                graph=self.graph_name,
                parameters={"id": relation.target},
            )
            
            # Only create empty nodes if they don't exist
            if not source_records:
                source_entity = Entity(
                    id=relation.source,
                    page_content=relation.source,  # Use ID as content for now
                    metadata={
                        "entity_type": "UNKNOWN",
                        "description": [],
                        "chunk_ids": [],
                    }
                )
                await self.upsert_node(source_entity)
            
            if not target_records:
                target_entity = Entity(
                    id=relation.target,
                    page_content=relation.target,  # Use ID as content for now
                    metadata={
                        "entity_type": "UNKNOWN",
                        "description": [],
                        "chunk_ids": [],
                    }
                )
                await self.upsert_node(target_entity)
            
            # Insert relation
            properties_json = json.dumps(relation.properties or {})
            relation_type = relation.properties.get("relation_type", "RELATED") if relation.properties else "RELATED"
            
            self.driver.execute_query(
                "INSERT INTO Relation VALUES ($source, $target, $type, $props)",
                namespace=self.namespace,
                graph=self.graph_name,
                parameters={
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation_type,
                    "props": properties_json,
                },
            )
        except Exception as e:
            raise e

    async def query_node(self, node_id: str) -> Entity:
        import json
        
        records, keys = self.driver.execute_query(
            "MATCH (n:Entity) WHERE n.id = $id RETURN n.id, n.entity_name, n.entity_type, n.description, n.chunk_ids",
            namespace=self.namespace,
            graph=self.graph_name,
            parameters={"id": node_id},
        )
        
        if not records:
            raise ValueError(f"Node {node_id} not found")
        
        node_data = records[0]
        
        # Parse JSON fields
        description = json.loads(node_data.get("n.description", "[]"))
        chunk_ids = json.loads(node_data.get("n.chunk_ids", "[]"))
        
        metadata = {
            "entity_type": node_data.get("n.entity_type", "UNKNOWN"),
            "description": description,
            "chunk_ids": chunk_ids,
        }
        
        return Entity(
            id=node_id,
            page_content=node_data.get("n.entity_name", ""),
            metadata=metadata,
        )
    # TODO: the implementation of query_edge is not correct, we are working on the support of edge id now.
    async def query_edge(self, edge_id: str) -> Relation:
        import json
        
        records, keys = self.driver.execute_query(
            "MATCH ()-[r:Relation]->() WHERE r.id = $edge_id RETURN r",
            namespace=self.namespace,
            graph=self.graph_name,
            parameters={"edge_id": edge_id},
        )
        
        if not records:
            raise ValueError(f"Edge {edge_id} not found")
        
        edge_data = records[0]
        properties = json.loads(edge_data.get("properties", "{}"))
        
        return Relation(
            source=edge_data.get("source_id"),
            target=edge_data.get("target_id"),
            properties=properties,
        )

    async def query_one_hop(self, node_id: str) -> (List[Entity], List[Relation]):
        # Query neighbors and edges in one query
        records, _ = self.driver.execute_query(
            "MATCH (n:Entity)-[r:Relation]->(neighbor:Entity) WHERE n.id = $id RETURN n.id, r, neighbor.id, neighbor.entity_name, neighbor.entity_type, neighbor.description, neighbor.chunk_ids",
            namespace=self.namespace,
            graph=self.graph_name,
            parameters={"id": node_id},
        )
        
        neighbors = []
        relations = []
        
        for record in records:
            # Convert neighbor to Entity
            import json
            description = json.loads(record.get("neighbor.description", "[]"))
            chunk_ids = json.loads(record.get("neighbor.chunk_ids", "[]"))
            
            metadata = {
                "entity_type": record.get("neighbor.entity_type", "UNKNOWN"),
                "description": description,
                "chunk_ids": chunk_ids,
            }
            
            neighbor = Entity(
                id=record.get("neighbor.id"),
                page_content=record.get("neighbor.entity_name", ""),
                metadata=metadata,
            )
            neighbors.append(neighbor)
            
            # Convert edge to Relation
            edge_data = record["r"]
            # Extract custom properties (excluding SRC, DST system fields)
            properties = {}
            for key, value in edge_data.items():
                if key not in ["SRC", "DST"]:  # Skip system fields
                    properties[key] = value
            
            relation = Relation(
                source=node_id,  # Use the query node_id as source
                target=record.get("neighbor.id"),  # Use neighbor id as target
                properties=properties,
            )
            relations.append(relation)
        
        return neighbors, relations

    async def dump(self):
        # For Flavius, we don't need to dump to file as data is persisted in the database
        # But we can save metadata about the namespace and graph
        if os.path.dirname(self.path) != "":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        metadata = {
            "namespace": self.namespace,
            "graph_name": self.graph_name,
            "flavius_url": "http://localhost:30000",  # Use default URL
        }
        
        import json
        with open(self.path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str):
        # This method is not fully implemented for Flavius as we need the driver instance
        # In practice, you would need to recreate the driver and reconnect
        import json
        with open(path, "r") as f:
            metadata = json.load(f)
        return metadata

    async def clean_up(self):
        # Close the driver connection
        if hasattr(self, 'driver') and self.driver:
            self.driver.close() 