import asyncio
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa

from hirag_prod._llm import gpt_4o_mini_complete, openai_embedding
from hirag_prod._utils import _limited_gather  # Concurrency Rate Limiting Tool
from hirag_prod.chunk import BaseChunk, FixTokenChunk
from hirag_prod.entity import BaseEntity, VanillaEntity
from hirag_prod.loader import load_document
from hirag_prod.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

# Log Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("HiRAG").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("HiRAG")


@dataclass
class HiRAG:
    # Chunk documents
    chunker: BaseChunk = field(
        default_factory=lambda: FixTokenChunk(chunk_size=1200, chunk_overlap=200)
    )

    # Entity extraction
    entity_extractor: BaseEntity = field(
        default_factory=lambda: VanillaEntity.create(extract_func=gpt_4o_mini_complete)
    )

    # Storage
    vdb: BaseVDB = field(default=None)
    gdb: BaseGDB = field(
        default_factory=lambda: NetworkXGDB.create(
            path="kb/hirag.gpickle",
            llm_func=gpt_4o_mini_complete,
        )
    )

    # Parallel Pool & Concurrency Rate Limiting Parameters
    _chunk_pool: ProcessPoolExecutor | None = None
    chunk_upsert_concurrency: int = 4
    entity_upsert_concurrency: int = 4
    relation_upsert_concurrency: int = 2

    async def initialize_tables(self):
        # Initialize the chunks table
        try:
            self.chunks_table = await self.vdb.db.open_table("chunks")
        except Exception as e:
            if str(e) == "Table 'chunks' was not found":
                self.chunks_table = await self.vdb.db.create_table(
                    "chunks",
                    schema=pa.schema(
                        [
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field("type", pa.string()),
                            pa.field("filename", pa.string()),
                            pa.field("page_number", pa.int8()),
                            pa.field("uri", pa.string()),
                            pa.field("private", pa.bool_()),
                            pa.field(
                                "chunk_idx", pa.int32()
                            ),  # The index of the chunk in the document
                            pa.field(
                                "document_id", pa.string()
                            ),  # The id of the document that the chunk is from
                            pa.field("vector", pa.list_(pa.float32(), 1536)),
                        ]
                    ),
                )
            else:
                raise e
        try:
            self.entities_table = await self.vdb.db.open_table("entities")
        except Exception as e:
            if str(e) == "Table 'entities' was not found":
                self.entities_table = await self.vdb.db.create_table(
                    "entities",
                    schema=pa.schema(
                        [
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field("vector", pa.list_(pa.float32(), 1536)),
                            pa.field("entity_type", pa.string()),
                            pa.field("description", pa.string()),
                            pa.field("chunk_ids", pa.list_(pa.string())),
                        ]
                    ),
                )
            else:
                raise e

    @classmethod
    async def create(cls, **kwargs):
        if kwargs.get("vdb") is None:
            lancedb = await LanceDB.create(
                embedding_func=openai_embedding,
                db_url="kb/hirag.db",
                strategy_provider=RetrievalStrategyProvider(),
            )
            kwargs["vdb"] = lancedb
        instance = cls(**kwargs)
        await instance.initialize_tables()
        return instance

    @classmethod
    def _get_pool(cls) -> ProcessPoolExecutor:
        if cls._chunk_pool is None:
            ctx = multiprocessing.get_context("spawn")
            cpu = os.cpu_count() or 1
            cls._chunk_pool = ProcessPoolExecutor(max_workers=cpu, mp_context=ctx)
        return cls._chunk_pool

    async def _process_document(self, document):
        """
        Single-document processing: chunk  upsert chunks  extract entities & upsert  extract relations & upsert
        """
        loop = asyncio.get_running_loop()
        pool = self._get_pool()
        # Chunking executed in process pool
        chunks = await loop.run_in_executor(pool, self.chunker.chunk, document)

        # Concurrently upsert chunks
        chunk_coros = [
            self.vdb.upsert_text(
                text_to_embed=chunk.page_content,
                properties={
                    "document_key": chunk.id,
                    "text": chunk.page_content,
                    **chunk.metadata.__dict__,
                },
                table_name="chunks",
                mode="overwrite",
            )
            for chunk in chunks
        ]
        await _limited_gather(chunk_coros, self.chunk_upsert_concurrency)

        # Entity extraction & upsert
        entities = await self.entity_extractor.entity(chunks)
        entity_coros = [
            self.vdb.upsert_text(
                text_to_embed=ent.metadata.description,
                properties={
                    "document_key": ent.id,
                    "text": ent.page_content,
                    **ent.metadata.__dict__,
                },
                table_name="entities",
                mode="overwrite",
            )
            for ent in entities
        ]
        await _limited_gather(entity_coros, self.entity_upsert_concurrency)

        # Relation extraction & upsert
        relations = await self.entity_extractor.relation(chunks, entities)
        relation_coros = [self.gdb.upsert_relation(rel) for rel in relations]
        await _limited_gather(relation_coros, self.relation_upsert_concurrency)

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ):
        # Load the document from the document path
        logger.info(f"Loading the document from the document path: {document_path}")

        start_total = time.perf_counter()
        documents = await asyncio.to_thread(
            load_document,
            document_path,
            content_type,
            document_meta,
            loader_configs,
            loader_type="mineru",
        )
        logger.info(f"Loaded {len(documents)} documents")

        # Concurrently process all documents
        tasks = [self._process_document(doc) for doc in documents]
        await asyncio.gather(*tasks)

        # dump the graph
        await self.gdb.dump()

        total = time.perf_counter() - start_total
        logger.info(f"Total pipeline time: {total:.3f}s")

    async def query_chunks(self, query: str, topk: int = 10):
        return await self.vdb.query(
            query=query,
            table=self.chunks_table,
            topk=topk,
            require_access="public",
            columns_to_select=["text", "document_key", "filename", "private"],
            distance_threshold=100,  # a very high threshold to ensure all results are returned
        )

    async def query_entities(self, query: str, topk: int = 10):
        return await self.vdb.query(
            query=query,
            table=self.entities_table,
            topk=topk,
            columns_to_select=["text", "document_key", "entity_type", "description"],
            distance_threshold=100,  # a very high threshold to ensure all results are returned
        )

    async def query_relations(self, query: str, topk: int = 10):
        # search the entities
        recall_entities = await self.query_entities(query, topk)
        recall_entities = [entity["document_key"] for entity in recall_entities]
        # search the relations
        recall_neighbors = []
        recall_edges = []
        for entity in recall_entities:
            neighbors, edges = await self.gdb.query_one_hop(entity)
            recall_neighbors.extend(neighbors)
            recall_edges.extend(edges)
        return recall_neighbors, recall_edges

    async def query_all(self, query: str, topk: int = 10):
        # search chunks
        recall_chunks = await self.query_chunks(query, topk)
        # search entities
        recall_entities = await self.query_entities(query, topk)
        # search relations
        recall_neighbors, recall_edges = await self.query_relations(query, topk)
        # merge the results
        return {
            "chunks": recall_chunks,
            "entities": recall_entities,
            "neighbors": recall_neighbors,
            "edges": recall_edges,
        }
