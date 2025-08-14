import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from dotenv import load_dotenv

from ._llm import (
    ChatCompletion,
    EmbeddingService,
    LocalChatService,
    LocalEmbeddingService,
    create_chat_service,
    create_embedding_service,
)
from ._utils import _limited_gather_with_factory
from .chunk import BaseChunk, FixTokenChunk
from .entity import BaseKG, VanillaKG
from .loader import load_document
from .loader.chunk_split import (
    chunk_docling_document,
    chunk_langchain_document,
)
from .parser import (
    DictParser,
    ReferenceParser,
)
from .prompt import PROMPTS
from .resume_tracker import JobStatus, ResumeTracker
from .schema import Relation
from .storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

load_dotenv("/chatbot/.env")

# ============================================================================
# Constants and Default Values
# ============================================================================

# Database Configuration
DEFAULT_VECTOR_DB_PATH = "kb/hirag.db"
DEFAULT_GRAPH_DB_PATH = "kb/hirag.gpickle"

# Redis Configuration for resume tracker
DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/2")
DEFAULT_REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "hirag")

# Model Configuration
DEFAULT_LLM_MODEL_NAME = "gpt-4o"
DEFAULT_MAX_TOKENS = 16000  # Default max tokens for LLM

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# Batch Processing Configuration
DEFAULT_EMBEDDING_BATCH_SIZE = 1000  # Optimized for both OpenAI and local services
DEFAULT_ENTITY_UPSERT_CONCURRENCY = 32
DEFAULT_RELATION_UPSERT_CONCURRENCY = 32

# Retry Configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# Reference Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # Default threshold for similarity, only shows references with similarity above this value
DEFAULT_SIMILARITY_MAX_DIFFERENCE = 0.15  # If found a most similar reference already, only accept other references with similarity having this difference or less
DEFAULT_MAX_REFERENCES = 3  # Maximum number of references to return

SUPPORTED_LANGUAGES = ["en", "cn"]  # Supported languages for generation

# Vector and Schema Configuration
try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
except ValueError as e:
    raise ValueError(f"EMBEDDING_DIMENSION must be an integer: {e}")

# Query and Operation Constants
MAX_CHUNK_IDS_PER_QUERY = 10
DEFAULT_QUERY_TOPK = 10
DEFAULT_QUERY_TOPN = 5
DEFAULT_LINK_TOP_K = 30
DEFAULT_PASSAGE_NODE_WEIGHT = 1.0
DEFAULT_PAGERANK_DAMPING = 0.85

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HiRAG")


# ============================================================================
# Exception Definitions
# ============================================================================


class HiRAGException(Exception):
    """HiRAG base exception class"""


class DocumentProcessingError(HiRAGException):
    """Document processing exception"""


class KGConstructionError(HiRAGException):
    """Knowledge graph construction exception"""


class StorageError(HiRAGException):
    """Storage exception"""


# ============================================================================
# Configuration Management
# ============================================================================


@dataclass
class HiRAGConfig:
    """HiRAG system configuration"""

    # Database configuration
    vector_db_path: str = DEFAULT_VECTOR_DB_PATH
    graph_db_path: str = DEFAULT_GRAPH_DB_PATH

    # Resume tracker configuration (Redis-based)
    redis_url: str = DEFAULT_REDIS_URL
    redis_key_prefix: str = DEFAULT_REDIS_KEY_PREFIX

    # Model configuration
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    llm_max_tokens: int = DEFAULT_MAX_TOKENS  # Default max tokens for LLM
    llm_timeout: float = 30.0  # Default timeout for LLM requests

    # Chunking configuration
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    # Batch processing configuration
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    entity_upsert_concurrency: int = DEFAULT_ENTITY_UPSERT_CONCURRENCY
    relation_upsert_concurrency: int = DEFAULT_RELATION_UPSERT_CONCURRENCY

    # Retry configuration
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY


# ============================================================================
# Metrics and Monitoring
# ============================================================================


@dataclass
class ProcessingMetrics:
    """Processing metrics"""

    total_chunks: int = 0
    processed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    processing_time: float = 0.0
    error_count: int = 0
    job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "processing_time": self.processing_time,
            "error_count": self.error_count,
            "job_id": self.job_id,
        }


class MetricsCollector:
    """Metrics collector"""

    def __init__(self):
        self.metrics = ProcessingMetrics()
        self.operation_times: Dict[str, float] = {}

    @asynccontextmanager
    async def track_operation(self, operation: str):
        """Track operation execution time"""
        start = time.perf_counter()
        try:
            logger.info(f"🚀 Starting {operation}")
            yield
            duration = time.perf_counter() - start
            self.operation_times[operation] = duration
            logger.info(f"✅ Completed {operation} in {duration:.3f}s")
        except Exception as e:
            self.metrics.error_count += 1
            duration = time.perf_counter() - start
            logger.error(f"❌ Failed {operation} after {duration:.3f}s: {e}")
            raise


# ============================================================================
# Utility Functions
# ============================================================================


def retry_async(
    max_retries: int = DEFAULT_MAX_RETRIES, delay: float = DEFAULT_RETRY_DELAY
):
    """Async retry decorator"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        break
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def validate_input(document_path: str, content_type: str) -> None:
    """Validate input parameters"""
    if not document_path or not isinstance(document_path, str):
        raise ValueError("document_path must be a non-empty string")

    if not Path(document_path).exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    if not content_type or not isinstance(content_type, str):
        raise ValueError("content_type must be a non-empty string")


# ============================================================================
# Storage Manager
# ============================================================================


class StorageManager:
    """Unified manager for vector database and graph database operations"""

    def __init__(self, vdb: BaseVDB, gdb: BaseGDB):
        self.vdb = vdb
        self.gdb = gdb
        self.chunks_table = None
        self.relations_table = None

    async def initialize(self) -> None:
        """Initialize storage tables"""
        await self._initialize_chunks_table()
        await self._initialize_relations_table()

    async def _initialize_chunks_table(self) -> None:
        """Initialize chunks table"""
        try:
            self.chunks_table = await self.vdb.db.open_table("chunks")
        except Exception as e:
            if "was not found" in str(e):
                self.chunks_table = await self.vdb.db.create_table(
                    "chunks",
                    schema=pa.schema(
                        [
                            pa.field("knowledge_base_id", pa.string()),
                            pa.field("workspace_id", pa.string()),
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field("type", pa.string()),
                            pa.field("filename", pa.string()),
                            pa.field("page_number", pa.int32()),
                            pa.field("uri", pa.string()),
                            pa.field("private", pa.bool_()),
                            pa.field("chunk_idx", pa.int32()),
                            pa.field("document_id", pa.string()),
                            pa.field("chunk_type", pa.string()),
                            # Optional metadata fields from ChunkMetadata
                            pa.field("page_image_url", pa.string()),
                            pa.field("page_width", pa.float32()),
                            pa.field("page_height", pa.float32()),
                            pa.field("x_0", pa.float32()),
                            pa.field("y_0", pa.float32()),
                            pa.field("x_1", pa.float32()),
                            pa.field("y_1", pa.float32()),
                            pa.field(
                                "vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)
                            ),
                            pa.field("uploaded_at", pa.timestamp("us")),
                        ]
                    ),
                )
            else:
                raise StorageError(f"Failed to initialize chunks table: {e}")

    async def _initialize_relations_table(self) -> None:
        """Initialize relations table"""
        try:
            self.relations_table = await self.vdb.db.open_table("relations")
        except Exception as e:
            if "was not found" in str(e):
                self.relations_table = await self.vdb.db.create_table(
                    "relations",
                    schema=pa.schema(
                        [
                            pa.field("source", pa.string()),
                            pa.field("target", pa.string()),
                            pa.field("description", pa.string()),
                            pa.field("file_name", pa.string()),
                            pa.field(
                                "vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)
                            ),
                            pa.field("knowledge_base_id", pa.string()),
                            pa.field("workspace_id", pa.string()),
                        ]
                    ),
                )
            else:
                raise StorageError(f"Failed to initialize relations table: {e}")

    @retry_async(max_retries=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY)
    async def upsert_chunks(self, chunks: List[BaseChunk]) -> None:
        """Batch insert chunks"""
        # TODO: Implement intelligent batching based on chunk size and content complexity
        if not chunks:
            return

        try:
            texts_to_embed = [chunk.page_content for chunk in chunks]
            properties_list = [
                {
                    "document_key": chunk.id,
                    "text": chunk.page_content,
                    **chunk.metadata.__dict__,
                }
                for chunk in chunks
            ]

            await self.vdb.upsert_texts(
                texts_to_embed=texts_to_embed,
                properties_list=properties_list,
                table=self.chunks_table,
                mode="append",
            )
        except Exception as e:
            raise StorageError(f"Failed to upsert chunks: {e}")

    @retry_async(max_retries=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY)
    async def upsert_relations_to_vdb(self, relations: List[Relation]) -> None:
        """Batch insert relations to vector database"""
        if not relations:
            return

        try:
            filtered_relations = [
                relation
                for relation in relations
                if not relation.source.startswith("chunk-")
            ]

            if not filtered_relations:
                return

            texts_to_embed = [
                relation.properties.get("description", "")
                for relation in filtered_relations
            ]
            properties_list = [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "description": relation.properties.get("description", ""),
                    "file_name": relation.properties.get("file_name", ""),
                    "knowledge_base_id": relation.properties.get(
                        "knowledge_base_id", ""
                    ),
                    "workspace_id": relation.properties.get("workspace_id", ""),
                }
                for relation in filtered_relations
            ]

            await self.vdb.upsert_texts(
                texts_to_embed=texts_to_embed,
                properties_list=properties_list,
                table=self.relations_table,
                mode="append",
            )
        except Exception as e:
            raise StorageError(f"Failed to upsert relations: {e}")

    async def get_existing_chunks(self, uri: str) -> List[str]:
        """Get existing chunk IDs"""
        try:
            existing_data = (
                await self.chunks_table.query().where(f"uri == '{uri}'").to_list()
            )
            return [chunk["document_key"] for chunk in existing_data]
        except Exception as e:
            logger.warning(f"Failed to get existing chunks: {e}")
            return []

    async def health_check(self) -> Dict[str, bool]:
        """Health check"""
        health = {}

        try:
            # Check vector database
            await self.vdb.db.table_names()
            health["vdb"] = True
        except Exception:
            health["vdb"] = False

        try:
            # Check graph database
            await self.gdb.health_check() if hasattr(self.gdb, "health_check") else None
            health["gdb"] = True
        except Exception:
            health["gdb"] = False

        return health

    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            await self.gdb.clean_up()
            await self.vdb.clean_up()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# ============================================================================
# Document processor
# ============================================================================


class DocumentProcessor:
    """Document processor for handling document ingestion pipeline"""

    def __init__(
        self,
        storage: StorageManager,
        chunker: BaseChunk,
        kg_constructor: BaseKG,
        resume_tracker: Optional[object] = None,
        config: Optional[HiRAGConfig] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.storage = storage
        self.chunker = chunker
        self.kg_constructor = kg_constructor
        self.resume_tracker = resume_tracker
        self.config = config or HiRAGConfig()
        self.metrics = metrics or MetricsCollector()

    async def process_document(
        self,
        document_path: str,
        content_type: str,
        with_graph: bool = True,
        document_meta: Optional[Dict] = None,
        loader_configs: Optional[Dict] = None,
        job_id: Optional[str] = None,
        loader_type: Literal["docling", "docling_cloud", "langchain"] = "docling_cloud",
    ) -> ProcessingMetrics:
        """Process a single document"""
        # TODO: Add document preprocessing pipeline for better quality - OCR, cleanup, etc.

        async with self.metrics.track_operation(f"process_document"):
            # Load and chunk document
            chunks = await self._load_and_chunk_document(
                document_path,
                content_type,
                document_meta,
                loader_configs,
            )

            if not chunks:
                logger.warning("⚠️ No chunks created from document")
                # Mark job failed if tracking is enabled
                if self.resume_tracker and job_id:
                    try:
                        await self.resume_tracker.set_job_failed(
                            job_id, "No chunks created from document"
                        )
                    except Exception:
                        pass
                return self.metrics.metrics

            self.metrics.metrics.total_chunks = len(chunks)
            self.metrics.metrics.job_id = job_id or ""

            # Check if document was already completed in a previous session
            if self.resume_tracker:
                document_id = chunks[0].metadata.document_id
                # Update job -> processing with doc info as soon as we know
                if job_id:
                    try:
                        await self.resume_tracker.set_job_processing(
                            job_id=job_id,
                            document_id=document_id,
                            total_chunks=len(chunks),
                        )
                    except Exception:
                        pass
                if self.resume_tracker.is_document_already_completed(document_id):
                    logger.info(
                        "🎉 Document already fully processed in previous session!"
                    )
                    if job_id:
                        try:
                            await self.resume_tracker.set_job_completed(job_id)
                        except Exception:
                            pass
                    return self.metrics.metrics
                else:
                    document_uri = chunks[0].metadata.uri
                    self.resume_tracker.register_chunks(
                        chunks, document_id, document_uri
                    )

            # Process chunks
            await self._process_chunks(chunks)
            # Update job progress for processed chunks
            if self.resume_tracker and job_id:
                try:
                    await self.resume_tracker.set_job_progress(
                        job_id,
                        processed_chunks=self.metrics.metrics.processed_chunks,
                    )
                except Exception:
                    pass

            # Process graph data
            if with_graph:
                await self._construct_kg(chunks)
                # Update job progress for entity/relation totals
                if self.resume_tracker and job_id:
                    try:
                        await self.resume_tracker.set_job_progress(
                            job_id,
                            total_entities=self.metrics.metrics.total_entities,
                            total_relations=self.metrics.metrics.total_relations,
                        )
                    except Exception:
                        pass

            # Mark as complete
            if self.resume_tracker:
                self.resume_tracker.mark_document_completed(
                    chunks[0].metadata.document_id
                )
                if job_id:
                    try:
                        await self.resume_tracker.set_job_completed(job_id)
                    except Exception:
                        pass

            return self.metrics.metrics

    async def _load_and_chunk_document(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[Dict],
        loader_configs: Optional[Dict],
    ) -> List[BaseChunk]:
        """Load and chunk document"""
        # TODO: Add parallel processing for multi-file documents and large files
        async with self.metrics.track_operation("load_and_chunk"):
            try:
                if content_type == "text/plain":
                    _, langchain_doc = await asyncio.to_thread(
                        load_document,
                        document_path,
                        content_type,
                        document_meta,
                        loader_configs,
                        loader_type="langchain",
                    )
                    chunks = chunk_langchain_document(langchain_doc)
                else:
                    docling_doc, doc_md = await asyncio.to_thread(
                        load_document,
                        document_path,
                        content_type,
                        document_meta,
                        loader_configs,
                        loader_type="docling",
                    )
                    chunks = chunk_docling_document(docling_doc, doc_md)

                logger.info(
                    f"📄 Created {len(chunks)} chunks from document {document_path}"
                )
                return chunks

            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to load document {document_path}: {e}"
                )

    async def _process_chunks(self, chunks: List[BaseChunk]) -> None:
        """Process chunks for vector storage"""
        async with self.metrics.track_operation("process_chunks"):
            # Get chunks that need processing
            pending_chunks = await self._get_pending_chunks(chunks)

            if not pending_chunks:
                logger.info("⏭️ All chunks already processed")
                return

            logger.info(f"📤 Processing {len(pending_chunks)} pending chunks...")

            # Batch storage
            await self.storage.upsert_chunks(pending_chunks)
            self.metrics.metrics.processed_chunks += len(pending_chunks)

            logger.info(f"✅ Processed {len(pending_chunks)} chunks")

    async def _get_pending_chunks(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """Get chunks that need processing"""
        if not chunks:
            return []

        if self.resume_tracker:
            # Check for existing chunks in vector database
            uri = chunks[0].metadata.uri
            existing_chunk_ids = await self.storage.get_existing_chunks(uri)
            return [chunk for chunk in chunks if chunk.id not in existing_chunk_ids]

        return chunks

    async def _construct_kg(self, chunks: List[BaseChunk]) -> None:
        """Construct knowledge graph from chunks"""
        logger.info(f"🔍 Constructing knowledge graph from {len(chunks)} chunks...")

        try:
            entities, relations = await self.kg_constructor.construct_kg(chunks)

            # Store entities only to graph database (not to vector database)
            if entities:
                await self.storage.gdb.upsert_nodes(
                    entities, concurrency=self.config.entity_upsert_concurrency
                )
                self.metrics.metrics.total_entities += len(entities)

            # Store relations to both graph database and vector database
            if relations:
                # Store to graph database for graph analysis
                gdb_relation_factories = [
                    lambda rel=rel: self.storage.gdb.upsert_relation(rel)
                    for rel in relations
                ]
                await _limited_gather_with_factory(
                    gdb_relation_factories, self.config.relation_upsert_concurrency
                )

                # Store to vector database for semantic search
                await self.storage.upsert_relations_to_vdb(relations)

                self.metrics.metrics.total_relations += len(relations)

            logger.info(
                f"✅ Extracted and stored {len(entities)} entities and {len(relations)} relations"
            )

        except Exception as e:
            raise KGConstructionError(f"Failed to construct knowledge graph: {e}")


# ============================================================================
# Query Service
# ============================================================================


class QueryService:
    """Query service"""

    def __init__(self, storage: StorageManager):
        self.storage = storage

    async def query_chunks(
        self,
        query: str,
        topk: int = DEFAULT_QUERY_TOPK,
        topn: int = DEFAULT_QUERY_TOPN,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query chunks"""
        return await self.storage.vdb.query(
            query=query,
            table=self.storage.chunks_table,
            topk=topk,
            topn=topn,
            columns_to_select=[
                "text",
                "uri",
                "filename",
                "private",
                "uploaded_at",
                "document_key",
            ],
            rerank=rerank,
        )

    async def recall_chunks(
        self,
        query: str,
        topk: int = DEFAULT_QUERY_TOPK,
        topn: int = DEFAULT_QUERY_TOPN,
        rerank: bool = True,
    ) -> Dict[str, Any]:
        """Recall chunks and return both raw results and extracted chunk_ids.

        Args:
            query: Query string.
            topk: Number of results to return.
            topn: Number of results to rerank.
            rerank: Whether to rerank the results.

        Returns:
            Dict with keys:
                - "chunks": raw chunk search results
                - "chunk_ids": list of document_key values
        """
        chunks = await self.query_chunks(query, topk=topk, topn=topn, rerank=rerank)
        chunk_ids = [c.get("document_key") for c in chunks if c.get("document_key")]
        return {"chunks": chunks, "chunk_ids": chunk_ids}

    async def query_triplets(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> List[Dict[str, Any]]:
        """Query relations from vector database using semantic search"""
        return await self.storage.vdb.query(
            query=query,
            table=self.storage.relations_table,
            topk=topk,
            topn=topn,
            columns_to_select=["source", "target", "description", "file_name"],
            rerank=False,
        )

    async def recall_triplets(
        self,
        query: str,
        topk: int = DEFAULT_QUERY_TOPK,
        topn: int = DEFAULT_QUERY_TOPN,
    ) -> Dict[str, Any]:
        """Recall triplets and return both raw results and aggregated entity_ids.

        Args:
            query: Query string.
            topk: Number of results to return.
            topn: Optional rerank pool size for relations (forwarded to underlying query).

        Returns:
            Dict with keys:
                - "relations": raw triplet search results
                - "entity_ids": unique list of entity ids appearing as source/target
        """
        relations = await self.query_triplets(query, topk=topk, topn=topn)
        entity_id_set = set()
        for rel in relations:
            src = rel.get("source")
            tgt = rel.get("target")
            if src:
                entity_id_set.add(src)
            if tgt:
                entity_id_set.add(tgt)
        return {"relations": relations, "entity_ids": list(entity_id_set)}

    async def pagerank_top_chunks(
        self,
        seed_chunk_ids: List[str],
        seed_entity_ids: List[str],
        topk: int,
    ) -> List[Tuple[str, float]]:
        """Delegate personalized PageRank on GDB to get top-k chunk nodes."""
        return await self.storage.gdb.pagerank_top_chunks(
            seed_chunk_ids=seed_chunk_ids, seed_entity_ids=seed_entity_ids, topk=topk
        )

    async def query_chunk_embeddings(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Query chunk embeddings"""
        if not chunk_ids:
            return {}

        res = {}
        try:
            # Query chunk embeddings by keys
            chunk_data = await self.storage.vdb.query_by_keys(
                key_value=chunk_ids,
                key_column="document_key",
                table=self.storage.chunks_table,
                columns_to_select=["document_key", "vector"],
            )

            # chunk data is a list of dicts with 'vector' key
            for chunk in chunk_data:
                if "vector" in chunk and chunk["vector"] is not None:
                    res[chunk["document_key"]] = chunk["vector"]
                else:
                    # Log missing vector data and raise exception
                    logger.warning(f"Chunk {chunk['document_key']} has no vector data")
                    res[chunk["document_key"]] = None

        except Exception as e:
            logger.error(f"Failed to query chunk embeddings: {e}")
            return {}

        return res

    async def get_chunks_by_ids(
        self,
        chunk_ids: List[str],
        columns_to_select: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch chunk rows by document_key list, preserving input order where possible."""
        if not chunk_ids:
            return []
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "filename",
                "private",
                "uploaded_at",
                "document_key",
            ]
        rows = await self.storage.vdb.query_by_keys(
            key_value=chunk_ids,
            key_column="document_key",
            table=self.storage.chunks_table,
            columns_to_select=columns_to_select,
        )
        # Build map for stable ordering
        by_id = {row.get("document_key"): row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def dual_recall_with_pagerank(
        self,
        query: str,
        topk: int = DEFAULT_QUERY_TOPK,
        topn: int = DEFAULT_QUERY_TOPN,
        link_top_k: int = DEFAULT_LINK_TOP_K,
        passage_node_weight: float = DEFAULT_PASSAGE_NODE_WEIGHT,
        damping: float = DEFAULT_PAGERANK_DAMPING,
    ) -> Dict[str, Any]:
        """Two-path retrieval + PageRank fusion.

        - Recall chunks to form passage reset weights
        - Recall triplets to form phrase (entity) reset weights with frequency penalty
        - Build reset = phrase_weights + passage_weights and run Personalized PageRank
        - If no facts, fall back to DPR order (query rerank order)
        """
        # Path 1: chunk recall (rerank happens in VDB query)
        chunk_recall = await self.recall_chunks(query, topk=topk, topn=topn)
        query_chunks = chunk_recall["chunks"]
        query_chunk_ids = chunk_recall["chunk_ids"]

        # Path 2: triplet recall -> entity seeds
        triplet_recall = await self.recall_triplets(query, topk=topk, topn=topn)
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
                    chunk_ids = (
                        node.metadata.get("chunk_ids", [])
                        if hasattr(node, "metadata")
                        else []
                    )
                    return len(chunk_ids) if isinstance(chunk_ids, list) else 0
                except Exception:
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
        )
        pr_ids = [cid for cid, _ in pr_ranked]

        pr_rows = await self.get_chunks_by_ids(pr_ids)
        pr_score_map = {cid: score for cid, score in pr_ranked}
        for row in pr_rows:
            row["pagerank_score"] = pr_score_map.get(row.get("document_key"), 0.0)

        return {
            "pagerank": pr_rows,
            "query_top": query_chunks,
        }

    async def query(self, query: str) -> Dict[str, Any]:
        """Query Strategy (default: dual_recall_with_pagerank)"""
        result = await self.dual_recall_with_pagerank(
            query=query,
            topk=DEFAULT_QUERY_TOPK,
            topn=DEFAULT_QUERY_TOPN,
        )
        result["chunks"] = (
            result.get("pagerank")
            if result.get("pagerank")
            else result.get("query_top", [])
        )
        return result


# ============================================================================
# Main HiRAG class
# ============================================================================


@dataclass
class HiRAG:
    """
    Hierarchical Retrieval-Augmented Generation (HiRAG) system

    Simplified main interface, coordinating the work of all components
    """

    config: HiRAGConfig = field(default_factory=HiRAGConfig)

    # Components (lazy initialization)
    _storage: Optional[StorageManager] = field(default=None, init=False)
    _processor: Optional[DocumentProcessor] = field(default=None, init=False)
    _query_service: Optional[QueryService] = field(default=None, init=False)
    _metrics: Optional[MetricsCollector] = field(default=None, init=False)
    _kg_constructor: Optional[VanillaKG] = field(default=None, init=False)
    _language: str = field(default=SUPPORTED_LANGUAGES[0], init=False)

    # Services
    chat_service: Optional[Union[ChatCompletion, LocalChatService]] = field(
        default=None, init=False
    )
    embedding_service: Optional[Union[EmbeddingService, LocalEmbeddingService]] = field(
        default=None, init=False
    )

    @classmethod
    async def create(
        cls,
        config: Optional[HiRAGConfig] = None,
        vector_db_path: Optional[str] = None,
        graph_db_path: Optional[str] = None,
        **kwargs,
    ) -> "HiRAG":
        """Create HiRAG instance"""
        config = config or HiRAGConfig()

        # Override the default database paths if provided
        config.vector_db_path = (
            vector_db_path if vector_db_path else DEFAULT_VECTOR_DB_PATH
        )
        config.graph_db_path = graph_db_path if graph_db_path else DEFAULT_GRAPH_DB_PATH

        instance = cls(config=config)
        await instance._initialize(**kwargs)
        return instance

    async def set_language(self, language: str) -> None:
        """Set the language for the HiRAG instance"""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {SUPPORTED_LANGUAGES}"
            )

        self._language = language
        self._kg_constructor.set_language(language)

        logger.info(f"Language set to {self._language}")

    async def set_db_paths(self, vector_db_path: str, graph_db_path: str) -> None:
        """Set the database paths for the HiRAG instance"""
        self.config.vector_db_path = vector_db_path
        self.config.graph_db_path = graph_db_path

        # Reinitialize storage with new paths
        await self._reinitialize_storage()

        logger.info(
            f"Database paths updated - VDB: {self.config.vector_db_path}, GDB: {self.config.graph_db_path}"
        )

    async def _reinitialize_storage(self) -> None:
        """Reinitialize storage components with current configuration"""
        if not self.chat_service or not self.embedding_service:
            raise HiRAGException(
                "Services not initialized - cannot reinitialize storage"
            )

        vdb = await LanceDB.create(
            embedding_func=self.embedding_service.create_embeddings,
            db_url=self.config.vector_db_path,
            strategy_provider=RetrievalStrategyProvider(),
        )

        gdb = NetworkXGDB.create(
            path=self.config.graph_db_path,
            llm_func=self.chat_service.complete,
        )

        # Initialize new storage manager
        self._storage = StorageManager(vdb, gdb)
        await self._storage.initialize()

        # Update dependent components
        if self._processor:
            self._processor.storage = self._storage
        if self._query_service:
            self._query_service.storage = self._storage

    # TODO: Enable initializing all resources (embedding_service, chat_service, vdb, gdb, etc.)
    # outside of the HiRAG class for better management of resources
    async def _initialize(self, **kwargs) -> None:
        """Initialize all components"""
        # Initialize services
        self.chat_service = create_chat_service()
        self.embedding_service = create_embedding_service(
            default_batch_size=self.config.embedding_batch_size
        )

        # Initialize storage
        if kwargs.get("vdb") is None:
            vdb = await LanceDB.create(
                embedding_func=self.embedding_service.create_embeddings,
                db_url=self.config.vector_db_path,
                strategy_provider=RetrievalStrategyProvider(),
            )
        else:
            vdb = kwargs["vdb"]

        if kwargs.get("gdb") is None:
            gdb = NetworkXGDB.create(
                path=self.config.graph_db_path,
                llm_func=self.chat_service.complete,
            )
        else:
            gdb = kwargs["gdb"]

        self._storage = StorageManager(vdb, gdb)
        await self._storage.initialize()

        # Initialize other components
        chunker = FixTokenChunk(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        self._kg_constructor = VanillaKG.create(
            extract_func=self.chat_service.complete,
            llm_model_name=self.config.llm_model_name,
            language=self._language,
        )

        # Initialize resume tracker
        resume_tracker = kwargs.get("resume_tracker")
        if resume_tracker is None:
            resume_tracker = ResumeTracker(
                redis_url=self.config.redis_url, key_prefix=self.config.redis_key_prefix
            )
            logger.info("Using Redis-based resume tracker")

        # Initialize components
        self._metrics = MetricsCollector()
        self._processor = DocumentProcessor(
            storage=self._storage,
            chunker=chunker,
            kg_constructor=self._kg_constructor,
            resume_tracker=resume_tracker,
            config=self.config,
            metrics=self._metrics,
        )
        self._query_service = QueryService(self._storage)

    # ========================================================================
    # Chat service methods
    # ========================================================================

    # Helper function for similarity calcuation
    async def calculate_similarity(
        self, sentence_embedding: List[float], references: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        """Calculate similarity between sentence embedding and reference embeddings"""
        from sklearn.metrics.pairwise import (
            cosine_similarity as sklearn_cosine_similarity,
        )

        similar_refs = []
        for entity_key, embedding in references.items():
            if embedding is not None:
                similarity = sklearn_cosine_similarity(
                    [sentence_embedding], [embedding]
                )[0][0]
                similar_refs.append(
                    {"document_key": entity_key, "similarity": similarity}
                )
        return similar_refs

    async def chat_complete(self, prompt: str, **kwargs: Any) -> str:
        """Chat with the user"""
        if not self.chat_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        try:
            response = await self.chat_service.complete(
                prompt=prompt,
                **kwargs,
            )
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise HiRAGException("Chat completion failed") from e

    async def generate_summary(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate summary from chunks"""
        DEBUG = False  # Set to True for debugging output

        if not self.chat_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        logger.info("🚀 Starting summary generation")
        start_time = time.perf_counter()

        try:
            prompt = PROMPTS["summary_all_" + self._language]

            placeholder = PROMPTS["REFERENCE_PLACEHOLDER"]

            parser = DictParser()

            # Should use parser to better format the data
            data = "Chunks:\n" + parser.parse_list_of_dicts(chunks, "table") + "\n\n"

            prompt = prompt.format(
                data=data, max_report_length="5000", reference_placeholder=placeholder
            )

            try:
                summary = await self.chat_complete(
                    prompt=prompt,
                    max_tokens=self.config.llm_max_tokens,
                    timeout=self.config.llm_timeout,
                    model=self.config.llm_model_name,
                )
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                raise HiRAGException("Summary generation failed") from e

            if DEBUG:
                print("\n\n\nGenerated Summary:\n", summary)

            # Find all sentences that contain the placeholder
            ref_parser = ReferenceParser()

            ref_sentences = await ref_parser.parse_references(summary, placeholder)

            if DEBUG:
                print("\n\n\nReference Sentences:\n", "\n".join(ref_sentences))

            # for each sentence, do a query and find the best matching document key to find the referenced chunk
            result = []

            chunk_keys = [c["document_key"] for c in chunks]

            # Generate embeddings for each reference sentence
            if not ref_sentences:
                logger.warning("No reference sentences found in summary")
                return summary

            sentence_embeddings = await self.embedding_service.create_embeddings(
                texts=ref_sentences
            )
            chunk_embeddings = await self._query_service.query_chunk_embeddings(
                chunk_keys
            )

            for sentence, sentence_embedding in zip(ref_sentences, sentence_embeddings):
                # If the sentence is empty, continue
                if not sentence.strip():
                    result.append("")
                    continue

                similar_chunks = await self.calculate_similarity(
                    sentence_embedding, chunk_embeddings
                )

                if DEBUG:
                    print(
                        "\n\n\nSimilar Chunks for Sentence:",
                        sentence,
                        "\n",
                        similar_chunks,
                    )

                # Sort by similarity
                reference_list = similar_chunks
                reference_list.sort(key=lambda x: x["similarity"], reverse=True)

                # If no similar chunks found, append empty string
                if not reference_list:
                    result.append("")
                    continue

                reference_threshold = DEFAULT_SIMILARITY_THRESHOLD
                max_similarity_difference = DEFAULT_SIMILARITY_MAX_DIFFERENCE

                # If we have a most similar reference, only accept others with similarity having this difference or less
                most_similar = reference_list[0]
                if most_similar["similarity"] > reference_threshold:
                    reference_threshold = max(
                        most_similar["similarity"] - max_similarity_difference,
                        reference_threshold,
                    )

                # Filter references based on similarity threshold
                filtered_references = [
                    ref
                    for ref in reference_list
                    if ref["similarity"] >= reference_threshold
                ]

                # Limit the number of references to DEFAULT_MAX_REFERENCES
                filtered_references = filtered_references[:DEFAULT_MAX_REFERENCES]

                # If no references found, append empty string
                if not filtered_references:
                    result.append([])
                    continue

                # Separate the references by "," and sort by type as primary, similarity as secondary
                filtered_references.sort(
                    key=lambda x: (x["document_key"].split("_")[0], -x["similarity"])
                )

                # Append the document keys to the result
                if DEBUG:
                    print(
                        "\n\n\nFiltered References for Sentence:",
                        sentence,
                        "\n",
                        filtered_references,
                    )

                if len(filtered_references) == 1:
                    result.append([filtered_references[0]["document_key"]])
                else:
                    # Join the document keys with ", "
                    result.append([ref["document_key"] for ref in filtered_references])

            format_prompt = PROMPTS["REFERENCE_FORMAT"]

            # fill the summary by ref chunks
            summary = await ref_parser.fill_placeholders(
                text=summary,
                references=result,
                reference_placeholder=placeholder,
                format_prompt=format_prompt,
            )

            if DEBUG:
                print("\n\n\nFormatted Summary:\n", summary)

            total_time = time.perf_counter() - start_time
            logger.info(f"✅ Summary generation completed in {total_time:.3f}s")

            return summary

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"❌ Summary generation failed after {total_time:.3f}s: {e}")
            raise

    # ========================================================================
    # Public interface methods
    # ========================================================================

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        with_graph: bool = True,
        document_meta: Optional[Dict] = None,
        loader_configs: Optional[Dict] = None,
        job_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        loader_type: Literal["docling", "docling_cloud", "langchain"] = "docling_cloud",
    ) -> ProcessingMetrics:
        """
        Insert document into knowledge base

        Args:
            document_path: document path
            content_type: document type
            with_graph: whether to process graph data (entities and relations)
            document_meta: document metadata
            loader_configs: loader configurations

        Returns:
            ProcessingMetrics: processing metrics
        """
        if not self._processor:
            raise HiRAGException("HiRAG instance not properly initialized")

        logger.info(f"🚀 Starting document processing: {document_path}")
        start_time = time.perf_counter()
        document_meta["knowledge_base_id"] = knowledge_base_id
        document_meta["workspace_id"] = workspace_id
        document_meta["uploaded_at"] = datetime.now().isoformat()
        if job_id and self._processor and self._processor.resume_tracker is not None:
            try:
                await self._processor.resume_tracker.set_job_status(
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    document_uri=(
                        document_meta.get("uri")
                        if isinstance(document_meta, dict)
                        else str(document_path)
                    ),
                    with_graph=with_graph,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize external job {job_id}: {e}")

        try:
            metrics = await self._processor.process_document(
                document_path=document_path,
                content_type=content_type,
                with_graph=with_graph,
                document_meta=document_meta,
                loader_configs=loader_configs,
                job_id=job_id,
                loader_type=loader_type,
            )

            # Save graph state
            if with_graph and self._storage:
                await self._storage.gdb.dump()

            total_time = time.perf_counter() - start_time
            metrics.processing_time = total_time
            logger.info(f"🏁 Total pipeline time: {total_time:.3f}s")

            if job_id and not metrics.job_id:
                metrics.job_id = job_id
            return metrics

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"❌ Document processing failed after {total_time:.3f}s: {e}")
            if (
                self._processor
                and self._processor.resume_tracker is not None
                and job_id
            ):
                try:
                    await self._processor.resume_tracker.set_job_failed(job_id, str(e))
                except Exception:
                    pass
            raise

    async def query_chunks(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> List[Dict[str, Any]]:
        """Query document chunks"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        return await self._query_service.query_chunks(query, topk, topn)

    async def query(self, query: str, summary: bool = False) -> Dict[str, Any]:
        """Query all types of data"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        if summary:
            query_results = await self._query_service.query(query)
            text_summary = await self.generate_summary(
                chunks=query_results["chunks"],
            )
            query_results["summary"] = text_summary
            return query_results
        return await self._query_service.query(query)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self._storage:
            return {"status": "not_initialized"}

        health = await self._storage.health_check()

        return {
            "status": "healthy" if all(health.values()) else "unhealthy",
            "components": health,
            "metrics": self._metrics.metrics.to_dict() if self._metrics else {},
        }

    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        if not self._metrics:
            return {}

        return {
            "metrics": self._metrics.metrics.to_dict(),
            "operation_times": self._metrics.operation_times,
        }

    async def clean_up(self) -> None:
        """Clean up resources"""
        logger.info("🧹 Cleaning up HiRAG resources...")

        try:
            if self._storage:
                await self._storage.cleanup()

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")

    # ========================================================================
    # Context manager support
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.clean_up()

    # ========================================================================
    # Backward compatibility property accessors
    # ========================================================================

    @property
    def chunks_table(self):
        """Backward compatibility: access chunks table"""
        return self._storage.chunks_table if self._storage else None

    @property
    def vdb(self):
        """Backward compatibility: access vector database"""
        return self._storage.vdb if self._storage else None

    @property
    def gdb(self):
        """Backward compatibility: access graph database"""
        return self._storage.gdb if self._storage else None

    # ========================================================================
    # DPR-like recall API
    # ========================================================================

    # TODO: whether to use this?
    async def dpr_recall_chunks(
        self,
        query: str,
        topk: int = DEFAULT_QUERY_TOPK,
        pool_size: int = 500,
    ) -> Dict[str, Any]:
        """Dense Passage Retrieval-style recall using current embeddings and stored vectors.

        Steps:
          - Retrieve a candidate pool without rerank
          - Fetch embeddings of candidates and the query
          - Compute cosine similarities, min-max normalize
          - Return top-k chunk rows with scores and ids
        """
        if not self._query_service or not self.embedding_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        # Step 1: candidate pool (no rerank)
        candidates = await self._query_service.query_chunks(
            query=query, topk=pool_size, topn=None, rerank=False
        )
        candidate_ids = [
            c.get("document_key") for c in candidates if c.get("document_key")
        ]
        if not candidate_ids:
            return {"chunk_ids": [], "scores": [], "chunks": []}

        # Step 2: fetch candidate embeddings and query embedding
        chunk_vec_map = await self._query_service.query_chunk_embeddings(candidate_ids)
        # Filter out None vectors while preserving id order
        filtered_ids = [
            cid for cid in candidate_ids if chunk_vec_map.get(cid) is not None
        ]
        if not filtered_ids:
            return {"chunk_ids": [], "scores": [], "chunks": []}

        chunk_matrix = np.array(
            [chunk_vec_map[cid] for cid in filtered_ids], dtype=np.float32
        )
        query_vec = await self.embedding_service.create_embeddings([query])
        # embedding services return numpy array (n, d); take first row
        if hasattr(query_vec, "shape"):
            query_vec = np.array(query_vec[0], dtype=np.float32)
        else:
            # fallback for list-like
            query_vec = np.array(query_vec[0], dtype=np.float32)

        # Step 3: cosine similarity
        # Normalize rows of chunk_matrix and query vector
        def _l2_normalize(mat: np.ndarray, axis: int) -> np.ndarray:
            denom = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
            denom[denom == 0] = 1.0
            return mat / denom

        chunk_matrix_norm = _l2_normalize(chunk_matrix, axis=1)
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scores = chunk_matrix_norm @ query_vec_norm

        # Min-max normalize
        s_min = float(scores.min())
        s_max = float(scores.max())
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores)

        # Step 4: sort and select top-k
        order = np.argsort(-norm_scores)[: max(0, topk)]
        top_ids = [filtered_ids[i] for i in order]
        top_scores = [float(norm_scores[i]) for i in order]
        top_rows = await self._query_service.get_chunks_by_ids(top_ids)
        # Attach score for convenience
        by_id = {row.get("document_key"): row for row in top_rows}
        result_rows = []
        for cid, sc in zip(top_ids, top_scores):
            row = by_id.get(cid, {"document_key": cid})
            row = dict(row)
            row["dpr_score"] = sc
            result_rows.append(row)

        return {"chunk_ids": top_ids, "scores": top_scores, "chunks": result_rows}
