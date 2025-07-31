import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
from dotenv import load_dotenv

from hirag_prod._llm import (
    ChatCompletion,
    EmbeddingService,
    LocalChatService,
    LocalEmbeddingService,
    create_chat_service,
    create_embedding_service,
)
from hirag_prod._utils import _limited_gather_with_factory
from hirag_prod.chunk import BaseChunk, FixTokenChunk
from hirag_prod.entity import BaseEntity, VanillaEntity
from hirag_prod.loader import load_document
from hirag_prod.loader.chunk_split import (
    chunk_docling_document,
    chunk_langchain_document,
)
from hirag_prod.parser import (
    DictParser,
    ReferenceParser,
)
from hirag_prod.prompt import PROMPTS
from hirag_prod.resume_tracker import ResumeTracker
from hirag_prod.schema import Entity
from hirag_prod.schema.entity import EntityMetadata
from hirag_prod.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

# from hirag_prod.similarity import CosineSimilarity


load_dotenv("/chatbot/.env", override=True)

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
DEFAULT_LLM_MODEL_NAME = "gpt-4o-mini"
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
    EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION"))
except ValueError as e:
    raise ValueError(f"EMBEDDING_DIMENSION must be an integer: {e}")

# Query and Operation Constants
MAX_CHUNK_IDS_PER_QUERY = 10
DEFAULT_QUERY_TOPK = 10
DEFAULT_QUERY_TOPN = 5

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


class EntityExtractionError(HiRAGException):
    """Entity extraction exception"""


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "processing_time": self.processing_time,
            "error_count": self.error_count,
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
        self.entities_table = None

    async def initialize(self) -> None:
        """Initialize storage tables"""
        await self._initialize_chunks_table()
        await self._initialize_entities_table()

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
                            pa.field(
                                "vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)
                            ),
                            pa.field("uploaded_at", pa.timestamp("ms")),
                        ]
                    ),
                )
            else:
                raise StorageError(f"Failed to initialize chunks table: {e}")

    async def _initialize_entities_table(self) -> None:
        """Initialize entities table"""
        try:
            self.entities_table = await self.vdb.db.open_table("entities")
        except Exception as e:
            if "was not found" in str(e):
                self.entities_table = await self.vdb.db.create_table(
                    "entities",
                    schema=pa.schema(
                        [
                            pa.field("text", pa.string()),
                            pa.field("document_key", pa.string()),
                            pa.field(
                                "vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)
                            ),
                            pa.field("entity_type", pa.string()),
                            pa.field("description", pa.list_(pa.string())),
                            pa.field("chunk_ids", pa.list_(pa.string())),
                            pa.field(
                                "extraction_timestamp",
                                pa.timestamp("ms"),
                                nullable=True,
                            ),
                            pa.field("source_document_id", pa.string(), nullable=True),
                        ]
                    ),
                )
            else:
                raise StorageError(f"Failed to initialize entities table: {e}")

    @retry_async(max_retries=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY)
    async def upsert_chunks(self, chunks: List[BaseChunk]) -> None:
        """Batch insert chunks"""
        # TODO: Implement intelligent batching based on chunk size and content complexity
        if not chunks:
            return

        try:
            if len(chunks) == 1:
                chunk = chunks[0]
                await self.vdb.upsert_text(
                    text_to_embed=chunk.page_content,
                    properties={
                        "document_key": chunk.id,
                        "text": chunk.page_content,
                        **chunk.metadata.__dict__,
                    },
                    table=self.chunks_table,
                    mode="append",
                )
            else:
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
    async def upsert_entities(self, entities: List[Entity]) -> None:
        """Batch insert entities"""
        if not entities:
            return

        try:
            if len(entities) == 1:
                entity = entities[0]
                description_text = (
                    " | ".join(entity.metadata.description)
                    if entity.metadata.description
                    else ""
                )
                await self.vdb.upsert_text(
                    text_to_embed=description_text,
                    properties={
                        "document_key": entity.id,
                        "text": entity.page_content,
                        **entity.metadata.__dict__,
                    },
                    table=self.entities_table,
                    mode="append",
                )
            else:
                texts_to_embed = [
                    (
                        " | ".join(entity.metadata.description)
                        if entity.metadata.description
                        else ""
                    )
                    for entity in entities
                ]
                properties_list = [
                    {
                        "document_key": entity.id,
                        "text": entity.page_content,
                        **entity.metadata.__dict__,
                    }
                    for entity in entities
                ]

                await self.vdb.upsert_texts(
                    texts_to_embed=texts_to_embed,
                    properties_list=properties_list,
                    table=self.entities_table,
                    mode="append",
                )
        except Exception as e:
            raise StorageError(f"Failed to upsert entities: {e}")

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

    async def get_existing_entities_for_chunks(
        self, chunk_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get existing entities for specified chunks"""
        if not chunk_ids:
            return []

        try:
            # Limit query conditions to avoid overly long queries
            limited_chunk_ids = chunk_ids[:MAX_CHUNK_IDS_PER_QUERY]
            chunk_conditions = " OR ".join(
                [
                    f"array_contains(chunk_ids, '{chunk_id}')"
                    for chunk_id in limited_chunk_ids
                ]
            )

            if chunk_conditions:
                existing_entities = (
                    await self.entities_table.query().where(chunk_conditions).to_list()
                )
                return existing_entities
            return []
        except Exception as e:
            logger.warning(f"Failed to get existing entities: {e}")
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
        entity_extractor: BaseEntity,
        resume_tracker: Optional[object] = None,
        config: Optional[HiRAGConfig] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.storage = storage
        self.chunker = chunker
        self.entity_extractor = entity_extractor
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
    ) -> ProcessingMetrics:
        """Process a single document"""
        # TODO: Add document preprocessing pipeline for better quality - OCR, cleanup, etc.
        validate_input(document_path, content_type)

        async with self.metrics.track_operation(f"process_document"):
            # Load and chunk document
            chunks = await self._load_and_chunk_document(
                document_path, content_type, document_meta, loader_configs
            )

            if not chunks:
                logger.warning("⚠️ No chunks created from document")
                return self.metrics.metrics

            self.metrics.metrics.total_chunks = len(chunks)

            # Check if document was already completed in a previous session
            if self.resume_tracker:
                document_id = chunks[0].metadata.document_id
                if self.resume_tracker.is_document_already_completed(document_id):
                    logger.info(
                        "🎉 Document already fully processed in previous session!"
                    )
                    return self.metrics.metrics
                else:
                    document_uri = chunks[0].metadata.uri
                    self.resume_tracker.register_chunks(
                        chunks, document_id, document_uri
                    )

            # Process chunks
            await self._process_chunks(chunks)

            # Process graph data
            if with_graph:
                entities = await self._process_entities(chunks)
                await self._process_relations(chunks, entities)

            # Mark as complete
            if self.resume_tracker:
                self.resume_tracker.mark_document_completed(
                    chunks[0].metadata.document_id
                )

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

    async def _process_entities(self, chunks: List[BaseChunk]) -> List[Entity]:
        """Process entity extraction and storage"""
        async with self.metrics.track_operation("process_entities"):
            # Get chunks that need entity processing
            pending_chunks = self._get_pending_entity_chunks(chunks)

            if not pending_chunks:
                logger.info("⏭️ All chunks already have entities extracted")
                return []

            logger.info(f"🔍 Extracting entities from {len(pending_chunks)} chunks...")

            # Mark start
            if self.resume_tracker:
                self.resume_tracker.mark_entity_extraction_started(pending_chunks)

            try:
                # Extract entities
                entities = await self.entity_extractor.entity(pending_chunks)

                if entities:
                    # Store to vector database
                    await self.storage.upsert_entities(entities)

                    # Store to graph database
                    await self.storage.gdb.upsert_nodes(
                        entities, concurrency=self.config.entity_upsert_concurrency
                    )

                    self.metrics.metrics.total_entities += len(entities)

                    # Mark complete
                    if self.resume_tracker:
                        entity_counts = self._count_entities_per_chunk(entities)
                        await self._mark_entity_extraction_complete(
                            pending_chunks, entity_counts
                        )

                logger.info(f"✅ Extracted and stored {len(entities)} entities")
                return entities

            except Exception as e:
                raise EntityExtractionError(f"Failed to extract entities: {e}")

    def _get_pending_entity_chunks(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """Get chunks that need entity extraction"""
        if self.resume_tracker:
            return self.resume_tracker.get_pending_entity_chunks(chunks)
        return chunks

    def _count_entities_per_chunk(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities per chunk"""
        counts = {}
        for entity in entities:
            for chunk_id in entity.metadata.chunk_ids:
                counts[chunk_id] = counts.get(chunk_id, 0) + 1
        return counts

    async def _mark_entity_extraction_complete(
        self, chunks: List[BaseChunk], entity_counts: Dict[str, int]
    ) -> None:
        """Mark entity extraction complete"""
        try:
            if self.resume_tracker:
                self.resume_tracker.mark_entity_extraction_completed(
                    chunks, entity_counts
                )
        except Exception as e:
            logger.warning(f"Failed to mark entity extraction complete: {e}")

    async def _process_relations(
        self, chunks: List[BaseChunk], new_entities: List[Entity]
    ) -> None:
        """Process relation extraction and storage"""
        async with self.metrics.track_operation("process_relations"):
            # Get chunks that need relation processing
            pending_chunks = self._get_pending_relation_chunks(chunks)

            if not pending_chunks:
                logger.info("⏭️ No chunks need relation extraction")
                return

            logger.info(f"🔗 Extracting relations from {len(pending_chunks)} chunks...")

            # Mark start
            if self.resume_tracker:
                self.resume_tracker.mark_relation_extraction_started(pending_chunks)

            try:
                # Get all related entities
                all_entities = await self._get_all_entities_for_chunks(
                    pending_chunks, new_entities
                )

                # Extract relations
                relations = await self.entity_extractor.relation(
                    pending_chunks, all_entities
                )

                if relations:
                    # Store relations
                    relation_factories = [
                        lambda rel=rel: self.storage.gdb.upsert_relation(rel)
                        for rel in relations
                    ]
                    await _limited_gather_with_factory(
                        relation_factories, self.config.relation_upsert_concurrency
                    )

                    self.metrics.metrics.total_relations += len(relations)

                    logger.info(f"✅ Extracted and stored {len(relations)} relations")
                else:
                    logger.info("ℹ️ No relations extracted")

                # Mark as complete
                relation_counts = self._count_relations_per_chunk(relations)
                await self._mark_relation_extraction_complete(
                    pending_chunks, relation_counts
                )

            except Exception as e:
                logger.error(f"Failed to extract relations: {e}")
                # Still mark as complete to avoid duplicate processing
                await self._mark_relation_extraction_complete(pending_chunks, {})

    def _get_pending_relation_chunks(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """Get chunks that need relation extraction"""
        if self.resume_tracker:
            return self.resume_tracker.get_pending_relation_chunks(chunks)
        return chunks

    async def _get_all_entities_for_chunks(
        self, chunks: List[BaseChunk], new_entities: List[Entity]
    ) -> List[Entity]:
        """Get all entities for chunks"""
        all_entities = list(new_entities)

        # Get existing entities
        chunk_ids = [chunk.id for chunk in chunks]
        existing_entities_data = await self.storage.get_existing_entities_for_chunks(
            chunk_ids
        )

        # Convert to entity objects
        for ent_data in existing_entities_data:
            if ent_data["document_key"] not in [e.id for e in new_entities]:
                entity_obj = Entity(
                    id=ent_data["document_key"],
                    page_content=ent_data["text"],
                    metadata=EntityMetadata(
                        entity_type=ent_data.get("entity_type", ""),
                        description=ent_data.get("description", []),
                        chunk_ids=ent_data.get("chunk_ids", []),
                    ),
                )
                all_entities.append(entity_obj)

        return all_entities

    def _count_relations_per_chunk(self, relations: List) -> Dict[str, int]:
        """Count relations per chunk"""
        counts = {}
        for rel in relations:
            chunk_id = rel.properties.get("chunk_id")
            if chunk_id:
                counts[chunk_id] = counts.get(chunk_id, 0) + 1
        return counts

    async def _mark_relation_extraction_complete(
        self, chunks: List[BaseChunk], relation_counts: Dict[str, int]
    ) -> None:
        """Mark relation extraction complete"""
        try:
            if self.resume_tracker:
                self.resume_tracker.mark_relation_extraction_completed(
                    chunks, relation_counts
                )
        except Exception as e:
            logger.warning(f"Failed to mark relation extraction complete: {e}")


# ============================================================================
# Query Service
# ============================================================================


class QueryService:
    """Query service"""

    def __init__(self, storage: StorageManager):
        self.storage = storage

    async def query_chunks(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
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
        )

    async def query_entities(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> List[Dict[str, Any]]:
        """Query entities"""
        return await self.storage.vdb.query(
            query=query,
            table=self.storage.entities_table,
            topk=topk,
            topn=topn,
            columns_to_select=["text", "document_key", "entity_type", "description"],
        )

    async def query_relations(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> Tuple[List[str], List[str]]:
        """Query relations"""
        # Search related entities
        entities = await self.query_entities(query, topk, topn)
        entity_ids = [entity["document_key"] for entity in entities]

        # Search relations
        neighbors = []
        edges = []
        for entity_id in entity_ids:
            entity_neighbors, entity_edges = await self.storage.gdb.query_one_hop(
                entity_id
            )
            neighbors.extend(entity_neighbors)
            edges.extend(entity_edges)

        return neighbors, edges

    async def query_all(self, query: str) -> Dict[str, Any]:
        """Query all"""
        chunks = await self.query_chunks(
            query, topk=DEFAULT_QUERY_TOPK, topn=DEFAULT_QUERY_TOPN
        )
        entities = await self.query_entities(
            query, topk=DEFAULT_QUERY_TOPK, topn=DEFAULT_QUERY_TOPN
        )
        neighbors, relations = await self.query_relations(
            query, topk=DEFAULT_QUERY_TOPK, topn=DEFAULT_QUERY_TOPN
        )

        return {
            "chunks": chunks,
            "entities": entities,
            "neighbors": neighbors,
            "relations": relations,
        }

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

    async def query_entity_embeddings(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Query entity embeddings"""
        if not entity_ids:
            return {}

        res = {}
        try:
            # Query entity embeddings by keys
            entity_data = await self.storage.vdb.query_by_keys(
                key_value=entity_ids,
                key_column="document_key",
                table=self.storage.entities_table,
                columns_to_select=["document_key", "vector"],
            )

            # entity data is a list of dicts with 'vector' key
            for entity in entity_data:
                if "vector" in entity and entity["vector"] is not None:
                    res[entity["document_key"]] = entity["vector"]
                else:
                    # Log missing vector data and raise exception
                    logger.warning(
                        f"Entity {entity['document_key']} has no vector data"
                    )
                    res[entity["document_key"]] = None

        except Exception as e:
            logger.error(f"Failed to query entity embeddings: {e}")
            return {}

        return res


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
    _entity_extractor: Optional[VanillaEntity] = field(default=None, init=False)
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
        self._entity_extractor.set_language(language)

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

        self._entity_extractor = VanillaEntity.create(
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
            entity_extractor=self._entity_extractor,
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
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> str:
        """Generate summary from chunks, entities, and relations"""
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
            data += (
                "Entities:\n" + parser.parse_list_of_dicts(entities, "table") + "\n\n"
            )
            # data += "Relations:\n" + str(relationships) + "\n\n"

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

            # for each sentence, do a query and find the best matching document key to find the referenced chunk, entity, or relationship
            result = []

            chunk_keys = [c["document_key"] for c in chunks]
            entity_keys = [e["document_key"] for e in entities]

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
            entity_embeddings = await self._query_service.query_entity_embeddings(
                entity_keys
            )

            # relation_descriptions = [rel.properties["description"] for rel in relationships]
            # relation_embeddings = await self.embedding_service.create_embeddings(texts=relation_descriptions)
            # # make relation embeddings be a dict with key as "source:target" and value as the embedding
            # relation_embeddings = {
            #     f"rel({rel.source}:{rel.target})": embedding
            #     for rel, embedding in zip(relationships, relation_embeddings)
            # }

            for sentence, sentence_embedding in zip(ref_sentences, sentence_embeddings):
                found = False

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

                similar_entities = await self.calculate_similarity(
                    sentence_embedding, entity_embeddings
                )

                if DEBUG:
                    print(
                        "\n\n\nSimilar Entities for Sentence:",
                        sentence,
                        "\n",
                        similar_entities,
                    )

                # similar_relations = await self.calculate_similarity(sentence_embedding, relation_embeddings)

                # if DEBUG:
                #     print("\n\n\nSimilar Relations for Sentence:", sentence, "\n", similar_relations)

                # Sort by similarity
                # reference_list = similar_chunks + similar_entities + similar_relations
                reference_list = similar_chunks + similar_entities
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

        try:
            metrics = await self._processor.process_document(
                document_path=document_path,
                content_type=content_type,
                with_graph=with_graph,
                document_meta=document_meta,
                loader_configs=loader_configs,
            )

            # Save graph state
            if with_graph and self._storage:
                await self._storage.gdb.dump()

            total_time = time.perf_counter() - start_time
            metrics.processing_time = total_time
            logger.info(f"🏁 Total pipeline time: {total_time:.3f}s")

            return metrics

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"❌ Document processing failed after {total_time:.3f}s: {e}")
            raise

    async def query_chunks(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> List[Dict[str, Any]]:
        """Query document chunks"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        return await self._query_service.query_chunks(query, topk, topn)

    async def query_entities(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> List[Dict[str, Any]]:
        """Query entities"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        return await self._query_service.query_entities(query, topk, topn)

    async def query_relations(
        self, query: str, topk: int = DEFAULT_QUERY_TOPK, topn: int = DEFAULT_QUERY_TOPN
    ) -> Tuple[List[str], List[str]]:
        """Query relations"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        return await self._query_service.query_relations(query, topk, topn)

    async def query_all(self, query: str, summary: bool = False) -> Dict[str, Any]:
        """Query all types of data"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        if summary:
            query_all_results = await self._query_service.query_all(query)
            text_summary = await self.generate_summary(
                chunks=query_all_results["chunks"],
                entities=query_all_results["entities"],
                relationships=query_all_results["relations"],
            )
            query_all_results["summary"] = text_summary
            return query_all_results
        return await self._query_service.query_all(query)

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
    def entities_table(self):
        """Backward compatibility: access entities table"""
        return self._storage.entities_table if self._storage else None

    @property
    def vdb(self):
        """Backward compatibility: access vector database"""
        return self._storage.vdb if self._storage else None

    @property
    def gdb(self):
        """Backward compatibility: access graph database"""
        return self._storage.gdb if self._storage else None
