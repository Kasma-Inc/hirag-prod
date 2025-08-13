import hashlib
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

import redis
from sqlmodel.ext.asyncio.session import AsyncSession

from .storage.pg_utils import DatabaseClient

logger = logging.getLogger(__name__)


REDIS_EXPIRE_TTL = os.getenv("REDIS_EXPIRE_TTL", 3600 * 24)  # 1 day by default

try:
    EXPIRE_TTL = int(REDIS_EXPIRE_TTL)
except ValueError:
    logger.warning(
        f"Invalid REDIS_EXPIRE_TTL value: {REDIS_EXPIRE_TTL}, using default 1 day"
    )
    EXPIRE_TTL = 3600 * 24


class ExtractionType(Enum):
    """Extraction type enumeration"""

    ENTITY = "entity"
    RELATION = "relation"


class JobStatus(Enum):
    """Job status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumeTracker:
    """
    Redis-based chunk-level processing status tracker for efficient resume functionality.

    Tracks the processing status of each chunk for:
    - Entity extraction completion
    - Relation extraction completion
    - Processing timestamps and metadata

    Maintains document completion status for resume functionality across sessions.
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        auto_cleanup: bool = True,
    ):
        """Initialize the resume tracker with Redis backend"""
        # TODO: Implement connection pooling for better Redis performance under load
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix
        self.auto_cleanup = auto_cleanup

        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _chunk_key(self, chunk_id: str) -> str:
        """Generate Redis key for chunk status"""
        return f"{self.key_prefix}:chunk:{chunk_id}"

    def _doc_chunks_key(self, document_id: str) -> str:
        """Generate Redis key for document's chunk set"""
        return f"{self.key_prefix}:doc:{document_id}:chunks"

    def _doc_entity_completed_key(self, document_id: str) -> str:
        """Generate Redis key for document's entity-completed chunks"""
        return f"{self.key_prefix}:doc:{document_id}:entity_completed"

    def _doc_relation_completed_key(self, document_id: str) -> str:
        """Generate Redis key for document's relation-completed chunks"""
        return f"{self.key_prefix}:doc:{document_id}:relation_completed"

    def _doc_info_key(self, document_id: str) -> str:
        """Generate Redis key for document info"""
        return f"{self.key_prefix}:doc:{document_id}:info"

    def _doc_completion_key(self, document_id: str) -> str:
        """Generate Redis key for document completion status (persistent)"""
        return f"{self.key_prefix}:completed:{document_id}"

    def _job_key(self, job_id: str) -> str:
        """Generate Redis key for ingestion job status"""
        return f"{self.key_prefix}:job:{job_id}"

    def _calculate_chunk_hash(self, chunk_content: str) -> str:
        """Calculate a hash for chunk content to detect changes"""
        return hashlib.md5(chunk_content.encode()).hexdigest()

    def is_document_already_completed(self, document_id: str) -> bool:
        """Check if document was already fully processed in a previous session"""
        completion_key = self._doc_completion_key(document_id)
        completion_data = self.redis_client.hgetall(completion_key)

        if not completion_data:
            return False

        is_completed = completion_data.get("pipeline_completed", "false") == "true"
        if is_completed:
            logger.info(
                f"Document {document_id} was already completed on {completion_data.get('completed_at')}"
            )

        return is_completed

    # ==========================================================================
    # Job-level tracking (for insert_to_kb)
    # ==========================================================================

    def _ensure_job_exists(
        self,
        job_id: str,
        document_uri: Optional[str] = None,
        with_graph: Optional[bool] = None,
    ) -> None:
        """Idempotently create a job hash if absent, defaulting to PENDING.

        This allows callers to pass an external job_id without a prior create step.
        """
        key = self._job_key(job_id)
        if self.redis_client.exists(key):
            return
        now = datetime.now().isoformat()
        mapping = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "document_uri": (document_uri or ""),
            "document_id": "",
            "with_graph": "true" if (with_graph is None or with_graph) else "false",
            "total_chunks": "0",
            "processed_chunks": "0",
            "total_entities": "0",
            "total_relations": "0",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        self.redis_client.hset(key, mapping=mapping)
        self.redis_client.expire(key, EXPIRE_TTL)

    async def _persist_job_status(
        self, job_id: str, status: str, extra: Optional[Dict[str, str]] = None
    ) -> None:
        """Hook to persist job status into PostgreSQL. No-op by default.

        Override or implement `save_job_status_to_postgres` to enable persistence.
        """
        try:
            await self.save_job_status_to_postgres(job_id, status, extra or {})
        except Exception:
            # Swallow persistence errors to avoid impacting pipeline
            pass

    async def save_job_status_to_postgres(
        self, job_id: str, status: str, extra: Dict[str, str]
    ) -> None:
        """Persist job status to PostgreSQL using DatabaseClient.

        - For terminal/explicit state changes (pending, processing, completed, failed),
          update both status and updatedAt.
        - For lightweight progress updates, only touch updatedAt.
        """
        try:
            db_client = DatabaseClient()
            engine = db_client.create_db_engine(db_client.connection_string)
            session = AsyncSession(engine)

            normalized_status = status or ""
            # Normalize transient progress ticks to processing for PG persistence
            if normalized_status.lower() == "progress":
                normalized_status = JobStatus.PROCESSING.value
            await db_client.update_job_status(
                session, job_id, normalized_status, updated_at=datetime.now()
            )
        except Exception:
            # Never let persistence issues break the pipeline
            pass

    async def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        document_uri: Optional[str] = None,
        with_graph: Optional[bool] = None,
    ) -> None:
        key = self._job_key(job_id)
        self._ensure_job_exists(
            job_id, document_uri=document_uri, with_graph=with_graph
        )
        now = datetime.now().isoformat()
        mapping = {"status": status.value, "updated_at": now}
        self.redis_client.hset(key, mapping=mapping)
        await self._persist_job_status(job_id, status.value, mapping)

    async def set_job_processing(
        self,
        job_id: str,
        document_id: Optional[str] = None,
        total_chunks: Optional[int] = None,
    ) -> None:
        """Mark job as processing and optionally set document_id/total_chunks."""
        key = self._job_key(job_id)
        self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        mapping = {"status": JobStatus.PROCESSING.value, "updated_at": now}
        if document_id is not None:
            mapping["document_id"] = document_id
        if total_chunks is not None:
            mapping["total_chunks"] = str(int(total_chunks))
        self.redis_client.hset(key, mapping=mapping)
        await self._persist_job_status(job_id, JobStatus.PROCESSING.value, mapping)

    async def set_job_progress(
        self,
        job_id: str,
        processed_chunks: Optional[int] = None,
        total_entities: Optional[int] = None,
        total_relations: Optional[int] = None,
    ) -> None:
        key = self._job_key(job_id)
        self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        mapping: Dict[str, str] = {"updated_at": now}
        if processed_chunks is not None:
            mapping["processed_chunks"] = str(int(processed_chunks))
        if total_entities is not None:
            mapping["total_entities"] = str(int(total_entities))
        if total_relations is not None:
            mapping["total_relations"] = str(int(total_relations))
        self.redis_client.hset(key, mapping=mapping)
        # Progress updates are frequent; persist if desired
        await self._persist_job_status(job_id, "progress", mapping)

    async def set_job_completed(self, job_id: str) -> None:
        await self.set_job_status(job_id, JobStatus.COMPLETED)

    async def set_job_failed(self, job_id: str, error_message: str) -> None:
        key = self._job_key(job_id)
        self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        self.redis_client.hset(
            key,
            mapping={
                "status": JobStatus.FAILED.value,
                "error": (error_message or "")[:5000],
                "updated_at": now,
            },
        )
        await self._persist_job_status(
            job_id, JobStatus.FAILED.value, {"error": (error_message or "")[:5000]}
        )

    def register_chunks(
        self, chunks: List, document_id: str, document_uri: str
    ) -> None:
        """Register chunks in the tracking system"""
        if not chunks:
            return

        # Check if document was already completed
        if self.is_document_already_completed(document_id):
            logger.info(
                f"Document {document_id} already completed, skipping registration"
            )
            return

        pipeline = self.redis_client.pipeline()
        now = datetime.now().isoformat()

        # Check if document already exists in current session
        doc_info_key = self._doc_info_key(document_id)
        if self.redis_client.exists(doc_info_key):
            logger.debug(
                f"Document {document_id} already registered in current session, skipping chunk registration"
            )
            return

        # Register document info
        doc_info = {
            "document_id": document_id,
            "document_uri": document_uri,
            "total_chunks": len(chunks),
            "created_at": now,
            "last_updated": now,
        }
        pipeline.hset(doc_info_key, mapping=doc_info)

        # Register chunks
        doc_chunks_key = self._doc_chunks_key(document_id)
        for chunk in chunks:
            chunk_key = self._chunk_key(chunk.id)
            chunk_data = {
                "chunk_id": chunk.id,
                "document_id": document_id,
                "document_uri": document_uri,
                "chunk_hash": self._calculate_chunk_hash(chunk.page_content),
                "entity_extraction_completed": "false",
                "relation_extraction_completed": "false",
                "entity_count": "0",
                "relation_count": "0",
                "created_at": now,
            }
            pipeline.hset(chunk_key, mapping=chunk_data)
            pipeline.sadd(doc_chunks_key, chunk.id)
            # Set TTL for chunk data (30 days default)
            pipeline.expire(chunk_key, EXPIRE_TTL)

        # Set TTL for document keys (but not completion key)
        pipeline.expire(doc_info_key, EXPIRE_TTL)
        pipeline.expire(doc_chunks_key, EXPIRE_TTL)
        pipeline.execute()

        logger.info(
            f"Registered {len(chunks)} chunks for document {document_id} in Redis"
        )

    def _get_chunk_ids_with_status(
        self, chunks: List, extraction_type: ExtractionType
    ) -> Set[str]:
        """Get chunk IDs that have completed the specified extraction type"""
        if not chunks:
            return set()

        document_id = chunks[0].metadata.document_id
        if extraction_type == ExtractionType.ENTITY:
            completed_key = self._doc_entity_completed_key(document_id)
        else:
            completed_key = self._doc_relation_completed_key(document_id)

        completed_chunks = self.redis_client.smembers(completed_key)
        return set(completed_chunks)

    def get_pending_chunks(self, chunks: List, extraction_type: ExtractionType) -> List:
        """Get chunks that need the specified extraction type"""
        if not chunks:
            return []

        completed_chunk_ids = self._get_chunk_ids_with_status(chunks, extraction_type)

        # For relations, also check that entity extraction is completed
        if extraction_type == ExtractionType.RELATION:
            entity_completed_ids = self._get_chunk_ids_with_status(
                chunks, ExtractionType.ENTITY
            )
            pending_chunks = [
                chunk
                for chunk in chunks
                if chunk.id in entity_completed_ids
                and chunk.id not in completed_chunk_ids
            ]
        else:
            pending_chunks = [
                chunk for chunk in chunks if chunk.id not in completed_chunk_ids
            ]

        logger.info(
            f"Found {len(pending_chunks)} chunks pending {extraction_type.value} extraction out of {len(chunks)} total"
        )
        return pending_chunks

    def get_pending_entity_chunks(self, chunks: List) -> List:
        """Get chunks that need entity extraction"""
        return self.get_pending_chunks(chunks, ExtractionType.ENTITY)

    def get_pending_relation_chunks(self, chunks: List) -> List:
        """Get chunks that need relation extraction"""
        return self.get_pending_chunks(chunks, ExtractionType.RELATION)

    def mark_extraction_started(
        self, chunks: List, extraction_type: ExtractionType
    ) -> None:
        """Mark chunks as having started the specified extraction"""
        if not chunks:
            return

        pipeline = self.redis_client.pipeline()
        now = datetime.now().isoformat()
        field_name = f"{extraction_type.value}_extraction_started_at"

        for chunk in chunks:
            chunk_key = self._chunk_key(chunk.id)
            pipeline.hset(chunk_key, field_name, now)

        pipeline.execute()
        logger.info(
            f"Marked {len(chunks)} chunks as {extraction_type.value} extraction started"
        )

    def mark_extraction_completed(
        self,
        chunks: List,
        extraction_type: ExtractionType,
        counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Mark chunks as having completed the specified extraction"""
        if not chunks:
            return

        counts = counts or {}
        pipeline = self.redis_client.pipeline()
        now = datetime.now().isoformat()
        document_id = chunks[0].metadata.document_id

        # Get the appropriate completed set key
        if extraction_type == ExtractionType.ENTITY:
            completed_key = self._doc_entity_completed_key(document_id)
        else:
            completed_key = self._doc_relation_completed_key(document_id)

        # Update chunk status and add to completed set
        for chunk in chunks:
            chunk_key = self._chunk_key(chunk.id)
            updates = {
                f"{extraction_type.value}_extraction_completed": "true",
                f"{extraction_type.value}_extraction_completed_at": now,
                f"{extraction_type.value}_count": str(counts.get(chunk.id, 0)),
                "updated_at": now,
            }
            pipeline.hset(chunk_key, mapping=updates)
            pipeline.sadd(completed_key, chunk.id)

        # Set TTL for completed set
        pipeline.expire(completed_key, EXPIRE_TTL)
        pipeline.execute()

        logger.info(
            f"Marked {len(chunks)} chunks as {extraction_type.value} extraction completed"
        )

        # Check if document is complete and cleanup if so (only if auto_cleanup is enabled)
        if self.auto_cleanup:
            self._check_and_cleanup_if_complete(document_id)

    def mark_entity_extraction_started(self, chunks: List) -> None:
        """Mark chunks as having started entity extraction"""
        self.mark_extraction_started(chunks, ExtractionType.ENTITY)

    def mark_entity_extraction_completed(
        self, chunks: List, entity_counts: Dict[str, int] = None
    ) -> None:
        """Mark chunks as having completed entity extraction"""
        self.mark_extraction_completed(chunks, ExtractionType.ENTITY, entity_counts)

    def mark_relation_extraction_started(self, chunks: List) -> None:
        """Mark chunks as having started relation extraction"""
        self.mark_extraction_started(chunks, ExtractionType.RELATION)

    def mark_relation_extraction_completed(
        self, chunks: List, relation_counts: Dict[str, int] = None
    ) -> None:
        """Mark chunks as having completed relation extraction"""
        self.mark_extraction_completed(chunks, ExtractionType.RELATION, relation_counts)

    def is_document_complete(self, document_id: str) -> bool:
        """Check if entire document processing is complete in current session"""
        doc_info_key = self._doc_info_key(document_id)
        doc_chunks_key = self._doc_chunks_key(document_id)
        entity_completed_key = self._doc_entity_completed_key(document_id)
        relation_completed_key = self._doc_relation_completed_key(document_id)

        # Get document info
        doc_info = self.redis_client.hgetall(doc_info_key)
        if not doc_info:
            return False

        total_chunks = int(doc_info.get("total_chunks", 0))
        if total_chunks == 0:
            return False

        # Check completion status
        entity_completed_count = self.redis_client.scard(entity_completed_key)
        relation_completed_count = self.redis_client.scard(relation_completed_key)

        is_complete = (
            entity_completed_count == total_chunks
            and relation_completed_count == total_chunks
        )

        return is_complete

    def mark_document_completed(self, document_id: str) -> None:
        """Mark entire document as completed and trigger cleanup"""
        now = datetime.now().isoformat()

        # Store persistent completion record
        completion_key = self._doc_completion_key(document_id)
        completion_data = {
            "document_id": document_id,
            "pipeline_completed": "true",
            "completed_at": now,
            "last_updated": now,
        }
        # Set long TTL for completion records (30 days)
        self.redis_client.hset(completion_key, mapping=completion_data)
        self.redis_client.expire(completion_key, EXPIRE_TTL)

        # Update document status in current session
        doc_info_key = self._doc_info_key(document_id)
        if self.redis_client.exists(doc_info_key):
            self.redis_client.hset(
                doc_info_key,
                mapping={
                    "pipeline_completed": "true",
                    "pipeline_completed_at": now,
                    "last_updated": now,
                },
            )

        logger.info(f"Marked document {document_id} as fully completed")

        # Cleanup tracking data since processing is complete
        self._cleanup_document_tracking(document_id)

    def _check_and_cleanup_if_complete(self, document_id: str) -> None:
        """Check if document is complete and cleanup if so"""
        if self.is_document_complete(document_id):
            logger.info(
                f"Document {document_id} processing complete, cleaning up tracking data"
            )
            self._cleanup_document_tracking(document_id)

    def _cleanup_document_tracking(self, document_id: str) -> None:
        """Clean up session tracking data for a completed document (keeps completion record)"""
        try:
            # Get all chunk IDs for this document
            doc_chunks_key = self._doc_chunks_key(document_id)
            chunk_ids = self.redis_client.smembers(doc_chunks_key)

            # Prepare pipeline for cleanup
            pipeline = self.redis_client.pipeline()

            # Delete chunk status keys
            for chunk_id in chunk_ids:
                chunk_key = self._chunk_key(chunk_id)
                pipeline.delete(chunk_key)

            # Delete document session keys (but keep completion record)
            pipeline.delete(doc_chunks_key)
            pipeline.delete(self._doc_entity_completed_key(document_id))
            pipeline.delete(self._doc_relation_completed_key(document_id))
            pipeline.delete(self._doc_info_key(document_id))

            # Execute cleanup
            pipeline.execute()

            logger.info(
                f"Cleaned up tracking data for {len(chunk_ids)} chunks in document {document_id}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to cleanup tracking data for document {document_id}: {e}"
            )

    def get_processing_stats(self, document_id: str) -> Dict:
        """Get detailed processing statistics for a document"""
        doc_info_key = self._doc_info_key(document_id)
        doc_info = self.redis_client.hgetall(doc_info_key)

        if not doc_info:
            # Check if document was completed in a previous session
            completion_key = self._doc_completion_key(document_id)
            completion_data = self.redis_client.hgetall(completion_key)
            if completion_data:
                return {
                    "document_id": document_id,
                    "pipeline_completed": True,
                    "completed_at": completion_data.get("completed_at"),
                    "status": "Previously completed",
                }
            return {"error": "Document not found"}

        total_chunks = int(doc_info.get("total_chunks", 0))
        entity_completed = self.redis_client.scard(
            self._doc_entity_completed_key(document_id)
        )
        relation_completed = self.redis_client.scard(
            self._doc_relation_completed_key(document_id)
        )

        # Calculate totals by checking individual chunks
        doc_chunks_key = self._doc_chunks_key(document_id)
        chunk_ids = self.redis_client.smembers(doc_chunks_key)

        total_entities = 0
        total_relations = 0

        if chunk_ids:
            pipeline = self.redis_client.pipeline()
            for chunk_id in chunk_ids:
                chunk_key = self._chunk_key(chunk_id)
                pipeline.hmget(chunk_key, "entity_count", "relation_count")

            results = pipeline.execute()
            for entity_count, relation_count in results:
                total_entities += int(entity_count or 0)
                total_relations += int(relation_count or 0)

        return {
            "document_id": document_id,
            "total_chunks": total_chunks,
            "entity_extraction": {
                "completed_chunks": entity_completed,
                "progress": f"{entity_completed}/{total_chunks}",
                "percentage": (
                    (entity_completed / total_chunks * 100) if total_chunks > 0 else 0
                ),
            },
            "relation_extraction": {
                "completed_chunks": relation_completed,
                "progress": f"{relation_completed}/{total_chunks}",
                "percentage": (
                    (relation_completed / total_chunks * 100) if total_chunks > 0 else 0
                ),
            },
            "totals": {
                "entities": total_entities,
                "relations": total_relations,
            },
            "pipeline_completed": doc_info.get("pipeline_completed", "false") == "true",
            "last_updated": doc_info.get("last_updated"),
        }

    def reset_document(self, document_id: str) -> None:
        """Reset processing status for a document (for testing/debugging)"""
        # Clean up both session data and completion record
        self._cleanup_document_tracking(document_id)
        completion_key = self._doc_completion_key(document_id)
        self.redis_client.delete(completion_key)
        logger.info(f"Reset processing status for document {document_id}")

    def cleanup_old_entries(self, days: int = 30) -> None:
        """Clean up tracking entries older than specified days (Redis TTL handles this automatically)"""
        # Redis TTL automatically handles cleanup, but we can manually clean expired keys
        # This is mainly for compatibility with the interface
        logger.info(
            f"Redis TTL automatically handles cleanup of entries older than {days} days"
        )
