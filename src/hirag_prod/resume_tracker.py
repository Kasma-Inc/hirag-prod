import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_envs
from hirag_prod.resources.functions import get_db_session_maker, get_resource_manager
from hirag_prod.storage.pg_utils import update_job_status

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumeTracker:
    """
    Redis-based tracker for:
    - Job-level progress/status
    - Chunk-level registration cache per document (session state)
    - Persistent document completion record

    Notes:
    - No entity/relation stage tracking; only chunk-session info and document completion.
    """

    def __init__(
        self,
        auto_cleanup: bool = True,
    ):
        """Initialize the resume tracker with Redis backend"""
        self.redis_client = get_resource_manager().get_redis_client(
            decode_responses=True
        )
        self.auto_cleanup = auto_cleanup

    # ============================== Helper Functions ==============================

    def _decode_dict(self, data: dict) -> dict:
        """Decode byte keys and values in a dictionary if they are bytes."""
        if not data:
            return {}

        result = {}
        for k, v in data.items():
            # Handle keys
            if isinstance(k, bytes):
                key = k.decode()
            else:
                key = str(k)

            # Handle values
            if isinstance(v, bytes):
                value = v.decode()
            else:
                value = str(v) if v is not None else ""

            result[key] = value

        return result

    def _decode_set(self, data: set) -> set:
        """Decode byte values in a set if they are bytes."""
        if not data:
            return set()

        result = set()
        for item in data:
            if isinstance(item, bytes):
                result.add(item.decode())
            else:
                result.add(str(item) if item is not None else "")

        return result

    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert a value to int, returning default if conversion fails."""
        if value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                return (
                    int(value)
                    if value.isdigit()
                    or (value.startswith("-") and value[1:].isdigit())
                    else default
                )
            return default
        except (ValueError, TypeError) as e:
            log_error_info(
                logging.INFO, f"Failed to convert {value} to int", e, debug_only=True
            )
            return default

    def _scope_prefix(self, workspace_id: str, knowledge_base_id: str) -> str:
        return f"{get_envs().REDIS_KEY_PREFIX}:ws:{workspace_id}:kb:{knowledge_base_id}"

    def _chunk_key(
        self, chunk_id: str, workspace_id: str, knowledge_base_id: str
    ) -> str:
        return f"{self._scope_prefix(workspace_id, knowledge_base_id)}:chunk:{chunk_id}"

    def _doc_chunks_key(
        self, document_id: str, workspace_id: str, knowledge_base_id: str
    ) -> str:
        return f"{self._scope_prefix(workspace_id, knowledge_base_id)}:doc:{document_id}:chunks"

    def _doc_info_key(
        self, document_id: str, workspace_id: str, knowledge_base_id: str
    ) -> str:
        return f"{self._scope_prefix(workspace_id, knowledge_base_id)}:doc:{document_id}:info"

    def _doc_completion_key(
        self, document_id: str, workspace_id: str, knowledge_base_id: str
    ) -> str:
        return f"{self._scope_prefix(workspace_id, knowledge_base_id)}:completed:{document_id}"

    def _job_key(self, job_id: str) -> str:
        return f"{get_envs().REDIS_KEY_PREFIX}:job:{job_id}"

    def _calculate_chunk_hash(
        self, chunk_content: str, workspace_id: str, knowledge_base_id: str
    ) -> str:
        """Calculate a hash for chunk content to detect changes"""
        return hashlib.md5(
            f"{chunk_content}:{workspace_id}:{knowledge_base_id}".encode()
        ).hexdigest()

    async def is_document_already_completed(
        self, document_id: str, workspace_id: str, knowledge_base_id: str
    ) -> bool:
        """Check if document was already fully processed in a previous session"""
        completion_key = self._doc_completion_key(
            document_id, workspace_id, knowledge_base_id
        )
        completion_data = await self.redis_client.hgetall(completion_key)
        if not completion_data:
            return False
        completion_data = self._decode_dict(completion_data)
        is_completed = completion_data.get("pipeline_completed", "false") == "true"
        if is_completed:
            logger.info(
                f"Document {document_id} was already completed on {completion_data.get('completed_at')}"
            )
        return is_completed

    # ============================== Job tracking =============================

    async def _ensure_job_exists(
        self,
        job_id: str,
        document_uri: Optional[str] = None,
        with_graph: Optional[bool] = None,
    ) -> None:
        """Idempotently create a job hash if absent, defaulting to PENDING."""
        key = self._job_key(job_id)
        if await self.redis_client.exists(key):
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
        await self.redis_client.hset(key, mapping=mapping)
        await self.redis_client.expire(key, get_envs().REDIS_EXPIRE_TTL)

    async def save_job_status_to_postgres(self, job_id: str, status: str) -> None:
        """Persist job status to PostgreSQL."""
        try:
            normalized_status = status or ""
            if normalized_status.lower() == "progress":
                normalized_status = JobStatus.PROCESSING.value
            async with get_db_session_maker()() as session:
                await update_job_status(
                    session, job_id, normalized_status, updated_at=datetime.now()
                )
        except Exception as e:
            log_error_info(logging.ERROR, f"Failed to save job status to Postgres", e)

    async def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        document_uri: Optional[str] = None,
        with_graph: Optional[bool] = None,
    ) -> None:
        key = self._job_key(job_id)
        await self._ensure_job_exists(
            job_id, document_uri=document_uri, with_graph=with_graph
        )
        now = datetime.now().isoformat()
        mapping = {"status": status.value, "updated_at": now}
        await self.redis_client.hset(key, mapping=mapping)
        await self.save_job_status_to_postgres(job_id, status.value)

    async def set_job_processing(
        self,
        job_id: str,
        document_id: Optional[str] = None,
        total_chunks: Optional[int] = None,
    ) -> None:
        """Mark job as processing and optionally set document_id/total_chunks."""
        key = self._job_key(job_id)
        await self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        mapping = {"status": JobStatus.PROCESSING.value, "updated_at": now}
        if document_id is not None:
            mapping["document_id"] = document_id
        if total_chunks is not None:
            mapping["total_chunks"] = str(self._safe_int(total_chunks))
        await self.redis_client.hset(key, mapping=mapping)
        await self.save_job_status_to_postgres(job_id, JobStatus.PROCESSING.value)

    async def set_job_progress(
        self,
        job_id: str,
        processed_chunks: Optional[int] = None,
        total_entities: Optional[int] = None,
        total_relations: Optional[int] = None,
    ) -> None:
        """Lightweight progress updates stored on the job hash."""
        key = self._job_key(job_id)
        await self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        mapping: Dict[str, str] = {"updated_at": now}
        if processed_chunks is not None:
            mapping["processed_chunks"] = str(self._safe_int(processed_chunks))
        if total_entities is not None:
            mapping["total_entities"] = str(self._safe_int(total_entities))
        if total_relations is not None:
            mapping["total_relations"] = str(self._safe_int(total_relations))
        await self.redis_client.hset(key, mapping=mapping)
        await self.save_job_status_to_postgres(job_id, "progress")

    async def set_job_completed(self, job_id: str) -> None:
        await self.set_job_status(job_id, JobStatus.COMPLETED)

    async def set_job_failed(self, job_id: str, error_message: str) -> None:
        key = self._job_key(job_id)
        await self._ensure_job_exists(job_id)
        now = datetime.now().isoformat()
        await self.redis_client.hset(
            key,
            mapping={
                "status": JobStatus.FAILED.value,
                "error": (error_message or "")[:5000],
                "updated_at": now,
            },
        )
        await self.save_job_status_to_postgres(job_id, JobStatus.FAILED.value)

    # ============================ Chunk registration ==========================

    async def register_chunks(
        self,
        chunks: List,
        document_id: str,
        document_uri: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Register chunks for a document in the current session."""
        if not chunks:
            return

        if await self.is_document_already_completed(
            document_id, workspace_id, knowledge_base_id
        ):
            logger.info(
                f"Document {document_id} already completed, skipping registration"
            )
            return

        pipeline = self.redis_client.pipeline()
        now = datetime.now().isoformat()

        # If already registered this session, skip
        doc_info_key = self._doc_info_key(document_id, workspace_id, knowledge_base_id)
        if await self.redis_client.exists(doc_info_key):
            logger.debug(
                f"Document {document_id} already registered in current session, skipping chunk registration"
            )
            return

        # Document info
        doc_info = {
            "document_id": document_id,
            "document_uri": document_uri,
            "total_chunks": len(chunks),
            "created_at": now,
            "last_updated": now,
            "workspace_id": workspace_id or "",
            "knowledge_base_id": knowledge_base_id or "",
        }
        await pipeline.hset(doc_info_key, mapping=doc_info)

        # Chunk set per document
        doc_chunks_key = self._doc_chunks_key(
            document_id, workspace_id, knowledge_base_id
        )
        for chunk in chunks:
            chunk_key = self._chunk_key(
                chunk.documentKey, workspace_id, knowledge_base_id
            )
            chunk_data = {
                "chunk_id": chunk.documentKey,
                "document_id": document_id,
                "document_uri": document_uri,
                "chunk_hash": self._calculate_chunk_hash(
                    chunk.text, workspace_id, knowledge_base_id
                ),
                "created_at": now,
            }
            await pipeline.hset(chunk_key, mapping=chunk_data)
            await pipeline.sadd(doc_chunks_key, chunk.documentKey)
            await pipeline.expire(chunk_key, get_envs().REDIS_EXPIRE_TTL)

        await pipeline.expire(doc_info_key, get_envs().REDIS_EXPIRE_TTL)
        await pipeline.expire(doc_chunks_key, get_envs().REDIS_EXPIRE_TTL)
        await pipeline.execute()

        logger.info(
            f"Registered {len(chunks)} chunks for document {document_id} in Redis"
        )

    # ======================== Completion + cleanup ============================

    async def mark_document_completed(
        self,
        document_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Mark entire document as completed and clean up session-tracking keys."""
        now = datetime.now().isoformat()

        # Persistent completion record
        completion_key = self._doc_completion_key(
            document_id, workspace_id, knowledge_base_id
        )
        completion_data = {
            "document_id": document_id,
            "pipeline_completed": "true",
            "completed_at": now,
            "last_updated": now,
            "workspace_id": (workspace_id or ""),
            "knowledge_base_id": (knowledge_base_id or ""),
        }
        await self.redis_client.hset(completion_key, mapping=completion_data)
        await self.redis_client.expire(completion_key, get_envs().REDIS_EXPIRE_TTL)

        # Update doc session status (if exists)
        doc_info_key = self._doc_info_key(document_id, workspace_id, knowledge_base_id)
        if await self.redis_client.exists(doc_info_key):
            await self.redis_client.hset(
                doc_info_key,
                mapping={
                    "pipeline_completed": "true",
                    "pipeline_completed_at": now,
                    "last_updated": now,
                },
            )

        logger.info(f"Marked document {document_id} as fully completed")

        # Cleanup session tracking
        await self._cleanup_document_tracking(
            document_id, workspace_id, knowledge_base_id
        )

    async def _cleanup_document_tracking(
        self,
        document_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Remove session tracking keys (keeps persistent completion record)."""
        try:
            doc_chunks_key = self._doc_chunks_key(
                document_id, workspace_id, knowledge_base_id
            )
            chunk_ids_raw = await self.redis_client.smembers(doc_chunks_key)
            chunk_ids = self._decode_set(chunk_ids_raw)

            pipeline = self.redis_client.pipeline()
            for chunk_id in chunk_ids:
                chunk_key = self._chunk_key(chunk_id, workspace_id, knowledge_base_id)
                await pipeline.delete(chunk_key)

            await pipeline.delete(doc_chunks_key)
            await pipeline.delete(
                self._doc_info_key(document_id, workspace_id, knowledge_base_id)
            )
            await pipeline.execute()

            logger.info(
                f"Cleaned up tracking data for {len(chunk_ids)} chunks in document {document_id}"
            )

        except Exception as e:
            log_error_info(
                logging.WARNING,
                f"Failed to cleanup tracking data for document {document_id}",
                e,
            )

    async def get_processing_stats(
        self,
        document_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> Dict:
        """Return simple doc processing stats (chunk-session only, no entity/relation)."""
        doc_info_key = self._doc_info_key(document_id, workspace_id, knowledge_base_id)
        doc_info = await self.redis_client.hgetall(doc_info_key)
        doc_info = self._decode_dict(doc_info)

        if not doc_info:
            # May have been completed in a previous session
            completion_key = self._doc_completion_key(
                document_id, workspace_id, knowledge_base_id
            )
            completion_data = await self.redis_client.hgetall(completion_key)
            completion_data = self._decode_dict(completion_data)
            if completion_data:
                return {
                    "document_id": document_id,
                    "pipeline_completed": True,
                    "completed_at": completion_data.get("completed_at"),
                    "status": "Previously completed",
                }
            return {"error": "Document not found"}

        total_chunks = self._safe_int(doc_info.get("total_chunks", 0))
        return {
            "document_id": document_id,
            "total_chunks": total_chunks,
            "pipeline_completed": doc_info.get("pipeline_completed", "false") == "true",
            "last_updated": doc_info.get("last_updated"),
        }

    async def reset_document(
        self,
        document_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Reset processing status for a document (testing/debug)."""
        await self._cleanup_document_tracking(
            document_id, workspace_id, knowledge_base_id
        )
        completion_key = self._doc_completion_key(
            document_id, workspace_id, knowledge_base_id
        )
        await self.redis_client.delete(completion_key)
        logger.info(f"Reset processing status for document {document_id}")
