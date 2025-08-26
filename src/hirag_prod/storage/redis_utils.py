#!/usr/bin/env python3
"""Redis Storage Management Utilities"""

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/chatbot/.env")

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/2")
DEFAULT_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "hirag")


@dataclass
class DocumentStatus:
    """Document processing status data structure"""

    document_id: str
    active_session: bool = False
    completed: bool = False
    total_chunks: int = 0
    entity_completed: int = 0
    relation_completed: int = 0
    completion_info: Optional[Dict] = None
    session_info: Optional[Dict] = None

    @property
    def entity_progress(self) -> float:
        """Calculate entity extraction progress percentage"""
        return (
            (self.entity_completed / self.total_chunks * 100)
            if self.total_chunks > 0
            else 0.0
        )

    @property
    def relation_progress(self) -> float:
        """Calculate relation extraction progress percentage"""
        return (
            (self.relation_completed / self.total_chunks * 100)
            if self.total_chunks > 0
            else 0.0
        )

    @property
    def status_icon(self) -> str:
        """Get appropriate status icon"""
        if self.completed:
            return "✅"
        elif self.active_session:
            return "🔄"
        else:
            return "❓"

    @property
    def status_text(self) -> str:
        """Get human-readable status text"""
        if self.completed:
            return "COMPLETED"
        elif self.active_session:
            return "IN PROGRESS"
        else:
            return "UNKNOWN"


class RedisStorageManager:
    """
    Elegant Redis storage manager for HiRAG document processing state.

    Provides comprehensive functionality for:
    - Document status inspection
    - State management and cleanup
    - Debugging and troubleshooting
    - Export and reporting capabilities
    """

    def __init__(
        self, redis_url: str = DEFAULT_REDIS_URL, key_prefix: str = DEFAULT_KEY_PREFIX
    ):
        """
        Initialize Redis storage manager.

        Args:
            redis_url: Redis connection URL
            key_prefix: Key prefix for all Redis operations
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._client = None

    @property
    def client(self) -> redis.Redis:
        """Lazy-loaded Redis client with connection pooling"""
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            self._verify_connection()
        return self._client

    def _verify_connection(self) -> None:
        """Verify Redis connection is working"""
        try:
            self.client.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")

    @contextmanager
    def _pipeline(self):
        """Context manager for Redis pipeline operations"""
        pipeline = self.client.pipeline()
        try:
            yield pipeline
            pipeline.execute()
        except Exception as e:
            logger.error(f"Pipeline operation failed: {e}")
            raise

    def _extract_document_ids(self) -> Set[str]:
        """Extract all tracked document IDs from Redis keys"""
        doc_ids = set()

        # Extract from document info keys
        info_pattern = f"{self.key_prefix}:*doc:*:info"
        for key in self.client.keys(info_pattern):
            parts = key.split(":")
            # find 'doc' and take the next token
            if "doc" in parts:
                try:
                    idx = parts.index("doc")
                    doc_ids.add(parts[idx + 1])
                except Exception:
                    continue

        # Extract from completion keys
        completion_pattern = f"{self.key_prefix}:*completed:*"
        for key in self.client.keys(completion_pattern):
            parts = key.split(":")
            if "completed" in parts:
                try:
                    idx = parts.index("completed")
                    doc_ids.add(parts[idx + 1])
                except Exception:
                    continue

        return doc_ids

    def list_documents(self) -> List[str]:
        """List all tracked document IDs"""
        return sorted(self._extract_document_ids())

    def get_document_status(self, document_id: str) -> DocumentStatus:
        """
        Get comprehensive status for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            DocumentStatus object with complete status information
        """
        status = DocumentStatus(document_id=document_id)

        # Check completion status
        completion_keys = self.client.keys(
            f"{self.key_prefix}:*completed:{document_id}"
        )
        if completion_keys:
            completion_data = self.client.hgetall(completion_keys[0])
            if completion_data:
                status.completed = (
                    completion_data.get("pipeline_completed", "false") == "true"
                )
                status.completion_info = completion_data

        # Check active session status
        doc_info_keys = self.client.keys(f"{self.key_prefix}:*doc:{document_id}:info")
        if doc_info_keys:
            session_data = self.client.hgetall(doc_info_keys[0])
            if session_data:
                status.active_session = True
                status.session_info = session_data
                status.total_chunks = int(session_data.get("total_chunks", 0))

                # Get completion counts
                entity_keys = self.client.keys(
                    f"{self.key_prefix}:*doc:{document_id}:entity_completed"
                )
                relation_keys = self.client.keys(
                    f"{self.key_prefix}:*doc:{document_id}:relation_completed"
                )

                if entity_keys:
                    status.entity_completed = self.client.scard(entity_keys[0])
                if relation_keys:
                    status.relation_completed = self.client.scard(relation_keys[0])

        return status

    def print_document_status(self, document_id: str) -> None:
        """Print formatted document status information"""
        status = self.get_document_status(document_id)

        print(f"\n📄 Document: {document_id}")
        print("=" * 60)
        print(f"{status.status_icon} Status: {status.status_text}")

        if status.completion_info and status.completed:
            completed_at = status.completion_info.get("completed_at", "unknown")
            print(f"   Completed at: {completed_at}")

        if status.active_session and status.total_chunks > 0:
            print(f"\n📊 Progress:")
            print(f"   Total chunks: {status.total_chunks}")
            print(
                f"   Entity extraction: {status.entity_completed}/{status.total_chunks} ({status.entity_progress:.1f}%)"
            )
            print(
                f"   Relation extraction: {status.relation_completed}/{status.total_chunks} ({status.relation_progress:.1f}%)"
            )

        if status.session_info:
            print(f"\n🕒 Session info:")
            print(f"   Created: {status.session_info.get('created_at', 'unknown')}")
            print(f"   Updated: {status.session_info.get('last_updated', 'unknown')}")

    def list_all_documents(self) -> None:
        """Display formatted list of all documents with status summary"""
        doc_ids = self.list_documents()

        if not doc_ids:
            print("📭 No documents found in Redis")
            return

        print(f"📋 Found {len(doc_ids)} documents:")
        print("=" * 80)

        for doc_id in doc_ids:
            status = self.get_document_status(doc_id)

            # Format chunks info
            chunks_info = ""
            if status.total_chunks > 0:
                chunks_info = f" ({status.entity_completed}/{status.total_chunks} entities, {status.relation_completed}/{status.total_chunks} relations)"

            # Truncate long document IDs for display
            display_id = doc_id[:50] if len(doc_id) > 50 else doc_id
            print(
                f"{status.status_icon} {display_id:<50} {status.status_text}{chunks_info}"
            )

    def reset_document(self, document_id: str) -> int:
        """
        Reset all tracking data for a specific document.

        Args:
            document_id: Document identifier to reset

        Returns:
            Number of keys deleted
        """
        logger.info(f"Resetting document: {document_id}")

        keys_to_delete = []

        # Collect document-related keys
        patterns = [
            f"{self.key_prefix}:*doc:{document_id}:*",
            f"{self.key_prefix}:*completed:{document_id}",
        ]

        for pattern in patterns:
            keys_to_delete.extend(self.client.keys(pattern))

        # Collect chunk keys
        doc_chunks_keys = self.client.keys(
            f"{self.key_prefix}:*doc:{document_id}:chunks"
        )
        if doc_chunks_keys:
            chunk_ids = self.client.smembers(doc_chunks_keys[0])
            for chunk_id in chunk_ids:
                # delete all scoped chunk keys
                chunk_keys = self.client.keys(f"{self.key_prefix}:*chunk:{chunk_id}")
                keys_to_delete.extend(chunk_keys)

        # Delete all collected keys
        deleted_count = 0
        if keys_to_delete:
            deleted_count = self.client.delete(*keys_to_delete)
            logger.info(f"Deleted {deleted_count} keys for document {document_id}")
        else:
            logger.info(f"No keys found for document {document_id}")

        return deleted_count

    def cleanup_completed_sessions(self) -> int:
        """
        Clean up session data for completed documents.

        Returns:
            Number of documents cleaned up
        """
        logger.info("Cleaning up completed session data...")

        doc_ids = self.list_documents()
        cleaned_count = 0

        for doc_id in doc_ids:
            status = self.get_document_status(doc_id)
            if status.completed and status.active_session:
                # Keep completion record, clean session data
                session_keys = [
                    f"{self.key_prefix}:doc:{doc_id}:info",
                    f"{self.key_prefix}:doc:{doc_id}:chunks",
                    f"{self.key_prefix}:doc:{doc_id}:entity_completed",
                    f"{self.key_prefix}:doc:{doc_id}:relation_completed",
                ]

                # Delete chunk keys
                doc_chunks_key = f"{self.key_prefix}:doc:{doc_id}:chunks"
                if self.client.exists(doc_chunks_key):
                    chunk_ids = self.client.smembers(doc_chunks_key)
                    for chunk_id in chunk_ids:
                        session_keys.append(f"{self.key_prefix}:chunk:{chunk_id}")

                if session_keys:
                    self.client.delete(*session_keys)
                    cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} completed sessions")
        return cleaned_count

    def export_status(
        self, output_path: Union[str, Path] = "redis_status.json"
    ) -> Dict:
        """
        Export comprehensive status information to JSON file.

        Args:
            output_path: Path to output JSON file

        Returns:
            Exported data dictionary
        """
        doc_ids = self.list_documents()

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "total_documents": len(doc_ids),
            "documents": {},
        }

        for doc_id in doc_ids:
            status = self.get_document_status(doc_id)
            export_data["documents"][doc_id] = {
                "status": status.status_text,
                "completed": status.completed,
                "active_session": status.active_session,
                "total_chunks": status.total_chunks,
                "entity_completed": status.entity_completed,
                "relation_completed": status.relation_completed,
                "entity_progress": round(status.entity_progress, 2),
                "relation_progress": round(status.relation_progress, 2),
                "completion_info": status.completion_info,
                "session_info": status.session_info,
            }

        # Write to file
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported status to {output_path}")
        return export_data

    def get_redis_info(self) -> Dict:
        """Get Redis server information and statistics"""
        try:
            info = self.client.info()
            total_keys = self.client.dbsize()
            hirag_keys = len(self.client.keys(f"{self.key_prefix}:*"))

            return {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_keys": total_keys,
                "hirag_keys": hirag_keys,
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"error": str(e)}


def main():
    """Command-line interface for Redis storage management"""
    import sys

    if len(sys.argv) < 2:
        print("Redis Storage Manager - HiRAG Document Processing")
        print("=" * 50)
        print("Usage: python redis_utils.py <command> [args...]")
        print("\nCommands:")
        print("  list                     - List all tracked documents")
        print("  status <doc_id>          - Show detailed document status")
        print("  reset <doc_id>           - Reset document tracking data")
        print("  cleanup                  - Clean up completed session data")
        print("  export [filename]        - Export status to JSON file")
        print("  info                     - Show Redis server information")
        print("\nEnvironment variables:")
        print("  REDIS_URL               - Redis connection URL")
        print("  REDIS_KEY_PREFIX        - Key prefix for tracking")
        return

    # Initialize storage manager
    manager = RedisStorageManager()
    command = sys.argv[1].lower()

    try:
        if command == "list":
            manager.list_all_documents()

        elif command == "status":
            if len(sys.argv) < 3:
                print("❌ Error: Please provide document ID")
                return
            doc_id = sys.argv[2]
            manager.print_document_status(doc_id)

        elif command == "reset":
            if len(sys.argv) < 3:
                print("❌ Error: Please provide document ID")
                return
            doc_id = sys.argv[2]
            deleted = manager.reset_document(doc_id)
            print(f"✅ Reset document: {doc_id} (deleted {deleted} keys)")

        elif command == "cleanup":
            cleaned = manager.cleanup_completed_sessions()
            print(f"✅ Cleaned up {cleaned} completed sessions")

        elif command == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "redis_status.json"
            data = manager.export_status(filename)
            print(f"✅ Exported {data['total_documents']} documents to {filename}")

        elif command == "info":
            info = manager.get_redis_info()
            print("📊 Redis Server Information:")
            print("=" * 30)
            for key, value in info.items():
                print(f"  {key}: {value}")

        else:
            print(f"❌ Unknown command: {command}")

    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
