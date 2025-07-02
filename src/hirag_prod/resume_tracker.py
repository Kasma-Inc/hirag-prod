import sqlite3
import logging
from typing import List, Set, Dict, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionType(Enum):
    """Extraction type enumeration"""
    ENTITY = "entity"
    RELATION = "relation"


class ChunkResumeTracker:
    """
    SQLite-based chunk-level processing status tracker for efficient resume functionality.
    
    Tracks the processing status of each chunk for:
    - Entity extraction completion
    - Relation extraction completion
    - Processing timestamps and metadata
    """
    
    # Constants
    DEFAULT_CLEANUP_DAYS = 30
    
    # Field mappings for different extraction types
    EXTRACTION_FIELDS = {
        ExtractionType.ENTITY: {
            'completed': 'entity_extraction_completed',
            'started_at': 'entity_extraction_started_at',
            'completed_at': 'entity_extraction_completed_at',
            'count': 'entity_count',
            'doc_completed_field': 'entity_completed_chunks'
        },
        ExtractionType.RELATION: {
            'completed': 'relation_extraction_completed',
            'started_at': 'relation_extraction_started_at',
            'completed_at': 'relation_extraction_completed_at',
            'count': 'relation_count',
            'doc_completed_field': 'relation_completed_chunks'
        }
    }
    
    def __init__(self, db_path: str):
        """Initialize the resume tracker with SQLite backend"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Chunk status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_status (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_uri TEXT NOT NULL,
                    chunk_hash TEXT,
                    
                    entity_extraction_completed BOOLEAN DEFAULT FALSE,
                    entity_extraction_started_at TIMESTAMP,
                    entity_extraction_completed_at TIMESTAMP,
                    entity_count INTEGER DEFAULT 0,
                    
                    relation_extraction_completed BOOLEAN DEFAULT FALSE,
                    relation_extraction_started_at TIMESTAMP,
                    relation_extraction_completed_at TIMESTAMP,
                    relation_count INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_document_id ON chunk_status(document_id)",
                "CREATE INDEX IF NOT EXISTS idx_document_uri ON chunk_status(document_uri)",
                "CREATE INDEX IF NOT EXISTS idx_entity_completed ON chunk_status(entity_extraction_completed)",
                "CREATE INDEX IF NOT EXISTS idx_relation_completed ON chunk_status(relation_extraction_completed)"
            ]
            for index_sql in indexes:
                conn.execute(index_sql)
            
            # Document status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_status (
                    document_id TEXT PRIMARY KEY,
                    document_uri TEXT UNIQUE NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    entity_completed_chunks INTEGER DEFAULT 0,
                    relation_completed_chunks INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pipeline_completed BOOLEAN DEFAULT FALSE,
                    pipeline_completed_at TIMESTAMP
                )
            """)
            conn.commit()
    
    def _calculate_chunk_hash(self, chunk_content: str) -> str:
        """Calculate a hash for chunk content to detect changes"""
        return hashlib.md5(chunk_content.encode()).hexdigest()
    
    def _get_chunk_ids_with_status(self, chunks: List, extraction_type: ExtractionType) -> Set[str]:
        """Get chunk IDs that have completed the specified extraction type"""
        if not chunks:
            return set()
        
        chunk_ids = [chunk.id for chunk in chunks]
        placeholders = ','.join('?' * len(chunk_ids))
        completed_field = self.EXTRACTION_FIELDS[extraction_type]['completed']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT chunk_id FROM chunk_status 
                WHERE chunk_id IN ({placeholders}) AND {completed_field} = TRUE
            """, chunk_ids)
            return {row[0] for row in cursor}
    
    def _update_document_completion_count(self, document_id: str, extraction_type: ExtractionType):
        """Update document-level completion count for the specified extraction type"""
        fields = self.EXTRACTION_FIELDS[extraction_type]
        completed_field = fields['completed']
        doc_field = fields['doc_completed_field']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT COUNT(*) FROM chunk_status 
                WHERE document_id = ? AND {completed_field} = TRUE
            """, (document_id,))
            completed_count = cursor.fetchone()[0]
            
            conn.execute(f"""
                UPDATE document_status 
                SET {doc_field} = ?, last_updated = CURRENT_TIMESTAMP
                WHERE document_id = ?
            """, (completed_count, document_id))
            conn.commit()
    
    def register_chunks(self, chunks: List, document_id: str, document_uri: str) -> None:
        """Register chunks in the tracking system (preserving existing status)"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if document already exists
            cursor = conn.execute("SELECT COUNT(*) FROM document_status WHERE document_id = ?", (document_id,))
            if cursor.fetchone()[0] > 0:
                logger.debug(f"Document {document_id} already registered, skipping chunk registration")
                return
            
            # Register new chunks
            now = datetime.now()
            chunk_data = [
                (chunk.id, document_id, document_uri, self._calculate_chunk_hash(chunk.page_content), now)
                for chunk in chunks
            ]
            
            conn.executemany("""
                INSERT OR IGNORE INTO chunk_status 
                (chunk_id, document_id, document_uri, chunk_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, chunk_data)
            
            conn.execute("""
                INSERT OR IGNORE INTO document_status 
                (document_id, document_uri, total_chunks, last_updated)
                VALUES (?, ?, ?, ?)
            """, (document_id, document_uri, len(chunks), now))
            
            conn.commit()
            logger.info(f"Registered {len(chunks)} new chunks for document {document_id}")
    
    def get_pending_chunks(self, chunks: List, extraction_type: ExtractionType) -> List:
        """Get chunks that need the specified extraction type"""
        if not chunks:
            return []
        
        completed_chunk_ids = self._get_chunk_ids_with_status(chunks, extraction_type)
        
        # For relations, also check that entity extraction is completed
        if extraction_type == ExtractionType.RELATION:
            entity_completed_ids = self._get_chunk_ids_with_status(chunks, ExtractionType.ENTITY)
            pending_chunks = [
                chunk for chunk in chunks 
                if chunk.id in entity_completed_ids and chunk.id not in completed_chunk_ids
            ]
        else:
            pending_chunks = [chunk for chunk in chunks if chunk.id not in completed_chunk_ids]
        
        logger.info(f"Found {len(pending_chunks)} chunks pending {extraction_type.value} extraction out of {len(chunks)} total")
        return pending_chunks
    
    def get_pending_entity_chunks(self, chunks: List) -> List:
        """Get chunks that need entity extraction"""
        return self.get_pending_chunks(chunks, ExtractionType.ENTITY)
    
    def get_pending_relation_chunks(self, chunks: List) -> List:
        """Get chunks that need relation extraction"""
        return self.get_pending_chunks(chunks, ExtractionType.RELATION)
    
    def mark_extraction_started(self, chunks: List, extraction_type: ExtractionType) -> None:
        """Mark chunks as having started the specified extraction"""
        if not chunks:
            return
        
        started_at_field = self.EXTRACTION_FIELDS[extraction_type]['started_at']
        
        with sqlite3.connect(self.db_path) as conn:
            chunk_data = [(datetime.now(), chunk.id) for chunk in chunks]
            conn.executemany(f"""
                UPDATE chunk_status 
                SET {started_at_field} = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, chunk_data)
            conn.commit()
        
        logger.info(f"Marked {len(chunks)} chunks as {extraction_type.value} extraction started")
    
    def mark_extraction_completed(self, chunks: List, extraction_type: ExtractionType, 
                                 counts: Optional[Dict[str, int]] = None) -> None:
        """Mark chunks as having completed the specified extraction"""
        if not chunks:
            return
        
        counts = counts or {}
        fields = self.EXTRACTION_FIELDS[extraction_type]
        
        with sqlite3.connect(self.db_path) as conn:
            chunk_data = [
                (datetime.now(), counts.get(chunk.id, 0), chunk.id) 
                for chunk in chunks
            ]
            conn.executemany(f"""
                UPDATE chunk_status 
                SET {fields['completed']} = TRUE,
                    {fields['completed_at']} = ?,
                    {fields['count']} = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, chunk_data)
            conn.commit()
        
        # Update document-level tracking
        if chunks:
            document_id = chunks[0].metadata.document_id
            self._update_document_completion_count(document_id, extraction_type)
        
        logger.info(f"Marked {len(chunks)} chunks as {extraction_type.value} extraction completed")
    
    # Convenience methods for backward compatibility
    def mark_entity_extraction_started(self, chunks: List) -> None:
        """Mark chunks as having started entity extraction"""
        self.mark_extraction_started(chunks, ExtractionType.ENTITY)
    
    def mark_entity_extraction_completed(self, chunks: List, entity_counts: Dict[str, int] = None) -> None:
        """Mark chunks as having completed entity extraction"""
        self.mark_extraction_completed(chunks, ExtractionType.ENTITY, entity_counts)
    
    def mark_relation_extraction_started(self, chunks: List) -> None:
        """Mark chunks as having started relation extraction"""
        self.mark_extraction_started(chunks, ExtractionType.RELATION)
    
    def mark_relation_extraction_completed(self, chunks: List, relation_counts: Dict[str, int] = None) -> None:
        """Mark chunks as having completed relation extraction"""
        self.mark_extraction_completed(chunks, ExtractionType.RELATION, relation_counts)
    
    def is_document_complete(self, document_id: str) -> bool:
        """Check if entire document processing is complete"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT total_chunks, entity_completed_chunks, relation_completed_chunks, pipeline_completed
                FROM document_status WHERE document_id = ?
            """, (document_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            total, entity_done, relation_done, pipeline_done = row
            return pipeline_done or (total > 0 and entity_done == total and relation_done == total)
    
    def mark_document_completed(self, document_id: str) -> None:
        """Mark entire document as completed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE document_status 
                SET pipeline_completed = TRUE, 
                    pipeline_completed_at = CURRENT_TIMESTAMP,
                    last_updated = CURRENT_TIMESTAMP
                WHERE document_id = ?
            """, (document_id,))
            conn.commit()
        
        logger.info(f"Marked document {document_id} as fully completed")
    
    def get_processing_stats(self, document_id: str) -> Dict:
        """Get detailed processing statistics for a document"""
        with sqlite3.connect(self.db_path) as conn:
            # Document-level stats
            cursor = conn.execute("""
                SELECT total_chunks, entity_completed_chunks, relation_completed_chunks, 
                       pipeline_completed, last_updated
                FROM document_status WHERE document_id = ?
            """, (document_id,))
            doc_row = cursor.fetchone()
            
            if not doc_row:
                return {"error": "Document not found"}
            
            total, entity_done, relation_done, pipeline_done, last_updated = doc_row
            
            # Chunk-level details
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN entity_extraction_completed THEN 1 ELSE 0 END) as entity_completed,
                    SUM(CASE WHEN relation_extraction_completed THEN 1 ELSE 0 END) as relation_completed,
                    SUM(entity_count) as total_entities,
                    SUM(relation_count) as total_relations
                FROM chunk_status WHERE document_id = ?
            """, (document_id,))
            chunk_stats = cursor.fetchone()
            
            return {
                "document_id": document_id,
                "total_chunks": total,
                "entity_extraction": {
                    "completed_chunks": entity_done,
                    "progress": f"{entity_done}/{total}",
                    "percentage": (entity_done / total * 100) if total > 0 else 0
                },
                "relation_extraction": {
                    "completed_chunks": relation_done,
                    "progress": f"{relation_done}/{total}",
                    "percentage": (relation_done / total * 100) if total > 0 else 0
                },
                "totals": {
                    "entities": chunk_stats[3] if chunk_stats else 0,
                    "relations": chunk_stats[4] if chunk_stats else 0
                },
                "pipeline_completed": pipeline_done,
                "last_updated": last_updated
            }
    
    def reset_document(self, document_id: str) -> None:
        """Reset processing status for a document (for testing/debugging)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunk_status WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_status WHERE document_id = ?", (document_id,))
            conn.commit()
        
        logger.info(f"Reset processing status for document {document_id}")
    
    def cleanup_old_entries(self, days: int = DEFAULT_CLEANUP_DAYS) -> None:
        """Clean up tracking entries older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                DELETE FROM chunk_status 
                WHERE created_at < datetime('now', '-{days} days')
            """)
            
            conn.execute(f"""
                DELETE FROM document_status 
                WHERE last_updated < datetime('now', '-{days} days')
            """)
            
            conn.commit()
        
        logger.info(f"Cleaned up tracking entries older than {days} days") 