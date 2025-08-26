import time
from typing import List
from unittest.mock import Mock

import pytest
import redis

from hirag_prod.resume_tracker import ResumeTracker

TEST_REDIS_URL = "redis://redis:6379/15"


class MockChunk:
    """Mock chunk object for testing purposes"""

    def __init__(self, chunk_id: str, content: str, document_id: str, uri: str):
        self.id = chunk_id
        self.page_content = content
        self.metadata = Mock()
        self.metadata.document_id = document_id
        self.metadata.uri = uri


def is_redis_available():
    """Check if Redis is available for testing"""
    try:
        client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
        client.ping()
        client.close()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


# Skip all Redis tests if Redis is not available
pytestmark = pytest.mark.skipif(
    not is_redis_available(),
    reason="Redis is not available. Make sure Redis container is running: docker compose up -d redis",
)


@pytest.fixture
def redis_client():
    """Redis client fixture with test database"""
    if not is_redis_available():
        pytest.skip("Redis is not available")

    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)  # Use test DB 15
    yield client
    # Cleanup: flush test database after each test
    try:
        client.flushdb()
        client.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def resume_tracker():
    """Resume tracker fixture with test configuration"""
    # Use unique key prefix for each test to ensure isolation
    unique_prefix = f"test_hirag_{int(time.time() * 1000000)}"
    tracker = ResumeTracker(
        redis_url=TEST_REDIS_URL,
        key_prefix=unique_prefix,
        auto_cleanup=False,  # Disable auto cleanup for testing
    )
    yield tracker
    # Cleanup is handled by redis_client fixture


def get_unique_doc_id(base_name: str = "doc") -> str:
    """Generate unique document ID for test isolation"""
    return f"{base_name}_{int(time.time() * 1000000)}"


@pytest.fixture
def sample_chunks() -> List[MockChunk]:
    """Sample chunks for testing"""
    # Use unique document ID for each test
    doc_id = get_unique_doc_id()
    return [
        MockChunk("chunk_1", "Sample content 1", doc_id, "/path/to/doc.txt"),
        MockChunk("chunk_2", "Sample content 2", doc_id, "/path/to/doc.txt"),
        MockChunk("chunk_3", "Sample content 3", doc_id, "/path/to/doc.txt"),
    ]


class TestRedisConnection:
    """Test Redis connection and basic operations"""

    def test_redis_connection(self, redis_client):
        """Test basic Redis connectivity"""
        # Test ping
        assert redis_client.ping() is True

        # Test basic operations
        redis_client.set("test_key", "test_value")
        assert redis_client.get("test_key") == "test_value"

        # Test key deletion
        redis_client.delete("test_key")
        assert redis_client.get("test_key") is None

    def test_resume_tracker_initialization(self, resume_tracker):
        """Test ResumeTracker initialization and Redis connection"""
        assert resume_tracker.redis_client.ping() is True
        assert resume_tracker.key_prefix.startswith("test_hirag_")


class TestChunkRegistration:
    """Test chunk registration functionality"""

    def test_register_chunks(self, resume_tracker, sample_chunks):
        """Test registering chunks in the tracking system"""
        document_id = sample_chunks[0].metadata.document_id
        document_uri = "/path/to/doc.txt"

        resume_tracker.register_chunks(sample_chunks, document_id, document_uri)

        # Verify document info is stored
        doc_info_key = resume_tracker._doc_info_key(document_id)
        doc_info = resume_tracker.redis_client.hgetall(doc_info_key)

        assert doc_info["document_id"] == document_id
        assert doc_info["document_uri"] == document_uri
        assert int(doc_info["total_chunks"]) == len(sample_chunks)
        assert "created_at" in doc_info

        # Verify chunks are registered
        doc_chunks_key = resume_tracker._doc_chunks_key(document_id)
        registered_chunks = resume_tracker.redis_client.smembers(doc_chunks_key)

        expected_chunk_ids = {chunk.id for chunk in sample_chunks}
        assert registered_chunks == expected_chunk_ids

        # Verify individual chunk data
        for chunk in sample_chunks:
            chunk_key = resume_tracker._chunk_key(chunk.id)
            chunk_data = resume_tracker.redis_client.hgetall(chunk_key)

            assert chunk_data["chunk_id"] == chunk.id
            assert chunk_data["document_id"] == document_id
            assert chunk_data["entity_extraction_completed"] == "false"
            assert chunk_data["relation_extraction_completed"] == "false"

    def test_register_empty_chunks(self, resume_tracker):
        """Test registering empty chunk list"""
        doc_id = get_unique_doc_id("empty")
        resume_tracker.register_chunks([], doc_id, "/empty/path")

        # Should not create any keys
        doc_info_key = resume_tracker._doc_info_key(doc_id)
        assert not resume_tracker.redis_client.exists(doc_info_key)

    def test_register_duplicate_document(self, resume_tracker, sample_chunks):
        """Test registering the same document twice"""
        document_id = sample_chunks[0].metadata.document_id
        document_uri = "/path/to/duplicate.txt"

        # Register first time
        resume_tracker.register_chunks(sample_chunks, document_id, document_uri)

        # Register second time - should skip
        resume_tracker.register_chunks(sample_chunks, document_id, document_uri)

        # Should still have only the original registration
        doc_chunks_key = resume_tracker._doc_chunks_key(document_id)
        registered_chunks = resume_tracker.redis_client.smembers(doc_chunks_key)
        assert len(registered_chunks) == len(sample_chunks)


class TestEntityExtraction:
    """Test entity extraction status tracking"""

    def test_pending_entity_chunks(self, resume_tracker, sample_chunks):
        """Test getting chunks pending entity extraction"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Initially all chunks should be pending
        pending = resume_tracker.get_pending_entity_chunks(sample_chunks)
        assert len(pending) == len(sample_chunks)

        # Mark some chunks as completed
        completed_chunks = sample_chunks[:2]
        resume_tracker.mark_entity_extraction_completed(
            completed_chunks, {"chunk_1": 5, "chunk_2": 3}
        )

        # Now only remaining chunks should be pending
        pending = resume_tracker.get_pending_entity_chunks(sample_chunks)
        assert len(pending) == 1
        assert pending[0].id == "chunk_3"

    def test_mark_entity_extraction_started(self, resume_tracker, sample_chunks):
        """Test marking entity extraction as started"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        resume_tracker.mark_entity_extraction_started(sample_chunks)

        # Verify started timestamp is set
        for chunk in sample_chunks:
            chunk_key = resume_tracker._chunk_key(chunk.id)
            chunk_data = resume_tracker.redis_client.hgetall(chunk_key)
            assert "entity_extraction_started_at" in chunk_data

    def test_mark_entity_extraction_completed(self, resume_tracker, sample_chunks):
        """Test marking entity extraction as completed"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        entity_counts = {"chunk_1": 5, "chunk_2": 3, "chunk_3": 2}
        resume_tracker.mark_entity_extraction_completed(sample_chunks, entity_counts)

        # Verify completion status
        for chunk in sample_chunks:
            chunk_key = resume_tracker._chunk_key(chunk.id)
            chunk_data = resume_tracker.redis_client.hgetall(chunk_key)

            assert chunk_data["entity_extraction_completed"] == "true"
            assert "entity_extraction_completed_at" in chunk_data
            assert chunk_data["entity_count"] == str(entity_counts[chunk.id])

        # Verify chunks are added to completed set
        entity_completed_key = resume_tracker._doc_entity_completed_key(document_id)
        completed_chunks = resume_tracker.redis_client.smembers(entity_completed_key)
        expected_chunk_ids = {chunk.id for chunk in sample_chunks}
        assert completed_chunks == expected_chunk_ids


class TestRelationExtraction:
    """Test relation extraction status tracking"""

    def test_pending_relation_chunks(self, resume_tracker, sample_chunks):
        """Test getting chunks pending relation extraction"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Mark entity extraction completed for all chunks
        resume_tracker.mark_entity_extraction_completed(sample_chunks)

        # All chunks should be pending relation extraction
        pending = resume_tracker.get_pending_relation_chunks(sample_chunks)
        assert len(pending) == len(sample_chunks)

        # Mark some relation extraction completed
        completed_chunks = sample_chunks[:1]
        resume_tracker.mark_relation_extraction_completed(
            completed_chunks, {"chunk_1": 2}
        )

        # Now fewer chunks should be pending
        pending = resume_tracker.get_pending_relation_chunks(sample_chunks)
        assert len(pending) == 2

    def test_relation_requires_entity_completion(self, resume_tracker, sample_chunks):
        """Test that relation extraction requires entity extraction completion"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Without entity extraction completed, no chunks should be pending relation
        pending = resume_tracker.get_pending_relation_chunks(sample_chunks)
        assert len(pending) == 0

        # Complete entity extraction for some chunks
        entity_completed = sample_chunks[:2]
        resume_tracker.mark_entity_extraction_completed(entity_completed)

        # Now only entity-completed chunks should be pending relation
        pending = resume_tracker.get_pending_relation_chunks(sample_chunks)
        assert len(pending) == 2

    def test_mark_relation_extraction_completed(self, resume_tracker, sample_chunks):
        """Test marking relation extraction as completed"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")
        resume_tracker.mark_entity_extraction_completed(sample_chunks)

        relation_counts = {"chunk_1": 3, "chunk_2": 1, "chunk_3": 4}
        resume_tracker.mark_relation_extraction_completed(
            sample_chunks, relation_counts
        )

        # Verify completion status
        for chunk in sample_chunks:
            chunk_key = resume_tracker._chunk_key(chunk.id)
            chunk_data = resume_tracker.redis_client.hgetall(chunk_key)

            assert chunk_data["relation_extraction_completed"] == "true"
            assert "relation_extraction_completed_at" in chunk_data
            assert chunk_data["relation_count"] == str(relation_counts[chunk.id])


class TestDocumentCompletion:
    """Test document completion tracking and cleanup"""

    def test_document_completion_status(self, resume_tracker, sample_chunks):
        """Test checking document completion status"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Initially not complete
        assert not resume_tracker.is_document_complete(document_id)

        # Complete entity extraction only
        resume_tracker.mark_entity_extraction_completed(sample_chunks)
        assert not resume_tracker.is_document_complete(document_id)

        # Complete relation extraction
        resume_tracker.mark_relation_extraction_completed(
            sample_chunks, {"chunk_1": 1, "chunk_2": 2, "chunk_3": 3}
        )
        assert resume_tracker.is_document_complete(document_id)

    def test_mark_document_completed(self, resume_tracker, sample_chunks):
        """Test marking entire document as completed with proper verification"""
        document_id = sample_chunks[0].metadata.document_id
        document_uri = "/path/to/complete_test.txt"

        # Register chunks and partially process them
        resume_tracker.register_chunks(sample_chunks, document_id, document_uri)
        resume_tracker.mark_entity_extraction_completed(
            sample_chunks[:2], {"chunk_1": 5, "chunk_2": 3}
        )

        # Verify document is not complete yet
        assert not resume_tracker.is_document_complete(document_id)

        # Get initial state for verification
        doc_chunks_key = resume_tracker._doc_chunks_key(document_id)
        initial_chunks = resume_tracker.redis_client.smembers(doc_chunks_key)
        assert len(initial_chunks) == len(sample_chunks)

        # Check completion status before cleanup (after calling mark_document_completed,
        # the session tracking data is cleaned up)
        completion_key = resume_tracker._doc_completion_key(document_id)

        # Mark document as completed
        resume_tracker.mark_document_completed(document_id)

        # Verify completion record was created
        completion_data = resume_tracker.redis_client.hgetall(completion_key)
        assert completion_data.get("pipeline_completed") == "true"
        assert "completed_at" in completion_data

        # Verify session tracking data was cleaned up as expected
        remaining_chunks = resume_tracker.redis_client.smembers(doc_chunks_key)
        assert len(remaining_chunks) == 0  # Should be cleaned up

    def test_automatic_cleanup_on_completion(self, sample_chunks):
        """Test automatic cleanup when document processing is complete"""
        # Create a tracker with auto_cleanup enabled for this specific test
        tracker = ResumeTracker(
            redis_url=TEST_REDIS_URL,
            key_prefix="test_hirag_auto",
            auto_cleanup=True,  # Enable auto cleanup for this test
        )

        document_id = sample_chunks[0].metadata.document_id
        tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Complete all processing
        tracker.mark_entity_extraction_completed(
            sample_chunks, {"chunk_1": 5, "chunk_2": 3, "chunk_3": 2}
        )
        tracker.mark_relation_extraction_completed(
            sample_chunks, {"chunk_1": 2, "chunk_2": 1, "chunk_3": 4}
        )

        # After relation completion, cleanup should happen automatically
        # Verify that tracking data is cleaned up
        doc_chunks_key = tracker._doc_chunks_key(document_id)
        assert not tracker.redis_client.exists(doc_chunks_key)


class TestProcessingStats:
    """Test processing statistics functionality"""

    def test_get_processing_stats(self, resume_tracker, sample_chunks):
        """Test getting detailed processing statistics"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")

        # Get initial stats
        stats = resume_tracker.get_processing_stats(document_id)

        assert stats["document_id"] == document_id
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["entity_extraction"]["completed_chunks"] == 0
        assert stats["relation_extraction"]["completed_chunks"] == 0
        assert stats["totals"]["entities"] == 0
        assert stats["totals"]["relations"] == 0
        assert not stats["pipeline_completed"]

        # Complete some entity extraction
        entity_counts = {"chunk_1": 5, "chunk_2": 3}
        resume_tracker.mark_entity_extraction_completed(
            sample_chunks[:2], entity_counts
        )

        stats = resume_tracker.get_processing_stats(document_id)
        assert stats["entity_extraction"]["completed_chunks"] == 2
        assert stats["totals"]["entities"] == 8  # 5 + 3

    def test_get_stats_nonexistent_document(self, resume_tracker):
        """Test getting stats for nonexistent document"""
        stats = resume_tracker.get_processing_stats("nonexistent_doc")
        assert "error" in stats
        assert stats["error"] == "Document not found"


class TestUtilities:
    """Test utility functions and edge cases"""

    def test_reset_document(self, resume_tracker, sample_chunks):
        """Test resetting document processing status"""
        document_id = sample_chunks[0].metadata.document_id
        resume_tracker.register_chunks(sample_chunks, document_id, "/path/to/doc.txt")
        resume_tracker.mark_entity_extraction_completed(sample_chunks)

        # Reset document
        resume_tracker.reset_document(document_id)

        # Verify all tracking data is removed
        doc_info_key = resume_tracker._doc_info_key(document_id)
        doc_chunks_key = resume_tracker._doc_chunks_key(document_id)

        assert not resume_tracker.redis_client.exists(doc_info_key)
        assert not resume_tracker.redis_client.exists(doc_chunks_key)

    def test_key_generation(self, resume_tracker):
        """Test Redis key generation methods"""
        chunk_id = "test_chunk"
        document_id = "test_doc"

        chunk_key = resume_tracker._chunk_key(chunk_id)
        doc_chunks_key = resume_tracker._doc_chunks_key(document_id)
        entity_completed_key = resume_tracker._doc_entity_completed_key(document_id)
        relation_completed_key = resume_tracker._doc_relation_completed_key(document_id)
        doc_info_key = resume_tracker._doc_info_key(document_id)

        prefix = resume_tracker.key_prefix
        assert chunk_key == f"{prefix}:chunk:test_chunk"
        assert doc_chunks_key == f"{prefix}:doc:test_doc:chunks"
        assert entity_completed_key == f"{prefix}:doc:test_doc:entity_completed"
        assert relation_completed_key == f"{prefix}:doc:test_doc:relation_completed"
        assert doc_info_key == f"{prefix}:doc:test_doc:info"

    def test_chunk_hash_calculation(self, resume_tracker):
        """Test chunk content hash calculation"""
        content1 = "This is test content"
        content2 = "This is different content"

        hash1 = resume_tracker._calculate_chunk_hash(content1)
        hash2 = resume_tracker._calculate_chunk_hash(content1)  # Same content
        hash3 = resume_tracker._calculate_chunk_hash(content2)  # Different content

        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != hash3  # Different content should produce different hash
        assert len(hash1) == 32  # MD5 hash should be 32 characters


# Integration test
@pytest.mark.asyncio
async def test_full_processing_workflow(redis_client, sample_chunks):
    """Test complete processing workflow from registration to completion"""
    # Create a tracker with auto_cleanup enabled for this integration test
    tracker = ResumeTracker(
        redis_url=TEST_REDIS_URL,
        key_prefix="test_hirag_workflow",
        auto_cleanup=True,  # Enable auto cleanup for this test
    )

    # Use unique document ID to avoid conflicts with other tests
    document_id = f"workflow_doc_{int(time.time())}"
    document_uri = "/workflow/test.txt"

    # Update chunk metadata to match the document_id
    for chunk in sample_chunks:
        chunk.metadata.document_id = document_id

    # Step 1: Register chunks
    tracker.register_chunks(sample_chunks, document_id, document_uri)

    # Step 2: Process entity extraction
    pending_entities = tracker.get_pending_entity_chunks(sample_chunks)
    assert len(pending_entities) == len(sample_chunks)

    tracker.mark_entity_extraction_started(pending_entities)
    entity_counts = {chunk.id: i + 1 for i, chunk in enumerate(pending_entities)}
    tracker.mark_entity_extraction_completed(pending_entities, entity_counts)

    # Step 3: Process relation extraction
    pending_relations = tracker.get_pending_relation_chunks(sample_chunks)
    assert len(pending_relations) == len(sample_chunks)

    tracker.mark_relation_extraction_started(pending_relations)
    relation_counts = {chunk.id: i + 1 for i, chunk in enumerate(pending_relations)}
    tracker.mark_relation_extraction_completed(pending_relations, relation_counts)

    # Step 4: Verify completion and cleanup
    # Note: With auto_cleanup enabled, the document data will be cleaned up immediately
    # so we can't check is_document_complete after completion

    # The document should be automatically cleaned up
    doc_chunks_key = tracker._doc_chunks_key(document_id)
    assert not tracker.redis_client.exists(doc_chunks_key)
