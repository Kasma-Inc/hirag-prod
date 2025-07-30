import asyncio
import os
import sys
import tempfile
import pytest
import pytest_asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hirag_prod.hirag import HiRAG, HiRAGConfig, ProcessingMetrics
from hirag_prod._llm import ChatCompletion, create_embedding_service
from hirag_prod.storage import LanceDB, NetworkXGDB, RetrievalStrategyProvider


class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    async def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Mock embedding creation that returns dummy vectors"""
        self.call_count += 1
        # Return dummy embeddings as numpy arrays (4096 dimensions to match .env)
        return [np.array([0.1] * 4096) for _ in texts]


class MockChatCompletion:
    """Mock chat completion service for testing"""
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Mock completion that returns a dummy response"""
        return "This is a mock response."


@pytest_asyncio.fixture
async def temp_hirag():
    """Create a temporary HiRAG instance for testing"""
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_db_path = os.path.join(temp_dir, "test_hirag.db")
        graph_db_path = os.path.join(temp_dir, "test_hirag.gpickle")
        
        # Create test configuration
        config = HiRAGConfig(
            vector_db_path=vector_db_path,
            graph_db_path=graph_db_path,
            redis_url="redis://localhost:6379/15",  # Use test database
            redis_key_prefix="test_hirag",
        )
        
        # Create a mock resume tracker that doesn't require Redis
        class MockResumeTracker:
            def is_document_already_completed(self, document_id):
                return False
            def register_chunks(self, chunks, document_id, document_uri):
                pass
            def mark_document_completed(self, document_id):
                pass
            def get_pending_entity_chunks(self, chunks):
                return chunks
            def get_pending_relation_chunks(self, chunks):
                return chunks
            def mark_entity_extraction_started(self, chunks):
                pass
            def mark_entity_extraction_completed(self, chunks, entity_counts):
                pass
            def mark_relation_extraction_started(self, chunks):
                pass
            def mark_relation_extraction_completed(self, chunks, relation_counts):
                pass
        
        # Create HiRAG instance with mock resume tracker
        hirag = await HiRAG.create(config=config, resume_tracker=MockResumeTracker())
        
        # Replace embedding service and chat service with mocks
        mock_embedding_service = MockEmbeddingService()
        mock_chat_service = MockChatCompletion()
        
        hirag.embedding_service = mock_embedding_service
        hirag.chat_service = mock_chat_service
        
        # Reinitialize the vector database with the mock embedding service
        from hirag_prod.storage import LanceDB, RetrievalStrategyProvider
        vdb = await LanceDB.create(
            embedding_func=mock_embedding_service.create_embeddings,
            db_url=vector_db_path,
            strategy_provider=RetrievalStrategyProvider(),
        )
        
        # Update the storage manager
        hirag._storage.vdb = vdb
        await hirag._storage._initialize_chats_table()
        
        try:
            yield hirag
        finally:
            # Cleanup
            await hirag.clean_up()


@pytest.mark.asyncio
async def test_insert_chat_message_basic(temp_hirag):
    """Test basic chat message insertion"""
    hirag = temp_hirag
    
    # Test data
    chat_id = "test_chat_001"
    role = "user"
    content = "Hello, how are you today?"
    
    # Insert chat message
    metrics = await hirag.insert_chat_to_kb(
        chat_id=chat_id,
        role=role,
        content=content
    )
    
    # Verify metrics
    assert isinstance(metrics, ProcessingMetrics)
    assert metrics.processed_chats == 1
    assert metrics.processing_time > 0


@pytest.mark.asyncio
async def test_insert_multiple_chat_messages(temp_hirag):
    """Test inserting multiple chat messages in a conversation"""
    hirag = temp_hirag
    
    chat_id = "test_chat_002"
    
    # Conversation messages
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! A common example is email spam detection. The system learns from thousands of emails labeled as spam or not spam, and then can automatically classify new emails."},
    ]
    
    # Insert all messages and track expected processed chunks
    expected_processed_chats = 0
    for msg in messages:
        expected_processed_chats += 1
        metrics = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=msg["role"],
            content=msg["content"]
        )
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.processed_chats == expected_processed_chats


@pytest.mark.asyncio
async def test_search_chat_history_basic(temp_hirag):
    """Test basic chat history search"""
    hirag = temp_hirag
    
    chat_id = "test_chat_003"
    
    # Insert some messages first
    messages = [
        {"role": "user", "content": "I need help with Python programming"},
        {"role": "assistant", "content": "I'd be happy to help you with Python! What specific topic or problem are you working on?"},
        {"role": "user", "content": "How do I create a list in Python?"},
        {"role": "assistant", "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3] or my_list = ['apple', 'banana', 'orange']"},
    ]
    
    for msg in messages:
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=msg["role"],
            content=msg["content"]
        )
    
    # Search for messages about Python
    results = await hirag.search_chat_history(
        user_query="Python programming",
        chat_id=chat_id
    )
    
    # Verify results
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check that results contain expected fields
    for result in results:
        assert "text" in result
        assert "role" in result
        assert "content" in result


@pytest.mark.asyncio
async def test_search_chat_history_with_role_filter(temp_hirag):
    """Test chat history search with role filtering"""
    hirag = temp_hirag
    
    chat_id = "test_chat_004"
    
    # Insert messages with different roles
    messages = [
        {"role": "user", "content": "What is the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data. You can check weather apps or websites for current conditions."},
        {"role": "tool", "content": "Weather API called successfully. Temperature: 72°F, Conditions: Sunny"},
        {"role": "assistant", "content": "Based on the weather data, it's currently 72°F and sunny today!"},
    ]
    
    for msg in messages:
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=msg["role"],
            content=msg["content"]
        )
    
    # Search for assistant messages only
    assistant_results = await hirag.search_chat_history(
        user_query="weather",
        chat_id=chat_id,
        role="assistant"
    )
    
    # Verify that only assistant messages are returned
    assert isinstance(assistant_results, list)
    for result in assistant_results:
        assert result["role"] == "assistant"
    
    # Search for user messages only
    user_results = await hirag.search_chat_history(
        user_query="weather",
        chat_id=chat_id,
        role="user"
    )
    
    # Verify that only user messages are returned
    assert isinstance(user_results, list)
    for result in user_results:
        assert result["role"] == "user"


@pytest.mark.asyncio
async def test_search_different_chat_sessions(temp_hirag):
    """Test that search is isolated between different chat sessions"""
    hirag = temp_hirag
    
    # Insert messages in different chat sessions
    chat_id_1 = "test_chat_005_session_1"
    chat_id_2 = "test_chat_005_session_2"
    
    # Chat session 1 - about cooking
    await hirag.insert_chat_to_kb(
        chat_id=chat_id_1,
        role="user",
        content="How do I make pasta?"
    )
    await hirag.insert_chat_to_kb(
        chat_id=chat_id_1,
        role="assistant",
        content="To make pasta, boil water, add salt, then add pasta and cook according to package directions."
    )
    
    # Chat session 2 - about programming
    await hirag.insert_chat_to_kb(
        chat_id=chat_id_2,
        role="user",
        content="How do I write a function in Python?"
    )
    await hirag.insert_chat_to_kb(
        chat_id=chat_id_2,
        role="assistant",
        content="To write a function in Python, use the 'def' keyword: def my_function(): pass"
    )
    
    # Search chat session 1 for cooking content
    cooking_results = await hirag.search_chat_history(
        user_query="pasta cooking",
        chat_id=chat_id_1
    )
    
    # Search chat session 2 for programming content
    programming_results = await hirag.search_chat_history(
        user_query="Python function",
        chat_id=chat_id_2
    )
    
    # Verify results are isolated to their respective sessions
    assert len(cooking_results) > 0
    assert len(programming_results) > 0
    
    # Search for programming content in cooking session should return fewer/no results
    wrong_session_results = await hirag.search_chat_history(
        user_query="Python function",
        chat_id=chat_id_1
    )
    
    # Should return fewer results since the content doesn't match the session
    assert len(wrong_session_results) <= len(programming_results)


@pytest.mark.asyncio
async def test_search_chat_history_topk_topn_parameters(temp_hirag):
    """Test topk and topn parameters in chat search"""
    hirag = temp_hirag
    
    chat_id = "test_chat_006"
    
    # Insert many messages about the same topic
    topics = [
        "What is artificial intelligence?",
        "How does AI work?",
        "What are the types of AI?",
        "Can AI replace humans?",
        "What is machine learning in AI?",
        "How is AI used in healthcare?",
        "What are the benefits of AI?",
        "What are the risks of AI?",
    ]
    
    for i, topic in enumerate(topics):
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="user",
            content=topic
        )
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="assistant",
            content=f"Answer {i+1}: {topic} - This is a comprehensive answer about the topic."
        )
    
    # Test with different topk values
    results_topk_3 = await hirag.search_chat_history(
        user_query="artificial intelligence AI",
        chat_id=chat_id,
        topk=3
    )
    
    results_topk_5 = await hirag.search_chat_history(
        user_query="artificial intelligence AI",
        chat_id=chat_id,
        topk=5
    )
    
    # Verify that topk limits the results
    assert len(results_topk_3) <= 3
    assert len(results_topk_5) <= 5
    assert len(results_topk_3) <= len(results_topk_5)


@pytest.mark.asyncio
async def test_error_handling_invalid_chat_id(temp_hirag):
    """Test error handling for invalid chat_id"""
    hirag = temp_hirag
    
    # Test with empty chat_id
    with pytest.raises(ValueError, match="chat_id must be provided"):
        await hirag.search_chat_history(
            user_query="test query",
            chat_id=""
        )
    
    # Test inserting with empty chat_id
    with pytest.raises(ValueError):
        await hirag.insert_chat_to_kb(
            chat_id="",
            role="user",
            content="test content"
        )


@pytest.mark.asyncio
async def test_error_handling_invalid_role(temp_hirag):
    """Test error handling for invalid role"""
    hirag = temp_hirag
    
    # Test inserting with empty role
    with pytest.raises(ValueError):
        await hirag.insert_chat_to_kb(
            chat_id="test_chat",
            role="",
            content="test content"
        )


@pytest.mark.asyncio
async def test_error_handling_empty_content(temp_hirag):
    """Test error handling for empty content"""
    hirag = temp_hirag
    
    # Test inserting with empty content
    with pytest.raises(ValueError):
        await hirag.insert_chat_to_kb(
            chat_id="test_chat",
            role="user",
            content=""
        )


@pytest.mark.asyncio
async def test_search_nonexistent_chat(temp_hirag):
    """Test searching in a non-existent chat session"""
    hirag = temp_hirag
    
    # Search in a chat that doesn't exist
    results = await hirag.search_chat_history(
        user_query="any query",
        chat_id="nonexistent_chat_id"
    )
    
    # Should return empty list
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_chat_message_content_similarity_search(temp_hirag):
    """Test that chat search actually works based on semantic similarity"""
    hirag = temp_hirag
    
    chat_id = "test_chat_007"
    
    # Insert messages with different but related content
    messages = [
        {"role": "user", "content": "I want to learn about machine learning algorithms"},
        {"role": "assistant", "content": "Machine learning algorithms are computational methods that learn patterns from data to make predictions or decisions."},
        {"role": "user", "content": "Tell me about neural networks"},
        {"role": "assistant", "content": "Neural networks are a type of ML algorithm inspired by the human brain, consisting of interconnected nodes that process information."},
        {"role": "user", "content": "What's the weather today?"},
        {"role": "assistant", "content": "I don't have access to current weather data. Please check a weather app or website."},
    ]
    
    for msg in messages:
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=msg["role"],
            content=msg["content"]
        )
    
    # Search for ML-related content
    ml_results = await hirag.search_chat_history(
        user_query="artificial intelligence and deep learning",
        chat_id=chat_id
    )
    
    # The ML-related messages should be more relevant than weather messages
    assert len(ml_results) > 0
    
    # Check that the results contain ML-related content
    ml_content_found = False
    for result in ml_results:
        content = result.get("content", "").lower()
        if any(term in content for term in ["machine learning", "neural networks", "algorithm"]):
            ml_content_found = True
            break
    
    assert ml_content_found, "Expected to find ML-related content in search results"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
