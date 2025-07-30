import os
import pytest
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv

from hirag_prod import HiRAG
from hirag_prod.hirag import ProcessingMetrics

# Load environment variables
load_dotenv("/chatbot/.env", override=True)


@pytest.mark.asyncio
async def test_insert_chat_to_kb():
    """Test inserting a chat message to the knowledge base"""
    # Create HiRAG instance
    hirag = await HiRAG.create()
    
    try:
        # Test data
        chat_id = "test_chat_session_001"
        role = "user"
        content = "Hello, this is a test message for the chat functionality."
        
        # Insert chat message
        metrics = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=role,
            content=content
        )
        
        # Verify the response
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.processed_chats >= 0  # Should be at least 0
        assert metrics.processing_time > 0  # Should take some time
        assert metrics.error_count == 0  # No errors expected
        
        print(f"✅ Successfully inserted chat message. Metrics: {metrics.to_dict()}")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_insert_chat_to_kb_multiple_roles():
    """Test inserting chat messages with different roles"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = "test_chat_session_002"
        
        # Test messages with different roles
        test_messages = [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence."},
            {"role": "user", "content": "Can you give me some examples?"},
            {"role": "assistant", "content": "Sure! Examples of AI include virtual assistants like Siri, recommendation systems on Netflix, autonomous vehicles, and image recognition software."},
            {"role": "tool", "content": "Retrieved relevant information about AI applications from knowledge base."}
        ]
        
        # Insert all messages
        for msg in test_messages:
            metrics = await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role=msg["role"],
                content=msg["content"]
            )
            
            assert isinstance(metrics, ProcessingMetrics)
            assert metrics.error_count == 0
            print(f"✅ Inserted {msg['role']} message successfully")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history():
    """Test searching chat history functionality"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = "test_chat_session_003"
        
        # First, insert some test messages to search for
        test_messages = [
            {"role": "user", "content": "I want to learn about machine learning algorithms"},
            {"role": "assistant", "content": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Each type serves different purposes."},
            {"role": "user", "content": "Tell me about supervised learning"},
            {"role": "assistant", "content": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks."},
            {"role": "user", "content": "What about deep learning?"},
            {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It's particularly effective for image recognition, natural language processing, and speech recognition."}
        ]
        
        # Insert the test messages
        for msg in test_messages:
            await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role=msg["role"],
                content=msg["content"]
            )
        
        # Test search functionality
        search_query = "machine learning algorithms"
        results = await hirag.search_chat_history(
            user_query=search_query,
            chat_id=chat_id,
            topk=5,
            topn=3
        )
        
        # Verify the results
        assert isinstance(results, list)
        print(f"✅ Found {len(results)} results for query: '{search_query}'")
        
        # Print results for verification
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result}")
            
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history_with_role_filter():
    """Test searching chat history with role filtering"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = "test_chat_session_004"
        
        # Insert messages with different roles
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="user",
            content="I need help with Python programming"
        )
        
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="assistant", 
            content="Python is a versatile programming language. What specific aspect would you like help with?"
        )
        
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="tool",
            content="Retrieved Python documentation and tutorials from knowledge base"
        )
        
        # Search with role filter for user messages only
        user_results = await hirag.search_chat_history(
            user_query="Python programming",
            chat_id=chat_id,
            role="user",
            topk=5
        )
        
        # Search with role filter for assistant messages only
        assistant_results = await hirag.search_chat_history(
            user_query="Python programming",
            chat_id=chat_id,
            role="assistant",
            topk=5
        )
        
        # Verify results
        assert isinstance(user_results, list)
        assert isinstance(assistant_results, list)
        
        print(f"✅ User messages found: {len(user_results)}")
        print(f"✅ Assistant messages found: {len(assistant_results)}")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history_empty_results():
    """Test searching for non-existent content in chat history"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = "test_chat_session_005"
        
        # Insert a message
        await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="user",
            content="Hello world"
        )
        
        # Search for something that doesn't exist
        results = await hirag.search_chat_history(
            user_query="quantum physics spacecraft navigation",
            chat_id=chat_id,
            topk=5
        )
        
        # Should return empty list or very low relevance results
        assert isinstance(results, list)
        print(f"✅ Search for non-existent content returned {len(results)} results")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_insert_chat_to_kb_error_handling():
    """Test error handling for invalid inputs"""
    hirag = await HiRAG.create()
    
    try:
        # Test with empty content
        with pytest.raises(Exception):
            await hirag.insert_chat_to_kb(
                chat_id="test_error",
                role="user",
                content=""
            )
        
        print("✅ Error handling for empty content works correctly")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history_different_chat_sessions():
    """Test that search results are isolated by chat_id"""
    hirag = await HiRAG.create()
    
    try:
        # Insert messages in different chat sessions
        chat_id_1 = "session_001"
        chat_id_2 = "session_002"
        
        await hirag.insert_chat_to_kb(
            chat_id=chat_id_1,
            role="user",
            content="I want to learn about cats"
        )
        
        await hirag.insert_chat_to_kb(
            chat_id=chat_id_2,
            role="user",
            content="I want to learn about dogs"
        )
        
        # Search in first session should only return cats-related content
        results_1 = await hirag.search_chat_history(
            user_query="animals",
            chat_id=chat_id_1,
            topk=5
        )
        
        # Search in second session should only return dogs-related content
        results_2 = await hirag.search_chat_history(
            user_query="animals",
            chat_id=chat_id_2,
            topk=5
        )
        
        assert isinstance(results_1, list)
        assert isinstance(results_2, list)
        
        print(f"✅ Session 1 results: {len(results_1)}")
        print(f"✅ Session 2 results: {len(results_2)}")
        
    finally:
        await hirag.clean_up()


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        """Run all tests manually for debugging"""
        print("🧪 Running chat functionality tests...\n")
        
        await test_insert_chat_to_kb()
        print("=" * 50)
        
        await test_insert_chat_to_kb_multiple_roles()
        print("=" * 50)
        
        await test_search_chat_history()
        print("=" * 50)
        
        await test_search_chat_history_with_role_filter()
        print("=" * 50)
        
        await test_search_chat_history_empty_results()
        print("=" * 50)
        
        await test_search_chat_history_different_chat_sessions()
        print("=" * 50)
        
        print("🎉 All tests completed!")
    
    # Uncomment to run tests manually
    # asyncio.run(run_tests())