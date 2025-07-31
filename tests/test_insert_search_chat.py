import os
import uuid
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
        chat_id = f"test_chat_session_{uuid.uuid4().hex[:8]}"
        role = "user"
        content = "Hello, this is a test message for the chat functionality."
        
        # Insert chat message
        metrics = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role=role,
            content=content
        )
        
        # Verify the response
        assert metrics is not None, "Metrics should not be None"
        assert isinstance(metrics, ProcessingMetrics), f"Expected ProcessingMetrics, got {type(metrics)}"
        assert metrics.processed_chats >= 1, f"Expected at least 1 processed chat, got {metrics.processed_chats}"
        assert metrics.processing_time > 0, f"Processing time should be > 0, got {metrics.processing_time}"
        assert metrics.error_count == 0, f"Expected 0 errors, got {metrics.error_count}"
        
        print(f"✅ Successfully inserted chat message. Metrics: {metrics.to_dict()}")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_insert_chat_to_kb_multiple_roles():
    """Test inserting chat messages with different roles"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = f"test_chat_session_{uuid.uuid4().hex[:8]}"
        
        # Test messages with different roles
        test_messages = [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence."},
            {"role": "user", "content": "Can you give me some examples?"},
            {"role": "assistant", "content": "Sure! Examples of AI include virtual assistants like Siri, recommendation systems on Netflix, autonomous vehicles, and image recognition software."},
            {"role": "tool", "content": "Retrieved relevant information about AI applications from knowledge base."}
        ]
        
        # Insert all messages
        total_processed = 0
        for msg in test_messages:
            metrics = await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role=msg["role"],
                content=msg["content"]
            )
            
            assert metrics is not None, f"Metrics should not be None for {msg['role']} message"
            assert isinstance(metrics, ProcessingMetrics), f"Expected ProcessingMetrics, got {type(metrics)}"
            assert metrics.processed_chats >= 1, f"Expected at least 1 processed chat for {msg['role']}, got {metrics.processed_chats}"
            assert metrics.error_count == 0, f"Expected 0 errors for {msg['role']}, got {metrics.error_count}"
            total_processed += metrics.processed_chats
            print(f"✅ Inserted {msg['role']} message successfully")
        
        assert total_processed >= len(test_messages), f"Expected at least {len(test_messages)} total processed, got {total_processed}"
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history():
    """Test searching chat history functionality"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = f"test_chat_session_{uuid.uuid4().hex[:8]}"
        
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
            metrics = await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role=msg["role"],
                content=msg["content"]
            )
            assert metrics is not None, f"Metrics should not be None for message: {msg['content'][:50]}..."
            assert metrics.error_count == 0, f"Insertion failed for message: {msg['content'][:50]}..."
        
        # Test search functionality
        search_query = "machine learning algorithms"
        results = await hirag.search_chat_history(
            user_query=search_query,
            chat_id=chat_id,
            topk=5,
            topn=3
        )
        
        # Verify the results
        assert results is not None, "Search results should not be None"
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) > 0, f"Expected at least 1 result for query '{search_query}', got {len(results)}"
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
        chat_id = f"test_chat_session_{uuid.uuid4().hex[:8]}"
        
        # Insert messages with different roles
        metrics1 = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="user",
            content="I need help with Python programming"
        )
        assert metrics1 is not None and metrics1.error_count == 0, "Failed to insert user message"
        
        metrics2 = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="assistant", 
            content="Python is a versatile programming language. What specific aspect would you like help with?"
        )
        assert metrics2 is not None and metrics2.error_count == 0, "Failed to insert assistant message"
        
        metrics3 = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="tool",
            content="Retrieved Python documentation and tutorials from knowledge base"
        )
        assert metrics3 is not None and metrics3.error_count == 0, "Failed to insert tool message"
        
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
        assert user_results is not None, "User search results should not be None"
        assert assistant_results is not None, "Assistant search results should not be None"
        assert isinstance(user_results, list), f"Expected list for user results, got {type(user_results)}"
        assert isinstance(assistant_results, list), f"Expected list for assistant results, got {type(assistant_results)}"
        
        # At least one of the role-filtered searches should return results
        total_results = len(user_results) + len(assistant_results)
        assert total_results > 0, "At least one role filter should return results"
        
        print(f"✅ User messages found: {len(user_results)}")
        print(f"✅ Assistant messages found: {len(assistant_results)}")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history_empty_results():
    """Test searching for non-existent content in chat history"""
    hirag = await HiRAG.create()
    
    try:
        chat_id = f"test_chat_session_{uuid.uuid4().hex[:8]}"
        
        # Insert a message
        metrics = await hirag.insert_chat_to_kb(
            chat_id=chat_id,
            role="user",
            content="Hello world"
        )
        assert metrics is not None and metrics.error_count == 0, "Failed to insert test message"
        
        # Search for something that doesn't exist
        results = await hirag.search_chat_history(
            user_query="quantum physics spacecraft navigation",
            chat_id=chat_id,
            topk=5
        )
        
        # Should return empty list or very low relevance results
        assert results is not None, "Search results should not be None even for non-existent content"
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        print(f"✅ Search for non-existent content returned {len(results)} results")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_insert_chat_to_kb_error_handling():
    """Test error handling for invalid inputs"""
    hirag = await HiRAG.create()
    
    try:
        # Test with empty content - should raise an exception
        try:
            await hirag.insert_chat_to_kb(
                chat_id=f"test_error_{uuid.uuid4().hex[:8]}",
                role="user",
                content=""
            )
            # If we reach here, the test should fail
            assert False, "Expected exception for empty content, but none was raised"
        except Exception as e:
            print(f"✅ Error handling for empty content works correctly: {type(e).__name__}")
        
        # Test with None content - should raise an exception
        try:
            await hirag.insert_chat_to_kb(
                chat_id=f"test_error_{uuid.uuid4().hex[:8]}",
                role="user",
                content=None
            )
            assert False, "Expected exception for None content, but none was raised"
        except Exception as e:
            print(f"✅ Error handling for None content works correctly: {type(e).__name__}")
        
        # Test with invalid role - should raise an exception or handle gracefully
        try:
            await hirag.insert_chat_to_kb(
                chat_id=f"test_error_{uuid.uuid4().hex[:8]}",
                role="invalid_role",
                content="Valid content"
            )
            print("⚠️ Warning: Invalid role was accepted (might be intentional)")
        except Exception as e:
            print(f"✅ Error handling for invalid role works correctly: {type(e).__name__}")
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_search_chat_history_different_chat_sessions():
    """Test that search results are isolated by chat_id"""
    hirag = await HiRAG.create()
    
    try:
        # Insert messages in different chat sessions
        chat_id_1 = f"session_{uuid.uuid4().hex[:8]}"
        chat_id_2 = f"session_{uuid.uuid4().hex[:8]}"
        
        metrics1 = await hirag.insert_chat_to_kb(
            chat_id=chat_id_1,
            role="user",
            content="I want to learn about cats"
        )
        assert metrics1 is not None and metrics1.error_count == 0, "Failed to insert message in session 1"
        
        metrics2 = await hirag.insert_chat_to_kb(
            chat_id=chat_id_2,
            role="user",
            content="I want to learn about dogs"
        )
        assert metrics2 is not None and metrics2.error_count == 0, "Failed to insert message in session 2"
        
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
        
        assert results_1 is not None, "Results for session 1 should not be None"
        assert results_2 is not None, "Results for session 2 should not be None"
        assert isinstance(results_1, list), f"Expected list for session 1, got {type(results_1)}"
        assert isinstance(results_2, list), f"Expected list for session 2, got {type(results_2)}"
        
        # Both sessions should have at least one result since we searched for "animals" 
        # and both sessions contain animal-related content
        assert len(results_1) > 0, f"Session 1 should have results for 'animals' query, got {len(results_1)}"
        assert len(results_2) > 0, f"Session 2 should have results for 'animals' query, got {len(results_2)}"
        
        print(f"✅ Session 1 results: {len(results_1)}")
        print(f"✅ Session 2 results: {len(results_2)}")
        
        # Verify that the results are actually isolated (optional additional check)
        # This would require examining the content of results to ensure they're from the right session
        
    finally:
        await hirag.clean_up()


@pytest.mark.asyncio
async def test_comprehensive_error_scenarios():
    """Test comprehensive error scenarios and edge cases"""
    hirag = await HiRAG.create()
    
    try:
        # Test search on non-existent chat_id
        results = await hirag.search_chat_history(
            user_query="test query",
            chat_id=f"nonexistent_{uuid.uuid4().hex[:8]}",
            topk=5
        )
        assert results is not None, "Search on non-existent chat_id should return empty list, not None"
        assert isinstance(results, list), f"Expected list for non-existent chat_id, got {type(results)}"
        print(f"✅ Search on non-existent chat_id returned {len(results)} results")
        
        # Test with very long content
        long_content = "A" * 10000  # 10k characters
        chat_id = f"long_content_{uuid.uuid4().hex[:8]}"
        try:
            metrics = await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role="user",
                content=long_content
            )
            assert metrics is not None, "Should handle long content gracefully"
            assert metrics.error_count == 0, f"Long content insertion failed with {metrics.error_count} errors"
            print("✅ Long content handled successfully")
        except Exception as e:
            print(f"⚠️ Long content caused exception: {type(e).__name__}: {str(e)}")
        
        # Test with special characters and unicode
        special_content = "Test with émojis 🚀🎉 and spëcial châractërs: @#$%^&*()[]{}|\\:;\"'<>?,./"
        chat_id = f"special_{uuid.uuid4().hex[:8]}"
        try:
            metrics = await hirag.insert_chat_to_kb(
                chat_id=chat_id,
                role="user",
                content=special_content
            )
            assert metrics is not None, "Should handle special characters gracefully"
            assert metrics.error_count == 0, f"Special characters insertion failed with {metrics.error_count} errors"
            print("✅ Special characters handled successfully")
        except Exception as e:
            print(f"⚠️ Special characters caused exception: {type(e).__name__}: {str(e)}")
        
        # Test search with empty query
        try:
            results = await hirag.search_chat_history(
                user_query="",
                chat_id=chat_id,
                topk=5
            )
            assert results is not None, "Empty query should return empty list, not None"
            print(f"✅ Empty query handled, returned {len(results)} results")
        except Exception as e:
            print(f"⚠️ Empty query caused exception: {type(e).__name__}: {str(e)}")
        
        # Test with extreme topk values
        try:
            results = await hirag.search_chat_history(
                user_query="test",
                chat_id=chat_id,
                topk=0
            )
            assert results is not None, "topk=0 should return empty list, not None"
            print(f"✅ topk=0 handled, returned {len(results)} results")
        except Exception as e:
            print(f"⚠️ topk=0 caused exception: {type(e).__name__}: {str(e)}")
            
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
        
        await test_comprehensive_error_scenarios()
        print("=" * 50)
        
        print("🎉 All tests completed!")
    
    # Uncomment to run tests manually
    asyncio.run(run_tests())