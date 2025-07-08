"""Quick test for LLM services"""

import asyncio
import os
import time
from unittest.mock import patch

import pytest

from hirag_prod._llm import APIConfig, ChatCompletion, EmbeddingService, TokenUsage


class TestAPIConfig:
    """Test API configuration functionality."""

    def test_api_config_from_env_success(self):
        """Test successful API config creation from environment variables."""
        with patch.dict(
            os.environ,
            {
                "TEST_API_KEY": "test-key-123",
                "TEST_BASE_URL": "https://test.example.com",
            },
        ):
            config = APIConfig.from_env("TEST_API_KEY", "TEST_BASE_URL")
            assert config.api_key == "test-key-123"
            assert config.base_url == "https://test.example.com"

    def test_api_config_missing_api_key(self):
        """Test API config fails gracefully when API key is missing."""
        with patch.dict(
            os.environ, {"TEST_BASE_URL": "https://test.example.com"}, clear=True
        ):
            with pytest.raises(
                ValueError, match="TEST_API_KEY environment variable is not set"
            ):
                APIConfig.from_env("TEST_API_KEY", "TEST_BASE_URL")

    def test_api_config_missing_base_url(self):
        """Test API config fails gracefully when base URL is missing."""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}, clear=True):
            with pytest.raises(
                ValueError, match="TEST_BASE_URL environment variable is not set"
            ):
                APIConfig.from_env("TEST_API_KEY", "TEST_BASE_URL")


class TestChatCompletion:
    """Test ChatCompletion service functionality."""

    @pytest.fixture
    def chat_service(self):
        """Create a ChatCompletion service instance for testing."""
        return ChatCompletion()

    def test_chat_service_singleton(self):
        """Test that ChatCompletion follows singleton pattern."""
        service1 = ChatCompletion()
        service2 = ChatCompletion()
        assert service1 is service2

    def test_build_messages_basic(self, chat_service):
        """Test basic message building functionality."""
        messages = chat_service._build_messages(None, None, "Test prompt")
        expected = [{"role": "user", "content": "Test prompt"}]
        assert messages == expected

    def test_build_messages_with_system_prompt(self, chat_service):
        """Test message building with system prompt."""
        messages = chat_service._build_messages(
            "You are a helpful assistant", None, "Test prompt"
        )
        expected = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Test prompt"},
        ]
        assert messages == expected

    def test_build_messages_with_history(self, chat_service):
        """Test message building with conversation history."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = chat_service._build_messages(None, history, "Test prompt")
        expected = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Test prompt"},
        ]
        assert messages == expected

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, chat_service):
        """Test basic chat completion functionality."""
        try:
            response = await chat_service.complete(
                model="gpt-4o-mini",
                prompt="What is 1+1? Reply with just the number.",
                timeout=30.0,
            )

            assert isinstance(response, str)
            assert len(response.strip()) > 0
            print(f"✅ Chat completion successful: {response.strip()}")

        except Exception as e:
            pytest.fail(f"Chat completion failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_prompt(self, chat_service):
        """Test chat completion with system prompt."""
        try:
            response = await chat_service.complete(
                model="gpt-4o-mini",
                prompt="What is the capital of France?",
                system_prompt="You are a geography expert. Give concise answers.",
                timeout=30.0,
            )

            assert isinstance(response, str)
            assert len(response.strip()) > 0
            print(
                f"✅ Chat completion with system prompt successful: {response.strip()}"
            )

        except Exception as e:
            pytest.fail(f"Chat completion with system prompt failed: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self, chat_service):
        """Test concurrent chat requests to identify potential rate limiting issues."""
        prompts = [
            "What is 6+6?",
            "What is 1+1?",
            "What is 8+8?",
        ]

        start_time = time.time()

        try:
            # Create concurrent tasks
            tasks = [
                chat_service.complete(model="gpt-4o-mini", prompt=prompt, timeout=30.0)
                for prompt in prompts
            ]

            # Execute concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            duration = end_time - start_time

            # Analyze results
            successful_responses = [r for r in responses if isinstance(r, str)]
            failed_responses = [r for r in responses if isinstance(r, Exception)]

            print(f"✅ Concurrent requests completed in {duration:.2f}s")
            print(f"   Successful: {len(successful_responses)}/{len(prompts)}")
            print(f"   Failed: {len(failed_responses)}")

            if failed_responses:
                for i, error in enumerate(failed_responses):
                    print(f"   Error {i+1}: {type(error).__name__}: {error}")

            # At least some requests should succeed
            assert len(successful_responses) > 0, "All concurrent requests failed"

        except Exception as e:
            pytest.fail(f"Concurrent chat requests failed: {e}")


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing."""
        return EmbeddingService(default_batch_size=10)

    def test_embedding_service_singleton(self):
        """Test that EmbeddingService follows singleton pattern."""
        service1 = EmbeddingService()
        service2 = EmbeddingService()
        assert service1 is service2

    @pytest.mark.asyncio
    async def test_single_embedding(self, embedding_service):
        """Test single text embedding."""
        try:
            embeddings = await embedding_service.create_embeddings(
                texts=["This is a test sentence."]
            )

            assert embeddings.shape[0] == 1
            assert embeddings.shape[1] > 0  # Should have embedding dimensions
            print(f"✅ Single embedding successful, shape: {embeddings.shape}")

        except Exception as e:
            pytest.fail(f"Single embedding failed: {e}")

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, embedding_service):
        """Test batch text embeddings."""
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence.",
        ]

        try:
            embeddings = await embedding_service.create_embeddings(texts=texts)

            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0
            print(f"✅ Batch embeddings successful, shape: {embeddings.shape}")

        except Exception as e:
            pytest.fail(f"Batch embeddings failed: {e}")

    @pytest.mark.asyncio
    async def test_large_batch_embeddings(self, embedding_service):
        """Test large batch processing with adaptive batching."""
        # Create more texts than the batch size to test batching logic
        texts = [f"Test sentence number {i}." for i in range(25)]

        try:
            start_time = time.time()
            embeddings = await embedding_service.create_embeddings(
                texts=texts, batch_size=5  # Force multiple batches
            )
            end_time = time.time()

            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0

            duration = end_time - start_time
            print(
                f"✅ Large batch embeddings successful in {duration:.2f}s, shape: {embeddings.shape}"
            )

        except Exception as e:
            pytest.fail(f"Large batch embeddings failed: {e}")

    def test_embedding_text_validation(self, embedding_service):
        """Test text validation in embedding service."""
        # Test empty list
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding_service._text_validator.validate_and_clean([])

        # Test None text
        with pytest.raises(ValueError, match="Text at index 0 is None"):
            embedding_service._text_validator.validate_and_clean([None])

        # Test non-string text
        with pytest.raises(ValueError, match="Text at index 0 is not a string"):
            embedding_service._text_validator.validate_and_clean([123])

        # Test empty string after strip
        with pytest.raises(
            ValueError, match="Text at index 0 is empty after stripping"
        ):
            embedding_service._text_validator.validate_and_clean(["   "])


class TestTokenUsage:
    """Test token usage tracking functionality."""

    def test_token_usage_creation(self):
        """Test TokenUsage object creation and string representation."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

        expected_str = "Tokens - Prompt: 10, Completion: 5, Total: 15"
        assert str(usage) == expected_str


class TestEnvironmentConfiguration:
    """Test environment configuration for LLM services."""

    def test_required_environment_variables(self):
        """Test that required environment variables are set."""
        required_vars = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_EMBEDDING_API_KEY",
            "OPENAI_EMBEDDING_BASE_URL",
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")

        print("✅ All required environment variables are set")

    def test_api_endpoints_format(self):
        """Test that API endpoints are properly formatted URLs."""
        base_urls = [
            ("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL")),
            ("OPENAI_EMBEDDING_BASE_URL", os.getenv("OPENAI_EMBEDDING_BASE_URL")),
        ]

        for var_name, url in base_urls:
            if url:
                assert url.startswith(
                    "http"
                ), f"{var_name} should start with http/https"
                assert not url.endswith(
                    "/"
                ), f"{var_name} should not end with trailing slash"
                print(f"✅ {var_name} is properly formatted: {url}")


@pytest.mark.asyncio
async def test_integration_chat_and_embedding():
    """Integration test combining chat completion and embedding services."""
    try:
        # Test chat completion
        chat_service = ChatCompletion()
        chat_response = await chat_service.complete(
            model="gpt-4o-mini",
            prompt="Generate a short sentence about AI.",
            timeout=30.0,
        )

        # Test embedding of the generated text
        embedding_service = EmbeddingService()
        embeddings = await embedding_service.create_embeddings([chat_response])

        assert isinstance(chat_response, str)
        assert len(chat_response.strip()) > 0
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

        print(f"✅ Integration test successful")
        print(f"   Generated text: {chat_response}")
        print(f"   Embedding shape: {embeddings.shape}")

    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    """Run tests directly for debugging purposes."""
    import sys

    print("🧪 Running LLM tests directly...")

    # Basic configuration test
    print("\n1. Testing environment configuration...")
    test_env = TestEnvironmentConfiguration()
    test_env.test_required_environment_variables()
    test_env.test_api_endpoints_format()

    # Run async tests
    async def run_async_tests():
        print("\n2. Testing chat completion...")
        chat_test = TestChatCompletion()
        chat_service = ChatCompletion()  # Create service directly
        await chat_test.test_chat_completion_basic(chat_service)

        print("\n3. Testing embeddings...")
        embed_test = TestEmbeddingService()
        embed_service = EmbeddingService(
            default_batch_size=10
        )  # Create service directly
        await embed_test.test_single_embedding(embed_service)

        print("\n4. Running integration test...")
        await test_integration_chat_and_embedding()

        print("\n✅ All direct tests completed successfully!")

    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
