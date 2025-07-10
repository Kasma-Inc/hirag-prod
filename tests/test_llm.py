import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from dotenv import load_dotenv

from src.hirag_prod._llm import (
    ChatCompletion,
    EmbeddingService,
    EmbeddingServiceType,
    LocalEmbeddingService,
    create_embedding_service,
)

load_dotenv("/chatbot/.env", override=True)


def clear_all_singletons():
    """Clear all singleton instances to avoid test pollution"""
    for singleton_class in [EmbeddingService, ChatCompletion]:
        if hasattr(singleton_class, "_instances"):
            singleton_class._instances.clear()


class TestConfig:
    """Test configuration and sample data"""

    SAMPLE_TEXTS = [
        "This is a test document.",
        "Another sample text for embedding.",
        "Third text to verify batch processing.",
    ]

    SAMPLE_PROMPT = "What is artificial intelligence?"
    SAMPLE_MODEL = "gpt-3.5-turbo"
    SAMPLE_EMBEDDING_MODEL = "text-embedding-3-small"


def get_required_env(key: str) -> str:
    """Get required environment variable or raise error"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_embedding_service_type() -> str:
    """Get embedding service type from environment"""
    return os.getenv("EMBEDDING_SERVICE_TYPE", "openai").lower()


def get_embedding_dimension() -> int:
    """Get embedding dimension from environment"""
    dimension_str = os.getenv("EMBEDDING_DIMENSION")
    if not dimension_str:
        raise ValueError("Required environment variable EMBEDDING_DIMENSION is not set")

    try:
        dimension = int(dimension_str)
    except ValueError:
        raise ValueError("EMBEDDING_DIMENSION must be an integer")

    return dimension


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for chat completion"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "AI is a field of computer science."
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 15
    mock_response.usage.total_tokens = 25
    return mock_response


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response for OpenAI"""
    dimension = get_embedding_dimension()
    embeddings = [[0.1] * dimension, [0.2] * dimension, [0.3] * dimension]

    mock_response = MagicMock()
    mock_response.data = []
    for embedding in embeddings:
        mock_data = MagicMock()
        mock_data.embedding = embedding
        mock_response.data.append(mock_data)

    return mock_response


@pytest.fixture
def mock_local_embedding_response():
    """Mock local embedding service response"""
    dimension = get_embedding_dimension()
    return {
        "data": [
            {"embedding": [0.1] * dimension},
            {"embedding": [0.2] * dimension},
            {"embedding": [0.3] * dimension},
        ]
    }


class TestChatCompletion:
    """Test suite for ChatCompletion service"""

    @pytest.mark.asyncio
    async def test_chat_completion_connectivity(self, mock_openai_response):
        """Test basic chat completion connectivity using environment configuration"""
        # Verify required environment variables
        api_key = get_required_env("LLM_API_KEY")
        base_url = get_required_env("LLM_BASE_URL")

        chat_service = ChatCompletion()

        with patch.object(
            chat_service.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ) as mock_create:

            result = await chat_service.complete(
                model=TestConfig.SAMPLE_MODEL, prompt=TestConfig.SAMPLE_PROMPT
            )

            assert result == "AI is a field of computer science."
            mock_create.assert_called_once()

            # Verify token usage tracking
            stats = chat_service.get_token_usage_stats()
            assert stats["request_count"] == 1
            assert stats["total_tokens"] == 25

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_prompt(self, mock_openai_response):
        """Test chat completion with system prompt and history"""
        chat_service = ChatCompletion()

        with patch.object(
            chat_service.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ) as mock_create:

            history = [{"role": "user", "content": "Previous question"}]
            result = await chat_service.complete(
                model=TestConfig.SAMPLE_MODEL,
                prompt=TestConfig.SAMPLE_PROMPT,
                system_prompt="You are a helpful AI assistant.",
                history_messages=history,
            )

            assert result == "AI is a field of computer science."

            # Verify messages structure
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 3  # system + history + user
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "user"


class TestEmbeddingServiceOpenAI:
    """Test suite for OpenAI embedding service"""

    @pytest.mark.skipif(
        get_embedding_service_type() != "openai",
        reason="Skipping OpenAI tests - service type is not 'openai'",
    )
    @pytest.mark.asyncio
    async def test_openai_embedding_service(self, mock_embedding_response):
        """Test OpenAI embedding service connectivity using environment configuration"""
        # Verify required environment variables
        api_key = get_required_env("EMBEDDING_API_KEY")
        base_url = get_required_env("EMBEDDING_BASE_URL")

        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        # Patch rate limiting to disable it during tests
        with patch("src.hirag_prod._llm.rate_limited", return_value=lambda func: func):
            # Use the factory function like hirag.py does
            embedding_service = create_embedding_service()

            with patch.object(
                embedding_service.client.embeddings,
                "create",
                new_callable=AsyncMock,
                return_value=mock_embedding_response,
            ) as mock_create:

                result = await embedding_service.create_embeddings(
                    texts=TestConfig.SAMPLE_TEXTS,
                    model=TestConfig.SAMPLE_EMBEDDING_MODEL,
                )

                assert isinstance(result, np.ndarray)
                assert result.shape == (3, get_embedding_dimension())
                mock_create.assert_called_once()

                await embedding_service.close()

    @pytest.mark.skipif(
        get_embedding_service_type() != "openai",
        reason="Skipping OpenAI tests - service type is not 'openai'",
    )
    @pytest.mark.asyncio
    async def test_batch_embedding_processing(self, mock_embedding_response):
        """Test batch processing functionality with large text lists for OpenAI service"""
        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        # Patch rate limiting to disable it during tests
        with patch("src.hirag_prod._llm.rate_limited", return_value=lambda func: func):
            # Use the factory function like hirag.py does, with custom batch size
            embedding_service = create_embedding_service(default_batch_size=2)

            # Create a larger list of texts to trigger batch processing
            large_text_list = TestConfig.SAMPLE_TEXTS * 5  # 15 texts total

            with patch.object(
                embedding_service.client.embeddings,
                "create",
                new_callable=AsyncMock,
                return_value=mock_embedding_response,
            ) as mock_create:

                result = await embedding_service.create_embeddings(
                    texts=large_text_list, batch_size=2
                )

                assert isinstance(result, np.ndarray)
                # Should process in multiple batches (15 texts, batch_size=2 → 8 batches)
                assert mock_create.call_count >= 7

                await embedding_service.close()


class TestEmbeddingServiceLocal:
    """Test suite for local embedding service"""

    @pytest.mark.skipif(
        get_embedding_service_type() != "local",
        reason="Skipping local tests - service type is not 'local'",
    )
    @pytest.mark.asyncio
    async def test_local_embedding_service(self, mock_local_embedding_response):
        """Test local embedding service connectivity using environment configuration"""
        # Verify required environment variables
        base_url = get_required_env("LOCAL_EMBEDDING_BASE_URL")
        model_name = get_required_env("LOCAL_EMBEDDING_MODEL_NAME")
        auth_token = get_required_env("LOCAL_EMBEDDING_AUTH_TOKEN")
        model_path = get_required_env("LOCAL_EMBEDDING_MODEL_PATH")

        # Use the factory function like hirag.py does
        embedding_service = create_embedding_service()

        # Mock the HTTP client for local service
        with patch.object(
            embedding_service.client._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_local_embedding_response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = await embedding_service.create_embeddings(
                texts=TestConfig.SAMPLE_TEXTS
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (3, get_embedding_dimension())
            mock_post.assert_called_once()

            await embedding_service.close()

    @pytest.mark.skipif(
        get_embedding_service_type() != "local",
        reason="Skipping local tests - service type is not 'local'",
    )
    @pytest.mark.asyncio
    async def test_local_embedding_batch_processing(
        self, mock_local_embedding_response
    ):
        """Test batch processing functionality for local embedding service"""
        # Use the factory function like hirag.py does, with custom batch size
        embedding_service = create_embedding_service(default_batch_size=2)

        # Create a larger list of texts to trigger batch processing
        large_text_list = TestConfig.SAMPLE_TEXTS * 5  # 15 texts total

        # Mock the HTTP client
        with patch.object(
            embedding_service.client._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_local_embedding_response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = await embedding_service.create_embeddings(
                texts=large_text_list, batch_size=2
            )

            assert isinstance(result, np.ndarray)
            # Should process in multiple batches (15 texts, batch_size=2 → 8 batches)
            assert mock_post.call_count >= 7

            await embedding_service.close()


class TestEmbeddingServiceCommon:
    """Common tests for both embedding service types"""

    @pytest.mark.asyncio
    async def test_embedding_service_factory(self):
        """Test the create_embedding_service factory function with current environment"""
        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        service_type = get_embedding_service_type()

        service = create_embedding_service()

        if service_type == "local":
            assert isinstance(service, LocalEmbeddingService)
        else:
            assert isinstance(service, EmbeddingService)

        await service.close()

    @pytest.mark.asyncio
    async def test_text_validation(self):
        """Test text validation and error handling"""
        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        # Use the factory function like hirag.py does
        embedding_service = create_embedding_service()

        # Test empty list
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            await embedding_service.create_embeddings([])

        # Test None values
        with pytest.raises(ValueError, match="Text at index .* is None"):
            await embedding_service.create_embeddings(["valid text", None])

        # Test non-string values
        with pytest.raises(ValueError, match="Text at index .* is not a string"):
            await embedding_service.create_embeddings(["valid text", 123])

        # Test empty strings
        with pytest.raises(ValueError, match="Text at index .* is empty"):
            await embedding_service.create_embeddings(["valid text", "   "])

        await embedding_service.close()


class TestIntegration:
    """Integration tests for the complete LLM module"""

    @pytest.mark.asyncio
    async def test_service_environment_detection(self):
        """Test that services correctly detect and use environment configuration"""
        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        service_type = get_embedding_service_type()

        service = create_embedding_service()

        if service_type == "local":
            assert isinstance(service, LocalEmbeddingService)
        else:
            assert isinstance(service, EmbeddingService)
            # Only check service type for EmbeddingService that uses UnifiedEmbeddingClient
            if hasattr(service.client, "config"):
                assert service.client.config.service_type == EmbeddingServiceType.OPENAI

        await service.close()

    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test proper cleanup of services"""
        # Clear all singleton instances to avoid conflicts
        clear_all_singletons()

        service = create_embedding_service()

        # Ensure close method exists and can be called
        assert hasattr(service, "close")
        await service.close()

        # Verify HTTP client is properly closed for local service
        if isinstance(service, LocalEmbeddingService):
            assert service.client._http_client.is_closed
