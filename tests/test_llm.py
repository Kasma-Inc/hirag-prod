import numpy as np
import pytest
from dotenv import load_dotenv

from hirag_prod._llm import (
    ChatCompletion,
    EmbeddingService,
    SingletonABCMeta,
    SingletonMeta,
    create_chat_service,
    create_embedding_service,
)

load_dotenv("/chatbot/.env")


def clear_all_singletons():
    """Clear all singleton instances to avoid test pollution"""
    # Reset token usage stats for existing ChatCompletion instances
    for singleton_class in [ChatCompletion]:
        if (
            hasattr(singleton_class, "_instances")
            and singleton_class in singleton_class._instances
        ):
            instance = singleton_class._instances[singleton_class]
            if hasattr(instance, "reset_token_usage_stats"):
                instance.reset_token_usage_stats()

    # Clear all singleton instances
    for singleton_class in [EmbeddingService, ChatCompletion]:
        if hasattr(singleton_class, "_instances"):
            singleton_class._instances.clear()

    # Also clear the metaclass instances for both SingletonMeta and SingletonABCMeta
    SingletonMeta._instances.clear()
    SingletonABCMeta._instances.clear()


class TestConfig:
    """Test configuration and sample data"""

    SAMPLE_TEXTS = [
        "This is a test document.",
        "Another sample text for embedding.",
        "Third text to verify batch processing.",
    ]

    SAMPLE_PROMPT = "What is artificial intelligence?"
    SAMPLE_SYSTEM_PROMPT = "You are a helpful AI assistant."


@pytest.fixture(autouse=True)
def auto_clear_singletons():
    """Automatically clear singleton instances before each test"""
    clear_all_singletons()
    yield
    clear_all_singletons()


class TestChatCompletion:
    """Test suite for ChatCompletion service"""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self):
        """Test basic chat completion"""
        chat_service = create_chat_service()

        try:
            result = await chat_service.complete(
                model="gpt-4o-mini", prompt=TestConfig.SAMPLE_PROMPT
            )

            assert isinstance(result, str)
            assert len(result.strip()) > 0

        finally:
            await chat_service.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_prompt(self):
        """Test chat completion with system prompt and history"""
        chat_service = create_chat_service()

        try:
            history = [{"role": "user", "content": "Previous question"}]
            result = await chat_service.complete(
                model="gpt-4o-mini",
                prompt=TestConfig.SAMPLE_PROMPT,
                system_prompt=TestConfig.SAMPLE_SYSTEM_PROMPT,
                history_messages=history,
            )

            assert isinstance(result, str)
            assert len(result.strip()) > 0

        finally:
            await chat_service.close()


class TestEmbeddingService:
    """Test suite for embedding service"""

    @pytest.mark.asyncio
    async def test_embedding_service_basic(self):
        """Test basic embedding service"""
        embedding_service = create_embedding_service()

        try:
            result = await embedding_service.create_embeddings(
                texts=TestConfig.SAMPLE_TEXTS
            )

            assert isinstance(result, np.ndarray)
            assert result.shape[0] == len(TestConfig.SAMPLE_TEXTS)
            assert result.shape[1] > 0

        finally:
            await embedding_service.close()

    @pytest.mark.asyncio
    async def test_batch_embedding_processing(self):
        """Test batch processing functionality"""
        embedding_service = create_embedding_service(default_batch_size=2)

        try:
            # Create a larger list of texts to trigger batch processing
            large_text_list = TestConfig.SAMPLE_TEXTS * 5  # 15 texts total

            result = await embedding_service.create_embeddings(
                texts=large_text_list, batch_size=2
            )

            assert isinstance(result, np.ndarray)
            assert result.shape[0] == len(large_text_list)
            assert result.shape[1] > 0

        finally:
            await embedding_service.close()


class TestServiceFactory:
    """Test service factory functions"""

    @pytest.mark.asyncio
    async def test_chat_service_factory(self):
        """Test chat service factory"""
        service = create_chat_service()
        assert service is not None
        assert hasattr(service, "complete")
        await service.close()

    @pytest.mark.asyncio
    async def test_embedding_service_factory(self):
        """Test embedding service factory"""
        service = create_embedding_service()
        assert service is not None
        assert hasattr(service, "create_embeddings")
        await service.close()
