import numpy as np
import pytest
import pytest_asyncio

from hirag_prod.configs.functions import initialize_config_manager
from hirag_prod.resources.functions import (
    get_chat_service,
    get_embedding_service,
    get_resource_manager,
    initialize_resource_manager,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")


class TestConfig:
    """Test configuration and sample data"""

    SAMPLE_TEXTS = [
        "This is a test document.",
        "Another sample text for embedding.",
        "Third text to verify batch processing.",
    ]

    SAMPLE_PROMPT = "What is artificial intelligence?"
    SAMPLE_SYSTEM_PROMPT = "You are a helpful AI assistant."


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_env():
    """Initialize config/resources once per module, and cleanup once."""
    initialize_config_manager(cli_options_dict={"debug": False})
    await initialize_resource_manager()
    chat_service = get_chat_service()
    embedding_service = get_embedding_service()
    yield
    try:
        await chat_service.close()
    except Exception:
        pass
    try:
        await embedding_service.close()
    except Exception:
        pass
    try:
        await get_resource_manager().cleanup()
    except Exception:
        pass


class TestChatCompletion:
    """Test suite for ChatCompletion service"""

    async def test_chat_completion_basic(self):
        """Test basic chat completion"""
        chat_service = get_chat_service()
        result = await chat_service.complete(
            model="gpt-4o-mini", prompt=TestConfig.SAMPLE_PROMPT
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    async def test_chat_completion_with_system_prompt(self):
        """Test chat completion with system prompt and history"""
        chat_service = get_chat_service()
        history = [{"role": "user", "content": "Previous question"}]
        result = await chat_service.complete(
            model="gpt-4o-mini",
            prompt=TestConfig.SAMPLE_PROMPT,
            system_prompt=TestConfig.SAMPLE_SYSTEM_PROMPT,
            history_messages=history,
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0


class TestEmbeddingService:
    """Test suite for embedding service"""

    async def test_embedding_service_basic(self):
        """Test basic embedding service"""
        embedding_service = get_embedding_service()
        result = await embedding_service.create_embeddings(
            texts=TestConfig.SAMPLE_TEXTS
        )
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(TestConfig.SAMPLE_TEXTS)
        assert result.shape[1] > 0

    async def test_batch_embedding_processing(self):
        """Test batch processing functionality"""
        embedding_service = get_embedding_service()
        large_text_list = TestConfig.SAMPLE_TEXTS * 5  # 15 texts total
        result = await embedding_service.create_embeddings(texts=large_text_list)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(large_text_list)
        assert result.shape[1] > 0

    async def test_embedding_with_empty_inputs(self):
        """Embedding should handle empty/None/whitespace by returning zeros in place."""
        embedding_service = get_embedding_service()
        texts = ["Hello", "", "   ", None, "\n", "World"]
        result = await embedding_service.create_embeddings(texts=texts)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(texts)

        empty_indices = [1, 2, 3, 4]
        for idx in empty_indices:
            assert np.allclose(result[idx], 0.0)

        for idx in [0, 5]:
            assert not np.allclose(result[idx], 0.0)


class TestServiceFactory:
    """Test service factory functions"""

    async def test_chat_service_factory(self):
        """Test chat service factory"""
        service = get_chat_service()
        assert service is not None
        assert hasattr(service, "complete")

    async def test_embedding_service_factory(self):
        """Test embedding service factory"""
        service = get_embedding_service()
        assert service is not None
        assert hasattr(service, "create_embeddings")
