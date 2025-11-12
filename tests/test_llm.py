import asyncio
import json

import numpy as np
import pytest
import pytest_asyncio
from pydantic import BaseModel

from hirag_prod.configs.functions import get_llm_config, initialize_config_manager
from hirag_prod.resources.functions import (
    get_chat_service,
    get_embedding_service,
    get_resource_manager,
    initialize_resource_manager,
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    # Reduce noise from third-party deprecations during networked tests
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


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

    async def test_chat_completion_json_object(self):
        """Should return strictly valid JSON when response_format='json_object'"""
        if get_llm_config().service_type != "openai":
            pytest.skip("response_format requires OpenAI-compatible backend")
        chat_service = get_chat_service()
        prompt = "Hello, respond with a JSON object containing a string field 'greeting' only."
        result = await chat_service.complete(
            model="gpt-4o-mini",
            prompt=prompt,
            response_format="json_object",
        )
        # Must be a JSON string
        assert isinstance(result, str)
        obj = json.loads(result)
        assert isinstance(obj, dict)
        assert set(obj.keys()) == {"greeting"}
        assert isinstance(obj["greeting"], str)

    async def test_chat_completion_json_schema(self):
        """Should conform to supplied JSON schema with strict=true"""
        if get_llm_config().service_type != "openai":
            pytest.skip("json_schema strict requires OpenAI-compatible backend")
        chat_service = get_chat_service()
        prompt = (
            "Hello, respond with a JSON object matching the schema (greeting: string)."
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "greeting_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"greeting": {"type": "string"}},
                    "required": ["greeting"],
                    "additionalProperties": False,
                },
            },
        }
        result = await chat_service.complete(
            model="gpt-4o-mini",
            prompt=prompt,
            response_format=response_format,
        )
        # Must be a JSON string matching schema
        assert isinstance(result, str)
        obj = json.loads(result)
        assert isinstance(obj, dict)
        assert set(obj.keys()) == {"greeting"}
        assert isinstance(obj["greeting"], str)

    async def test_chat_completion_pydantic_parse(self):
        """Should parse into a Pydantic model when provided as response_format"""
        if get_llm_config().service_type != "openai":
            pytest.skip("Pydantic parse requires OpenAI-compatible backend")

        class GreetingResponse(BaseModel):
            greeting: str

        chat_service = get_chat_service()
        prompt = 'Hello, respond with JSON containing only: {"greeting": string}.'
        result = await chat_service.complete(
            model="gpt-4o-mini",
            prompt=prompt,
            response_format=GreetingResponse,
        )
        # Parsed model returned
        assert isinstance(result, GreetingResponse)
        assert isinstance(result.greeting, str)

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


if __name__ == "__main__":

    async def main():
        # Initialize config and resources (avoid using the pytest fixture here)
        initialize_config_manager(cli_options_dict={"debug": False})
        await initialize_resource_manager()

        chat_service = get_chat_service()
        try:
            # Use JSON Schema with a string typed field & date format
            json_schema = {
                "type": "object",
                "properties": {
                    # Keep the key name as-is; JSON Schema type for datetimes is string + format
                    "time-stamp": {"type": "string", "format": "date"}
                },
                "required": ["time-stamp"],
                "additionalProperties": False,
            }

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "time_extraction_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

            from datetime import datetime

            random_prompt = f"Please provide the current date: {datetime.now()}"
            print(f"Prompt: {random_prompt}")
            result = await chat_service.complete(
                model="gpt-4o-mini",
                prompt=random_prompt,
                response_format=response_format,
            )
            print(result)

            # verify result
            obj = json.loads(result)
            print(obj)
        finally:
            # Best-effort cleanup
            try:
                await chat_service.close()
            except Exception:
                pass
            try:
                await get_resource_manager().cleanup()
            except Exception:
                pass

    asyncio.run(main())
