import os
from typing import Any, Dict, List, Optional

import numpy as np
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class OpenAIConfig:
    """Configuration for OpenAI API"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self._validate()

    def _validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL environment variable is not set")


class OpenAIClient:
    """Singleton wrapper for OpenAI async client"""

    _instance: Optional["OpenAIClient"] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            config = OpenAIConfig()
            self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client


# Retry decorator for API calls
api_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)


class ChatCompletion:
    """Handler for OpenAI chat completions"""

    def __init__(self):
        self.client = OpenAIClient().client

    @api_retry
    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Complete a chat prompt using the specified model.

        Args:
            model: The model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt: The user prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            **kwargs: Additional parameters for the API call

        Returns:
            The completion response as a string
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        return response.choices[0].message.content


class EmbeddingService:
    """Handler for OpenAI embeddings"""

    def __init__(self):
        self.client = OpenAIClient().client

    @api_retry
    async def create_embeddings(
        self, texts: List[str], model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Create embeddings for the given texts.

        Args:
            texts: List of texts to embed
            model: The embedding model to use

        Returns:
            Numpy array of embeddings
        """
        response = await self.client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])
