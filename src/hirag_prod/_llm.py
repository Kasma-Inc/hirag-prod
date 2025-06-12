import os
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
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
        self,
        texts: List[str] | str,
        model: str = "text-embedding-3-small",
        *,
        max_tokens: int = 100,  # max tokens per chunk; controls granularity of text splitting
        batch_size: int = 1,  # number of chunks per API call; controls throughput vs. safety
    ) -> np.ndarray:
        """
        Create embeddings for the given texts.

        This helper takes care of splitting long texts into smaller chunks,
        batching requests to the OpenAI API and retrying with smaller batches
        if a request exceeds the service limits. The API remains backward
        compatible with the previous version which accepted either a single
        string or a list of strings and returned a numpy array of embeddings
        corresponding to the original inputs.

        Args:
            texts: Text or list of texts to embed.
            model: The embedding model to use.
            max_tokens: Maximum number of tokens allowed per chunk.
            batch_size: Maximum number of chunks to send in a single request.

        Returns:
            Numpy array of embeddings corresponding to ``texts``.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoding = tiktoken.get_encoding("cl100k_base")

        split_texts: List[str] = []
        mapping: List[List[int]] = []

        for text in texts:
            tokens = encoding.encode(text)
            indices: List[int] = []
            for i in range(0, len(tokens), max_tokens):
                chunk = encoding.decode(tokens[i : i + max_tokens])
                split_texts.append(chunk)
                indices.append(len(split_texts) - 1)
            mapping.append(indices)

        embeddings: List[List[float]] = []
        i = 0
        while i < len(split_texts):
            batch = split_texts[i : i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=model, input=batch, encoding_format="float"
                )
                embeddings.extend([dp.embedding for dp in response.data])
                i += batch_size
            except BadRequestError as e:
                if "Too many" in str(e) or "maximum context" in str(e):
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                else:
                    raise

        # aggregate embeddings for the original texts
        final_embeds: List[np.ndarray] = []
        for idx_list in mapping:
            parts = [embeddings[j] for j in idx_list]
            final_embeds.append(np.mean(np.array(parts), axis=0))

        return np.array(final_embeds)
