import asyncio
import logging
import os
import threading
import weakref
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
from aiolimiter import AsyncLimiter
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ============================================================================
# Constants
# ============================================================================


class EmbeddingServiceType(Enum):
    """Types of embedding services"""

    OPENAI = "openai"
    LOCAL = "local"


class APIConstants:
    """API configuration constants"""

    DEFAULT_RETRY_ATTEMPTS = 5
    DEFAULT_RETRY_MIN_WAIT = 1
    DEFAULT_RETRY_MAX_WAIT = 4
    DEFAULT_RATE_LIMIT = 20
    DEFAULT_RATE_PERIOD = 1
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    # Error keywords for batch size reduction
    BATCH_SIZE_ERROR_KEYWORDS = [
        "invalid_request_error",
        "too large",
        "limit exceeded",
        "input invalid",
        "request too large",
    ]


class LoggerNames:
    """Logger name constants"""

    TOKEN_USAGE = "HiRAG.TokenUsage"
    EMBEDDING = "HiRAG.Embedding"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TokenUsage:
    """Token usage information from OpenAI API response"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __str__(self) -> str:
        return f"Tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}, Total: {self.total_tokens}"


@dataclass
class APIConfig:
    """Base configuration for OpenAI APIs"""

    api_key: str
    base_url: str

    @classmethod
    def from_env(cls, api_key_env: str, base_url_env: str) -> "APIConfig":
        """Create config from environment variables"""
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(base_url_env)

        if not api_key:
            raise ValueError(f"{api_key_env} environment variable is not set")
        if not base_url:
            raise ValueError(f"{base_url_env} environment variable is not set")

        return cls(api_key=api_key, base_url=base_url)


@dataclass
class LocalEmbeddingConfig:
    """Configuration for local embedding service"""

    base_url: str
    model_name: str
    entry_point: str
    authorization_token: str
    model_path: str
    default_batch_size: int = 200  # Default batch size for local embedding service

    @classmethod
    def from_env(cls) -> "LocalEmbeddingConfig":
        """Create local embedding config from environment variables"""
        base_url = os.getenv("LOCAL_EMBEDDING_BASE_URL")
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
        entry_point = os.getenv("LOCAL_EMBEDDING_ENTRY_POINT", "/v1/embeddings")
        authorization_token = os.getenv("LOCAL_EMBEDDING_AUTH_TOKEN")
        model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")
        default_batch_size = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "200"))

        if not base_url:
            raise ValueError("LOCAL_EMBEDDING_BASE_URL environment variable is not set")
        if not model_name:
            raise ValueError(
                "LOCAL_EMBEDDING_MODEL_NAME environment variable is not set"
            )
        if not authorization_token:
            raise ValueError(
                "LOCAL_EMBEDDING_AUTH_TOKEN environment variable is not set"
            )
        if not model_path:
            raise ValueError(
                "LOCAL_EMBEDDING_MODEL_PATH environment variable is not set"
            )

        return cls(
            base_url=base_url,
            model_name=model_name,
            entry_point=entry_point,
            authorization_token=authorization_token,
            model_path=model_path,
            default_batch_size=default_batch_size,
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service with service type selection"""

    service_type: EmbeddingServiceType
    openai_config: Optional[APIConfig] = None
    local_config: Optional[LocalEmbeddingConfig] = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create embedding config from environment variables"""
        service_type_str = os.getenv("EMBEDDING_SERVICE_TYPE", "openai").lower()

        try:
            service_type = EmbeddingServiceType(service_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid EMBEDDING_SERVICE_TYPE: {service_type_str}. Must be 'openai' or 'local'"
            )

        if service_type == EmbeddingServiceType.OPENAI:
            openai_config = APIConfig.from_env(
                "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"
            )
            return cls(service_type=service_type, openai_config=openai_config)
        else:
            local_config = LocalEmbeddingConfig.from_env()
            return cls(service_type=service_type, local_config=local_config)


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiterManager:
    """Manages rate limiters for async operations"""

    @staticmethod
    def get_or_create_limiter(
        instance, limiter_attr: str, max_rate: int, time_period: int
    ) -> AsyncLimiter:
        """Get existing limiter or create new one for current event loop"""
        loop_attr = f"{limiter_attr}_loop"

        # Get current event loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new limiter
            limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
            setattr(instance, limiter_attr, limiter)
            setattr(instance, loop_attr, None)
            return limiter

        # Check if we have a limiter and if it's for the current loop
        if (
            hasattr(instance, limiter_attr)
            and hasattr(instance, loop_attr)
            and getattr(instance, limiter_attr) is not None
        ):
            stored_loop_ref = getattr(instance, loop_attr)
            # If stored loop reference exists and is the same as current loop
            if stored_loop_ref is not None and stored_loop_ref() is current_loop:
                return getattr(instance, limiter_attr)

        # Create new limiter for current loop
        limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
        setattr(instance, limiter_attr, limiter)
        # Store weak reference to current loop to avoid circular references
        setattr(instance, loop_attr, weakref.ref(current_loop))
        return limiter


def rate_limited(max_rate: int, time_period: int, limiter_attr: str):
    """Decorator for rate limiting async methods"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            limiter = RateLimiterManager.get_or_create_limiter(
                self, limiter_attr, max_rate, time_period
            )
            async with limiter:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


# Retry decorator for API calls
api_retry = retry(
    stop=stop_after_attempt(APIConstants.DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=APIConstants.DEFAULT_RETRY_MIN_WAIT,
        max=APIConstants.DEFAULT_RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)


# ============================================================================
# Base Classes
# ============================================================================


class SingletonABCMeta(type(ABC)):
    """Thread-safe singleton metaclass that works with ABC"""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonMeta(type):
    """Thread-safe singleton metaclass"""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseAPIClient(ABC, metaclass=SingletonABCMeta):
    """Base class for API clients with singleton pattern"""

    def __init__(self, config: APIConfig):
        if not hasattr(self, "_initialized"):
            self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
            self._initialized = True

    @property
    def client(self) -> AsyncOpenAI:
        return self._client


# ============================================================================
# Client Implementations
# ============================================================================


class ChatClient(BaseAPIClient):
    """Singleton wrapper for OpenAI async client for Chat"""

    def __init__(self):
        if not hasattr(self, "_initialized"):
            config = APIConfig.from_env("LLM_API_KEY", "LLM_BASE_URL")
            super().__init__(config)


class EmbeddingClient(BaseAPIClient):
    """Singleton wrapper for OpenAI async client for Embedding"""

    def __init__(self):
        if not hasattr(self, "_initialized"):
            config = APIConfig.from_env("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL")
            super().__init__(config)


class LocalEmbeddingClient:
    """Client for local embedding service"""

    def __init__(self, config: LocalEmbeddingConfig):
        self.config = config
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using local service API"""
        # Convert texts to messages format expected by local service
        batch_texts_to_embed = [text for text in texts]

        headers = {
            "Content-Type": "application/json",
            "Model-Name": self.config.model_name,
            "Entry-Point": self.config.entry_point,
            "Authorization": f"Bearer {self.config.authorization_token}",
        }

        payload = {"model": self.config.model_path, "input": batch_texts_to_embed}

        response = await self._http_client.post(
            self.config.base_url, headers=headers, json=payload
        )

        response.raise_for_status()
        result = response.json()

        # Extract embeddings from response
        if "data" in result:
            return [item["embedding"] for item in result["data"]]
        else:
            if isinstance(result, list):
                return result
            else:
                raise ValueError(
                    f"Unexpected response format from local embedding service: {result}"
                )

    async def close(self):
        """Close the HTTP client"""
        await self._http_client.aclose()


# ============================================================================
# Token Usage Tracking
# ============================================================================


class TokenUsageTracker:
    """Handles token usage tracking and statistics"""

    def __init__(self):
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._request_count = 0
        self._logger = logging.getLogger(LoggerNames.TOKEN_USAGE)

    def track_usage(self, usage_data, model: str, prompt_preview: str) -> TokenUsage:
        """Track token usage from API response"""
        token_usage = TokenUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

        # Update cumulative usage
        self._total_prompt_tokens += usage_data.prompt_tokens
        self._total_completion_tokens += usage_data.completion_tokens
        self._total_tokens += usage_data.total_tokens
        self._request_count += 1

        # Log usage information
        self._logger.info(f"🔢 {model} - Current: {token_usage}")
        self._logger.info(
            f"📊 Cumulative - Requests: {self._request_count}, "
            f"Prompt: {self._total_prompt_tokens}, "
            f"Completion: {self._total_completion_tokens}, "
            f"Total: {self._total_tokens}"
        )
        self._logger.debug(
            f"📝 Prompt preview: {prompt_preview[:100]}{'...' if len(prompt_preview) > 100 else ''}"
        )

        return token_usage

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cumulative token usage statistics"""
        return {
            "request_count": self._request_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_request": (
                round(self._total_tokens / self._request_count, 2)
                if self._request_count > 0
                else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset cumulative token usage statistics"""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._request_count = 0
        self._logger.info("🔄 Token usage statistics have been reset")

    def log_final_stats(self) -> None:
        """Log final token usage statistics summary"""
        if self._request_count > 0:
            stats = self.get_stats()
            self._logger.info("=" * 60)
            self._logger.info("🏁 FINAL TOKEN USAGE SUMMARY")
            self._logger.info(f"📝 Total Requests: {stats['request_count']}")
            self._logger.info(f"🔤 Total Prompt Tokens: {stats['total_prompt_tokens']}")
            self._logger.info(
                f"💬 Total Completion Tokens: {stats['total_completion_tokens']}"
            )
            self._logger.info(f"📊 Total Tokens: {stats['total_tokens']}")
            self._logger.info(
                f"📈 Average Tokens per Request: {stats['avg_tokens_per_request']}"
            )
            self._logger.info("=" * 60)
        else:
            self._logger.info("ℹ️ No API calls were made, no token usage to report")


# ============================================================================
# Main Service Classes
# ============================================================================


class ChatCompletion(metaclass=SingletonMeta):
    """Singleton handler for OpenAI chat completions"""

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.client = ChatClient().client
            self._completion_limiter = None
            self._token_tracker = TokenUsageTracker()
            self._initialized = True

    @rate_limited(
        max_rate=APIConstants.DEFAULT_RATE_LIMIT,
        time_period=APIConstants.DEFAULT_RATE_PERIOD,
        limiter_attr="_completion_limiter",
    )
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
        messages = self._build_messages(system_prompt, history_messages, prompt)

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        # Track token usage
        self._token_tracker.track_usage(response.usage, model, prompt)

        return response.choices[0].message.content

    def _build_messages(
        self,
        system_prompt: Optional[str],
        history_messages: Optional[List[Dict[str, str]]],
        prompt: str,
    ) -> List[Dict[str, str]]:
        """Build messages list for API call"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})
        return messages

    def get_token_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get cumulative token usage statistics"""
        return self._token_tracker.get_stats()

    def reset_token_usage_stats(self) -> None:
        """Reset cumulative token usage statistics"""
        self._token_tracker.reset_stats()

    def log_final_stats(self) -> None:
        """Log final token usage statistics summary"""
        self._token_tracker.log_final_stats()


# ============================================================================
# Embedding Service
# ============================================================================


class BatchProcessor:
    """Handles batch processing logic for embeddings"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def process_with_adaptive_batching(
        self, texts: List[str], batch_size: int, process_func, model: str
    ) -> np.ndarray:
        """Process texts with adaptive batch sizing"""
        self._logger.info(
            f"🔄 Processing {len(texts)} texts in batches of {batch_size}"
        )

        all_embeddings = []
        current_batch_size = batch_size
        i = 0

        while i < len(texts):
            batch_texts = texts[i : i + current_batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            self._logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} "
                f"({len(batch_texts)} texts, batch_size={current_batch_size})"
            )

            try:
                batch_embeddings = await process_func(batch_texts, model)
                all_embeddings.append(batch_embeddings)

                self._logger.info("✅ Batch completed successfully")
                i += current_batch_size

                # Reset batch size to original after successful batch
                if current_batch_size < batch_size:
                    current_batch_size = min(batch_size, current_batch_size * 2)
                    self._logger.info(
                        f"📈 Increasing batch size back to {current_batch_size}"
                    )

            except Exception as e:
                current_batch_size = self._handle_batch_error(
                    e, current_batch_size, batch_texts[0] if batch_texts else ""
                )
                if current_batch_size == 0:  # Error was re-raised
                    raise

        return np.concatenate(all_embeddings, axis=0)

    def _handle_batch_error(
        self, error: Exception, current_batch_size: int, sample_text: str
    ) -> int:
        """Handle batch processing errors with adaptive sizing"""
        error_msg = str(error).lower()

        # Check if error is related to input size/limits
        if any(
            keyword in error_msg for keyword in APIConstants.BATCH_SIZE_ERROR_KEYWORDS
        ):
            if current_batch_size > 1:
                # Reduce batch size and retry
                new_batch_size = max(1, current_batch_size // 2)
                self._logger.warning(
                    f"⚠️ API limit error, reducing batch size from {current_batch_size} to {new_batch_size}"
                )
                self._logger.warning(f"⚠️ Error details: {error}")
                return new_batch_size
            else:
                # Even single text fails, this is a different issue
                self._logger.error(
                    "❌ Even single text embedding failed, this may be a content issue"
                )
                self._logger.error(f"❌ Failed text preview: {sample_text[:200]}...")
                raise error
        else:
            # Different type of error, don't retry
            self._logger.error(
                f"❌ Non-batch-size related error in batch processing: {error}"
            )
            raise error


class TextValidator:
    """Validates and cleans text inputs for embedding"""

    @staticmethod
    def validate_and_clean(texts: List[str]) -> List[str]:
        """Validate and clean texts for embedding"""
        if not texts:
            raise ValueError("texts list cannot be empty")

        valid_texts = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"Text at index {i} is None")
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} is not a string: {type(text)}")

            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError(
                    f"Text at index {i} is empty after stripping whitespace"
                )

            valid_texts.append(cleaned_text)

        return valid_texts


class EmbeddingService(metaclass=SingletonMeta):
    """Singleton handler for OpenAI embeddings"""

    def __init__(self, default_batch_size: int = APIConstants.DEFAULT_BATCH_SIZE):
        if not hasattr(self, "_initialized"):
            self.client = EmbeddingClient().client
            self._embedding_limiter = None
            self.default_batch_size = default_batch_size
            self._logger = logging.getLogger(LoggerNames.EMBEDDING)
            self._batch_processor = BatchProcessor(self._logger)
            self._text_validator = TextValidator()
            self._initialized = True
            self._logger.debug(
                f"🔧 EmbeddingService initialized with batch_size={default_batch_size}"
            )
        else:
            self._logger.debug(
                f"🔧 EmbeddingService already initialized, keeping existing batch_size={self.default_batch_size}"
            )

    @rate_limited(
        max_rate=APIConstants.DEFAULT_RATE_LIMIT,
        time_period=APIConstants.DEFAULT_RATE_PERIOD,
        limiter_attr="_embedding_limiter",
    )
    @api_retry
    async def _create_embeddings_batch(
        self, texts: List[str], model: str = APIConstants.DEFAULT_EMBEDDING_MODEL
    ) -> np.ndarray:
        """Create embeddings for a single batch of texts (internal method)"""
        response = await self.client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    async def create_embeddings(
        self,
        texts: List[str],
        model: str = APIConstants.DEFAULT_EMBEDDING_MODEL,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create embeddings for the given texts with automatic batching for large inputs.

        Args:
            texts: List of texts to embed
            model: The embedding model to use (ignored for local service)
            batch_size: Maximum number of texts to process in a single API call (uses default if None)

        Returns:
            Numpy array of embeddings
        """
        # Validate and clean texts
        valid_texts = self._text_validator.validate_and_clean(texts)

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size

        # If batch size is small enough, process directly
        if len(valid_texts) <= batch_size:
            return await self._create_embeddings_batch(valid_texts, model)

        # Process in batches for large inputs with adaptive batch sizing
        result = await self._batch_processor.process_with_adaptive_batching(
            valid_texts, batch_size, self._create_embeddings_batch, model
        )

        self._logger.info(
            f"✅ All {len(valid_texts)} embeddings processed successfully"
        )
        return result

    async def close(self):
        """Close underlying clients"""
        await self.client.close()


class LocalEmbeddingService:
    """Simplified embedding service for local services with batch support"""

    def __init__(self, default_batch_size: Optional[int] = None):
        """Initialize with direct local client (no singleton)"""
        self.config = LocalEmbeddingConfig.from_env()
        self.client = LocalEmbeddingClient(self.config)
        self._logger = logging.getLogger(LoggerNames.EMBEDDING)

        # Set batch size
        self.default_batch_size = default_batch_size or self.config.default_batch_size

        # Initialize batch processor and text validator
        self._batch_processor = BatchProcessor(self._logger)
        self._text_validator = TextValidator()

        self._logger.info(
            f"🔧 LocalEmbeddingService initialized with batch_size={self.default_batch_size}"
        )

    async def _create_embeddings_batch(
        self, texts: List[str], model: str = ""
    ) -> np.ndarray:
        """Create embeddings for a single batch of texts (internal method)"""
        embeddings_list = await self.client.create_embeddings(texts)
        return np.array(embeddings_list)

    async def create_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Create embeddings using local service with batch support.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts to process in a single API call (uses default if None)

        Returns:
            Numpy array of embeddings
        """
        # Validate and clean texts
        valid_texts = self._text_validator.validate_and_clean(texts)

        # Use default batch size if not specified
        effective_batch_size = batch_size or self.default_batch_size

        self._logger.info(
            f"🔄 Processing {len(valid_texts)} texts with batch_size={effective_batch_size}"
        )

        # If batch size is small enough, process directly
        if len(valid_texts) <= effective_batch_size:
            return await self._create_embeddings_batch(valid_texts)

        # Process in batches for large inputs with adaptive batch sizing
        result = await self._batch_processor.process_with_adaptive_batching(
            valid_texts, effective_batch_size, self._create_embeddings_batch, ""
        )

        self._logger.info(
            f"✅ Completed processing {len(valid_texts)} texts, result shape: {result.shape}"
        )
        return result

    async def close(self):
        """Close the underlying client"""
        await self.client.close()


def create_embedding_service(
    default_batch_size: int = APIConstants.DEFAULT_BATCH_SIZE,
) -> Union[EmbeddingService, LocalEmbeddingService]:
    """
    Factory function to create appropriate embedding service.

    For local services, returns LocalEmbeddingService with batch support.
    For OpenAI services, returns the full EmbeddingService.
    """
    # Check service type from environment
    service_type = os.getenv("EMBEDDING_SERVICE_TYPE", "openai").lower()

    if service_type == "local":
        return LocalEmbeddingService(default_batch_size)
    else:
        return EmbeddingService(default_batch_size)
