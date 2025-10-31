import logging
import threading
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import httpx
import numpy as np
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hirag_prod._utils import log_error_info
from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.functions import (
    get_embedding_config,
    get_envs,
    get_init_config,
    get_llm_config,
    get_shared_variables,
)
from hirag_prod.configs.llm_config import LLMConfig
from hirag_prod.rate_limiter import RateLimiter
from hirag_prod.tracing import traced
from hirag_prod.usage import (
    ModelIdentifier,
    ModelProvider,
    ModelUsage,
    UnknownModelName,
    UsageRecorder,
)

# ============================================================================
# Constants
# ============================================================================


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
    CHAT = "HiRAG.Chat"


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


# ============================================================================
# Rate Limiting
# ============================================================================

rate_limiter = RateLimiter()

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

    def __init__(self, config: Union[EmbeddingConfig, LLMConfig]):
        if not hasattr(self, "_initialized"):
            self._client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                max_retries=0,
            )
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
            super().__init__(get_llm_config())


class EmbeddingClient(BaseAPIClient):
    """Singleton wrapper for OpenAI async client for Embedding"""

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(get_embedding_config())


class LocalEmbeddingClient:
    """Client for local embedding service (OpenAI SDK based)"""

    def __init__(self):
        self._logger = logging.getLogger(LoggerNames.EMBEDDING)
        self._client = AsyncOpenAI(
            base_url=get_embedding_config().base_url,
            api_key=get_embedding_config().api_key.get_secret_value(),
            max_retries=0,
        )

    @traced()
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI SDK against local service"""
        batch_texts_to_embed = [text for text in texts]

        headers = {
            "Model-Name": get_embedding_config().model_name,
            "Entry-Point": get_embedding_config().entry_point,
            "Authorization": f"Bearer {get_embedding_config().api_key.get_secret_value()}",
        }

        resp = await self._client.embeddings.create(
            model=get_embedding_config().model_path,
            input=batch_texts_to_embed,
            extra_headers=headers,
        )

        if get_envs().ENABLE_TOKEN_COUNT:
            get_shared_variables().input_token_count_dict["embedding"].value += (
                resp.usage.prompt_tokens if resp.usage.prompt_tokens is not None else 0
            )
        UsageRecorder.add_usage(
            ModelIdentifier(
                id=resp.model,
                provider=getattr(resp, "provider", ModelProvider.UNKNOWN.value),
            ),
            ModelUsage(
                prompt_tokens=resp.usage.prompt_tokens,
            ),
        )

        embeddings = [item.embedding for item in resp.data]
        self._logger.info(f"✅ Completed processing {len(embeddings)} texts")
        return embeddings

    async def close(self):
        """Close the OpenAI client"""
        await self._client.close()


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

    T = TypeVar("T", bound=BaseModel)

    @api_retry
    @rate_limiter.limit()
    @traced()
    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[type[T]] = None,
        **kwargs: Any,
    ) -> Union[str, T]:
        """
        Complete a chat prompt using the specified model.

        Args:
            model: The model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt: The user prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            response_format: Optional response format
            **kwargs: Additional parameters for the API call

        Returns:
            The completion response as a string
        """
        messages = self._build_messages(system_prompt, history_messages, prompt)

        if response_format is None:
            response = await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )

        # Track token usage
        token_usage = self._token_tracker.track_usage(response.usage, model, prompt)
        if get_envs().ENABLE_TOKEN_COUNT:
            get_shared_variables().input_token_count_dict["llm"].value += (
                token_usage.prompt_tokens
                if token_usage.prompt_tokens is not None
                else 0
            )
            get_shared_variables().output_token_count_dict["llm"].value += (
                token_usage.completion_tokens
                if token_usage.completion_tokens is not None
                else 0
            )
        UsageRecorder.add_usage(
            ModelIdentifier(
                id=response.model,
                provider=getattr(response, "provider", ModelProvider.UNKNOWN.value),
            ),
            ModelUsage(
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
            ),
        )

        if response_format is None:
            return response.choices[0].message.content
        else:
            return response.choices[0].message.parsed

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

    async def close(self):
        """Close underlying client"""
        await self.client.close()

# ============================================================================
# Embedding Service
# ============================================================================


class BatchProcessor:
    """Handles batch processing logic for embeddings"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def process_with_adaptive_batching(
        self, texts: List[str], batch_size: int, process_func 
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
                batch_embeddings = await process_func(batch_texts, APIConstants.DEFAULT_EMBEDDING_MODEL)
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
    ) -> Optional[int]:
        """Handle batch processing errors with adaptive sizing"""
        error_msg = str(error).lower()

        # Check if error is related to input size/limits
        if any(
            keyword in error_msg for keyword in APIConstants.BATCH_SIZE_ERROR_KEYWORDS
        ):
            if current_batch_size > 1:
                # Reduce batch size and retry
                new_batch_size = max(1, current_batch_size // 2)
                log_error_info(
                    logging.WARNING,
                    f"⚠️ API limit error, reducing batch size from {current_batch_size} to {new_batch_size}",
                    error,
                )
                return new_batch_size
            else:
                # Even single text fails, this is a different issue
                log_error_info(
                    logging.ERROR,
                    f"❌ Even single text embedding failed, this may be a content issue. Failed text preview: {sample_text[:200]}...",
                    error,
                    raise_error=True,
                )
                return None
        else:
            # Different type of error, don't retry
            log_error_info(
                logging.ERROR,
                "❌ Non-batch-size related error in batch processing",
                error,
                raise_error=True,
            )
            return None


class TextValidator:
    """Validates and cleans text inputs for embedding"""

    @staticmethod
    def validate_and_clean(
        texts: List[Optional[str]],
    ) -> Tuple[List[str], List[int], int]:
        """Return (non-empty cleaned texts, their original indices, total count)."""
        if not texts:
            return [], [], 0

        valid_texts: List[str] = []
        valid_indices: List[int] = []
        for i, text in enumerate(texts):
            if text is None:
                continue
            cleaned_text = text.strip()
            if not cleaned_text:
                continue
            valid_texts.append(cleaned_text)
            valid_indices.append(i)

        return valid_texts, valid_indices, len(texts)


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

    @api_retry
    @rate_limiter.limit(
        "cloud-embedding"
    )
    async def _create_embeddings_batch(
        self, texts: List[str]
    ) -> np.ndarray:
        """Create embeddings for a single batch of texts (internal method)"""
        response = await self.client.embeddings.create(
            model=APIConstants.DEFAULT_EMBEDDING_MODEL, input=texts, encoding_format="float"
        )
        token_usage = response.usage
        if get_envs().ENABLE_TOKEN_COUNT:
            # embedding model only needs to count the prompt tokens
            get_shared_variables().input_token_count_dict[
                "embedding"
            ].value += token_usage.prompt_tokens
        UsageRecorder.add_usage(
            ModelIdentifier(
                id=response.model,
                provider=getattr(response, "provider", ModelProvider.UNKNOWN.value),
            ),
            ModelUsage(prompt_tokens=token_usage.prompt_tokens),
        )

        return np.array([dp.embedding for dp in response.data])

    @traced()
    async def create_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create embeddings for the given texts with automatic batching for large inputs.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts to process in a single API call (uses default if None)

        Returns:
            Numpy array of embeddings
        """
        # Partition texts into valid and empty
        valid_texts, valid_indices, total_count = (
            self._text_validator.validate_and_clean(texts)
        )

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size

        # If nothing valid, return an empty (0, dim) array
        if total_count == 0 or len(valid_texts) == 0:
            dim = get_init_config().EMBEDDING_DIMENSION
            return np.zeros((total_count, dim), dtype=np.float32)

        # Embed valid texts (batched if necessary)
        if len(valid_texts) <= batch_size:
            embeddings = await self._create_embeddings_batch(valid_texts)
        else:
            embeddings = await self._batch_processor.process_with_adaptive_batching(
                valid_texts, batch_size, self._create_embeddings_batch
            )

        # If no empties, return directly
        if len(valid_texts) == total_count:
            self._logger.info(
                f"✅ All {len(valid_texts)} embeddings processed successfully"
            )
            return embeddings

        # Restore original order with zeros for empty inputs
        embedding_dim = (
            embeddings.shape[1]
            if embeddings.size > 0
            else get_init_config().EMBEDDING_DIMENSION
        )
        result = np.zeros(
            (total_count, embedding_dim),
            dtype=embeddings.dtype if embeddings.size > 0 else np.float32,
        )
        result[valid_indices] = embeddings
        self._logger.info(
            f"✅ Processed {len(valid_texts)} non-empty texts; filled {total_count - len(valid_texts)} empty with zeros"
        )
        return result

    async def close(self):
        """Close underlying clients"""
        await self.client.close()


class LocalEmbeddingService:
    """Simplified embedding service for local services with batch support"""

    def __init__(self, default_batch_size: Optional[int] = None):
        """Initialize with direct local client (no singleton)"""
        self.client = LocalEmbeddingClient()
        self._logger = logging.getLogger(LoggerNames.EMBEDDING)

        # Set batch size
        self.default_batch_size = (
            default_batch_size or get_embedding_config().default_batch_size
        )

        # Initialize batch processor and text validator
        self._batch_processor = BatchProcessor(self._logger)
        self._text_validator = TextValidator()

        self._logger.info(
            f"🔧 LocalEmbeddingService initialized with batch_size={self.default_batch_size}"
        )

    @rate_limiter.limit(
        "qwen3-embedding-local"
    )
    async def _create_embeddings_batch(
        self, texts: List[str], model: str = ""
    ) -> np.ndarray:
        """Create embeddings for a single batch of texts (internal method)"""
        embeddings_list = await self.client.create_embeddings(texts)

        # token counter moves to LocalEmbeddingClient.create_embeddings()
        return np.array(embeddings_list)

    @traced()
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
        # Partition texts into valid and empty
        valid_texts, valid_indices, total_count = (
            self._text_validator.validate_and_clean(texts)
        )

        # Use default batch size if not specified
        effective_batch_size = batch_size or self.default_batch_size

        # If nothing valid, return an empty (0, dim) array
        if total_count == 0 or len(valid_texts) == 0:
            dim = get_init_config().EMBEDDING_DIMENSION
            return np.zeros((total_count, dim), dtype=np.float32)

        self._logger.info(
            f"🔄 Processing {len(valid_texts)} texts with batch_size={effective_batch_size}"
        )

        # Embed valid texts (batched if necessary)
        if len(valid_texts) <= effective_batch_size:
            embeddings = await self._create_embeddings_batch(valid_texts)
        else:
            embeddings = await self._batch_processor.process_with_adaptive_batching(
                valid_texts, effective_batch_size, self._create_embeddings_batch, ""
            )

        # If no empties, return directly
        if len(valid_texts) == total_count:
            self._logger.info(
                f"✅ Completed processing {total_count} texts, result shape: {embeddings.shape}"
            )
            return embeddings

        # Restore original order with zeros for empty inputs
        embedding_dim = (
            embeddings.shape[1]
            if embeddings.size > 0
            else get_init_config().EMBEDDING_DIMENSION
        )
        result = np.zeros(
            (total_count, embedding_dim),
            dtype=embeddings.dtype if embeddings.size > 0 else np.float32,
        )
        result[valid_indices] = embeddings
        self._logger.info(
            f"✅ Processed {len(valid_texts)} non-empty texts; filled {total_count - len(valid_texts)} empty with zeros"
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
    service_type = get_embedding_config().service_type.lower()

    if service_type == "local":
        return LocalEmbeddingService(default_batch_size)
    else:
        return EmbeddingService(default_batch_size)


def create_chat_service() -> ChatCompletion:
    """
    Factory function to create appropriate chat service.

    For OpenAI services, returns the singleton ChatCompletion.
    """
    return ChatCompletion()
