import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from hirag_prod._utils import log_error_info
from hirag_prod.configs.provider_key_config import ProviderKeyConfigs
from hirag_prod.resources.client_common import APIConstants
from hirag_prod.tracing import traced
from hirag_prod.usage import (
    ModelIdentifier,
    ModelProvider,
    ModelUsage,
    UsageRecorder,
)


class BaseEmbeddingClient(ABC):
    """Base class for embedding clients"""

    @abstractmethod
    async def create_embeddings(
        self, model: str, texts: List[str]
    ) -> List[List[float]]:
        """Create embeddings using the embedding client"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the embedding client"""
        ...


# ============================================================================
# Client Implementations
# ============================================================================


class AsyncOpenAIEmbeddingClient(BaseEmbeddingClient):
    """Client for local embedding service (OpenAI SDK based)"""

    def __init__(self, cfg: ProviderKeyConfigs):
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key.get_secret_value(),
            max_retries=0,
        )

    @traced()
    async def create_embeddings(
        self, model: str, texts: List[str]
    ) -> List[List[float]]:
        """Create embeddings using OpenAI SDK"""
        batch_texts_to_embed = [text for text in texts]

        embedding_config = get_embedding_config(model)
        resp = await self._client.embeddings.create(
            model=model,
            input=batch_texts_to_embed,
            extra_headers=embedding_config.extra_headers,
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


class BatchProcessor:
    """Handles batch processing logic for embeddings"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def process_with_adaptive_batching(
        self, model: str, texts: List[str], batch_size: int, process_func
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
                batch_embeddings = await process_func(model, batch_texts)
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


class BatchEmbeddingService:
    """Batch embedding service with adaptive batching"""

    client_dict: Dict[str, BaseEmbeddingClient] = {}

    def __init__(self, provider: str, config: Dict[str, Any]):
        if provider not in self.client_dict:
            self.client_dict[provider] = AsyncOpenAIEmbeddingClient(config)

        self.client = self.client_dict[provider]

        self._logger = logging.getLogger()
        # Initialize batch processor and text validator
        self._batch_processor = BatchProcessor(self._logger)
        self._text_validator = TextValidator()

        self._logger.info(
            f"🔧 LocalEmbeddingService initialized for provider {provider}"
        )

    async def _create_embeddings_batch(
        self, model: str, texts: List[str]
    ) -> np.ndarray:
        """Create embeddings for a single batch of texts (internal method)"""
        embeddings_list = await self.client.create_embeddings(model, texts)

        # token counter moves to LocalEmbeddingClient.create_embeddings()
        return np.array(embeddings_list)

    async def create_embeddings(
        self, model: str, texts: List[str], batch_size: Optional[int] = None
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
            dim = get_embedding_config().dimension
            return np.zeros((total_count, dim), dtype=np.float32)

        self._logger.info(
            f"🔄 Processing {len(valid_texts)} texts with batch_size={effective_batch_size}"
        )

        # Embed valid texts (batched if necessary)
        if len(valid_texts) <= effective_batch_size:
            embeddings = await self._create_embeddings_batch(model, valid_texts)
        else:
            embeddings = await self._batch_processor.process_with_adaptive_batching(
                model, valid_texts, effective_batch_size, self._create_embeddings_batch
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
            else get_embedding_config().dimension
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
