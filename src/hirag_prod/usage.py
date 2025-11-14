import contextvars
import functools
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

UnknownModelName = "unknown"


class ModelProvider(Enum):
    INTERNAL = "ofnil"
    ALIYUN = "aliyun"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelIdentifier:
    id: str
    provider: str


class ModelUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    count: int = 1

    @property
    def total(self) -> int:
        """Return the total number of tokens used."""
        return self.prompt_tokens + self.completion_tokens

    def add_usage(self, usage: "ModelUsage") -> None:
        """Accumulate usage from another ModelUsage instance."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.count += usage.count


class AggregateUsage(BaseModel):
    model_to_usages: dict[ModelIdentifier, ModelUsage] = Field(default_factory=dict)

    @property
    def count(self) -> int:
        """Return the number of unique model usages recorded."""
        count = 0
        for _, usage in self.model_to_usages.items():
            count += usage.count
        return count

    def add_usage(self, model_ident: ModelIdentifier, usage: ModelUsage) -> None:
        """Add usage for a specific model."""
        if model_ident not in self.model_to_usages:
            self.model_to_usages[model_ident] = usage
        else:
            self.model_to_usages[model_ident].add_usage(usage)

    def merge(self, other: "AggregateUsage") -> None:
        """Merge usage from another AggregateUsage instance."""
        for model_ident, usage in other.model_to_usages.items():
            self.add_usage(model_ident, usage)


_current_collector: contextvars.ContextVar[Optional["UsageCollector"]] = (
    contextvars.ContextVar("current_usage_collector", default=None)
)


class UsageCollector:
    """
    UsageCollector provides a context-aware mechanism for tracking model usage
    (e.g., token consumption and request counts) within asynchronous or synchronous workflows.

    It acts as a lightweight, context-bound accumulator of usage data,
    supporting both direct and indirect (static) updates within the same logical context.

    Key features:
    - Maintains a per-context global usage collector via `contextvars`.
    - Supports both async context management (`async with UsageCollector():`) and direct method calls.
    - Aggregates usage data across multiple models or providers through `AggregateUsage`.
    - Designed to be extended for customized behaviors like reporting.

    Typical use case:
    ```python
    async with UsageCollector() as collector:
        UsageCollector.add_usage(model_ident, usage)
        ...
        total = collector.get_usage()
    ```

    Attributes:
        _usage (AggregateUsage): The aggregated usage across models.
        _add_count (int): The total number of usage additions.
        has_error (bool): Indicates whether an error occurred during collection.
        _token (Optional[contextvars.Token]): Used internally for context variable management.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize a UsageCollector.
        """
        self._usage = AggregateUsage()
        self._add_count = 0
        self.has_error = False
        self._token: Optional[contextvars.Token] = None

    # -------------------------
    # Async context management
    # -------------------------
    async def __aenter__(self) -> "UsageCollector":
        self._token = _current_collector.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is not None:
                self.has_error = True
            await self._finalize()
        finally:
            if self._token is not None:
                _current_collector.reset(self._token)

    async def _finalize(self) -> None:
        pass

    # -------------------------
    # Accessors
    # -------------------------
    def get_usage(self) -> AggregateUsage:
        """Return the aggregated usage."""
        return self._usage

    def get_add_count(self) -> int:
        """Return the number of usage additions."""
        return self._add_count

    # -------------------------
    # Instance method
    # -------------------------
    def mark_error(self) -> None:
        """Mark that an error has occurred during usage recording."""
        self.has_error = True

    def add(self, model_ident: ModelIdentifier, usage: ModelUsage) -> None:
        """Add usage for a specific model instance."""
        self._usage.add_usage(model_ident, usage)
        self._add_count += 1

    def merge(self, agg_usage: AggregateUsage) -> None:
        """Add usage for a specific model instance."""
        self._usage.merge(agg_usage)
        self._add_count += agg_usage.count

    # -------------------------
    # Static method
    # -------------------------
    @staticmethod
    def add_usage(model_ident: ModelIdentifier, usage: ModelUsage) -> None:
        """Add usage to the current recorder, if any."""
        if (collector := _current_collector.get()) is not None:
            collector.add(model_ident, usage)

    @staticmethod
    def merge_usage(agg_usage: AggregateUsage) -> None:
        """Add usage to the current recorder, if any."""
        if (collector := _current_collector.get()) is not None:
            collector.merge(agg_usage)

    @staticmethod
    def current_collector() -> Optional["UsageCollector"]:
        """Return the current UsageCollector in context, if any."""
        return _current_collector.get()


def with_usage_collector():
    """
    Decorator that automatically wraps an async function with UsageCollector.

    Example:
        @with_usage_collector()
        async def on_message(...):
            ...

    Equivalent to:
        async def on_message(...):
            async with UsageCollector():
                ...
    """

    def decorator(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with UsageCollector():
                return await func(*args, **kwargs)

        return wrapper

    return decorator
