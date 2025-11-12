import contextvars
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

# Type alias for an async reporter callback function
AsyncReporter = Callable[["UsageRecorder", bool], Awaitable[None]]

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
    model_to_usages: dict[ModelIdentifier, ModelUsage] = {}

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


_current_recorder: contextvars.ContextVar[Optional["UsageRecorder"]] = (
    contextvars.ContextVar("current_usage_recorder", default=None)
)


class UsageRecorder:
    """
    UsageRecorder tracks model usage in a context (async/sync).

    Features:
    - Supports async context management
    - Allows global add_usage calls
    - Automatically aggregates usage
    """

    def __init__(
        self,
        user_id: str,
        workspace_id: str,
        chat_id: Optional[str] = None,
        msg_id: Optional[str] = None,
        reporter: Optional[AsyncReporter] = None,
    ) -> None:
        """
        Initialize a UsageRecorder.

        :param user_id: ID of the user
        :param workspace_id: ID of the workspace
        :param chat_id: Optional chat session ID
        :param msg_id: Optional message ID
        :param reporter: Optional async callback for reporting usage on exit
        """
        self.user_id: str = user_id
        self.workspace_id: str = workspace_id
        self.chat_id: Optional[str] = chat_id
        self.msg_id: Optional[str] = msg_id
        self._reporter = reporter
        self._usage = AggregateUsage()
        self._add_count = 0
        self.has_error = False
        self._token: Optional[contextvars.Token] = None

    # -------------------------
    # Async context management
    # -------------------------
    async def __aenter__(self) -> "UsageRecorder":
        self._token = _current_recorder.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is not None:
                self.has_error = True
            await self._finalize()
        finally:
            if self._token is not None:
                _current_recorder.reset(self._token)

    async def _finalize(self) -> None:
        """Call the async reporter on exit if provided."""
        if self._reporter:
            await self._reporter(self, self.has_error)

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

    # -------------------------
    # Static method
    # -------------------------
    @staticmethod
    def add_usage(model_ident: ModelIdentifier, usage: ModelUsage) -> None:
        """Add usage to the current recorder, if any."""
        recorder = _current_recorder.get()
        if recorder is not None:
            recorder.add(model_ident, usage)
