from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Union

from configs.functions import (
    get_client_key_config,
    get_envs,
    get_shared_variables,
)
from configs.provider_key_config import ProviderKeyConfigs
from openai import AsyncOpenAI
from pydantic import BaseModel

from hirag_prod.tracing import traced

# ============================================================================
# Base Classes
# ============================================================================


class BaseLLMClient(ABC):
    """all providers must implement the unified interface"""

    @abstractmethod
    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel]:
        """return text or Pydantic parsed object"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """release resources"""
        ...


# ============================================================================
# Specific Provider Implementation
# ============================================================================
class AsyncOpenAIChatCompletionClient(BaseLLMClient):
    def __init__(self, cfg: ProviderKeyConfigs):
        self._client = AsyncOpenAI(
            api_key=cfg.api_key.get_secret_value(),
            base_url=cfg.base_url,
            max_retries=0,
        )

    @traced()
    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel]:
        messages = self._build_messages(system_prompt, history_messages, prompt)

        if response_format is None:
            resp = await self._client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            content = resp.choices[0].message.content
        else:
            resp = await self._client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            content = resp.choices[0].message.parsed

        self._track_usage(resp)
        return content

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _build_messages(
        system_prompt: Optional[str],
        history_messages: Optional[List[Dict[str, str]]],
        prompt: str,
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if history_messages:
            msgs.extend(history_messages)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _track_usage(self, resp: Any) -> None:
        if not get_envs().ENABLE_TOKEN_COUNT:
            return
        usage = resp.usage
        get_shared_variables().input_token_count_dict[
            "llm"
        ].value += usage.prompt_tokens
        get_shared_variables().output_token_count_dict[
            "llm"
        ].value += usage.completion_tokens


def get_llm_client(
    provider: str, cfg: Optional[ProviderKeyConfigs] = None
) -> BaseLLMClient:
    """
    return the corresponding client instance based on provider.
    """
    api_compatible = get_client_key_config(provider)["api_compatible"]
    if api_compatible == "openai":
        return AsyncOpenAIChatCompletionClient(cfg)
    else:
        raise ValueError(f"Unsupported api compatible: {api_compatible}")


# ============================================================================
# Unified ChatCompletion
# ============================================================================
class ChatCompletion:
    """
    Each time `create_chat_service` is called, the corresponding client is created based on the provider.
    The internal tracing, and token-count are kept consistent.
    """

    T = TypeVar("T", bound=BaseModel)
    provider_dict: Dict[str, BaseLLMClient] = {}

    def __init__(self, provider: str, config: Optional[ProviderKeyConfigs] = None):
        if provider not in self.provider_dict:
            self.provider_dict[provider] = get_llm_client(provider, config)
        self._client: BaseLLMClient = self.provider_dict[provider]

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
        return await self._client.complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            response_format=response_format,
            **kwargs,
        )

    async def close(self) -> None:
        await self._client.close()


def create_chat_service(
    provider: str, config: Optional[ProviderKeyConfigs] = None
) -> ChatCompletion:
    return ChatCompletion(provider, config)
