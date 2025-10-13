import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx

from hirag_prod._utils import logger
from hirag_prod.configs.functions import (
    get_envs,
    get_shared_variables,
    get_translator_config,
)
from hirag_prod.rate_limiter import RateLimiter
from hirag_prod.resources.functions import get_chinese_convertor

rate_limiter = RateLimiter()

LANGUAGE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "zh-t-hk": "Traditional Chinese - Hong Kong",
    "zh-s": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "auto": "Auto",  # Could be used as source language only
}


class LocalTranslatorClient:
    """Client for local translator service"""

    def __init__(self):
        config = get_translator_config()
        self.base_url: str = config.base_url
        self.api_key: str = config.api_key
        self.model_name: str = config.model_name
        self.entry_point: str = config.entry_point
        self.timeout: float = config.timeout
        self._http_client = httpx.AsyncClient(timeout=self.timeout)

    async def create_translation(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create translation using local service API"""
        headers = {
            "Content-Type": "application/json",
            "Model-Name": self.model_name,
            "Entry-Point": self.entry_point,
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {"messages": messages, **kwargs}

        response = await self._http_client.post(
            self.base_url, headers=headers, json=payload
        )

        response.raise_for_status()
        result = response.json()

        return result

    async def close(self):
        """Close the HTTP client"""
        await self._http_client.aclose()


class LocalTranslator:

    class LocalTranslated:
        def __init__(
            self,
            text: str,
            src: str,
            dest: str,
            origin: str,
            extra_data: Optional[dict] = None,
        ) -> None:
            self.text = text
            self.src = src
            self.dest = dest
            self.origin = origin
            self.extra_data = extra_data

    def __init__(self) -> None:
        self._client: Optional[LocalTranslatorClient] = None

    def _get_client(self) -> LocalTranslatorClient:
        """Get or create the HTTP client for local translation."""
        if self._client is None:
            self._client = LocalTranslatorClient()
            logger.info(
                f"🌐 Using local translator with model: {self._client.model_name}, timeout: {self._client.timeout}"
            )
        return self._client

    def _get_language_name(self, lang: str) -> str:
        if lang in LANGUAGE_MAPPING.values():
            return lang
        elif lang in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[lang]
        else:
            raise ValueError(f"Unsupported language: {lang}")

    async def translate(
        self, text: Union[str, list[str]], dest: str = "English", src: str = "Auto"
    ) -> Union[LocalTranslated, list[LocalTranslated]]:
        """
        Translate text or list of texts.

        Args:
            text: str or list of str to translate
            dest: destination language
            src: source language

        Returns:
            Translated object or list of Translated objects
        """
        if isinstance(text, list):
            return await self._translate_batch(text, dest, src)
        else:
            return await self._translate_single(text, dest, src)

    @rate_limiter.limit(
        "translator",
        "TRANSLATOR_RATE_LIMIT_MIN_INTERVAL_SECONDS",
        "TRANSLATOR_RATE_LIMIT",
        "TRANSLATOR_RATE_LIMIT_TIME_UNIT",
    )
    async def _translate_single(
        self, text: str, dest: str = "English", src: str = "Auto"
    ) -> LocalTranslated:
        """Translate a single text string using HTTP client."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if dest == "Auto":
            raise ValueError("Destination language cannot be 'Auto'")

        try:
            original_dest_lang: str = self._get_language_name(dest)
            if original_dest_lang in [
                "Simplified Chinese",
                "Traditional Chinese - Hong Kong",
            ]:
                dest_lang = "Chinese"
            else:
                dest_lang = original_dest_lang
            src_lang: str = self._get_language_name(src)

            messages = [
                {
                    "role": "user",
                    "content": f"Translate the following segment into {dest_lang}, without additional explanation.\n\n{text}",
                }
            ]

            client = self._get_client()
            response = await client.create_translation(
                messages=messages,
            )
            if get_envs().ENABLE_TOKEN_COUNT:
                get_shared_variables().input_token_count_dict["translator"].value += (
                    response["usage"]["prompt_tokens"]
                    if ("usage" in response) and ("prompt_tokens" in response["usage"])
                    else 0
                )
                get_shared_variables().output_token_count_dict["translator"].value += (
                    response["usage"]["completion_tokens"]
                    if ("usage" in response)
                    and ("completion_tokens" in response["usage"])
                    else 0
                )

            translated_text: str = response["choices"][0]["message"]["content"]
            if original_dest_lang in [
                "Simplified Chinese",
                "Traditional Chinese - Hong Kong",
            ]:
                translated_text = get_chinese_convertor(
                    "hk2s" if original_dest_lang == "Simplified Chinese" else "s2hk"
                ).convert(translated_text)
            translated = self.LocalTranslated(
                text=translated_text,
                src=src_lang,
                dest=dest_lang,
                origin=text,
                extra_data={"usage": response["usage"]},
            )
            return translated
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

    async def _translate_batch(
        self,
        texts: list[str],
        dest: str = "English",
        src: str = "Auto",
        max_concurrency: int = 1000,
    ) -> list[LocalTranslated]:
        if not texts:
            return []
        sem = asyncio.Semaphore(max_concurrency)

        async def worker(t: str):
            async with sem:
                return await self._translate_single(t, dest, src)

        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(
            *[worker(t) for t in texts],
            desc="Translating",
        )
        return results
