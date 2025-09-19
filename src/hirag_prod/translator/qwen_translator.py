from typing import Optional, Union

from openai import AsyncOpenAI

from hirag_prod._utils import logger
from hirag_prod.configs.functions import get_qwen_translator_config

LANGUAGE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "zh-TW": "Traditional Chinese",
    "zh-CN": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "auto": "Auto",  # Could be used as source language only
}


class QwenTranslator:

    class QwenTranslated:
        def __init__(
            self,
            text: str,
            src: str,
            dest: str,
            origin: str,
            pronunciation: Optional[str] = "",
            extra_data: Optional[dict] = None,
        ):
            self.text = text
            self.src = src
            self.dest = dest
            self.origin = origin
            self.extra_data = extra_data
            self.pronunciation = pronunciation

    def __init__(self):
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client for Qwen translation."""
        if self._client is None:
            config = get_qwen_translator_config()
            self._client = AsyncOpenAI(
                api_key=config.api_key, base_url=config.base_url, timeout=config.timeout
            )
            logger.info(f"🌐 Using Qwen Translator with model: {config.model_name}")
        return self._client

    def _validate_language(self, lang: str) -> None:
        """Validate language code"""
        if not lang:
            raise ValueError("Language code cannot be empty")
        if lang not in LANGUAGE_MAPPING and lang not in LANGUAGE_MAPPING.values():
            supported = list(LANGUAGE_MAPPING.keys()) + list(LANGUAGE_MAPPING.values())
            raise ValueError(f"Unsupported language: {lang}. Supported: {supported}")

    def _get_language_code(self, lang: str) -> str:
        if lang in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[lang]
        elif lang in LANGUAGE_MAPPING.values():
            return lang
        else:
            raise ValueError(f"Unsupported language: {lang}")

    async def translate(
        self, text, dest: str = "English", src: str = "Auto"
    ) -> Union[QwenTranslated, list[QwenTranslated]]:
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

    async def _translate_single(
        self, text: str, dest: str = "English", src: str = "Auto"
    ) -> QwenTranslated:
        """Translate a single text string using OpenAI client."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        self._validate_language(src)
        self._validate_language(dest)
        if dest == "Auto":
            raise ValueError("Destination language cannot be 'Auto'")

        try:
            dest_lang = self._get_language_code(dest)
            src_lang = self._get_language_code(src)

            messages = [{"role": "user", "content": f"{text}"}]
            translation_options = {"source_lang": src_lang, "target_lang": dest_lang}

            config = get_qwen_translator_config()
            client = self._get_client()
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                extra_body={"translation_options": translation_options},
            )

            translated = self.QwenTranslated(
                text=response.choices[0].message.content,
                src=src_lang,
                dest=dest_lang,
                origin=text,
                extra_data={"usage": response.usage},
            )
            return translated
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

    async def _translate_batch(
        self, texts: list[str], dest: str = "English", src: str = "Auto"
    ) -> list[QwenTranslated]:
        """Translate a batch of texts. Qwen does not support batch translation, so translate one by one."""
        results = []
        for text in texts:
            translated_text = await self._translate_single(text, dest, src)
            results.append(translated_text)
        return results
