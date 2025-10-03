from typing import Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class QwenTranslatorConfig(BaseSettings):
    """Qwen Translator configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"DASHSCOPE_TRANSLATOR_{x.upper()}",
        populate_by_name=True,
        extra="ignore",
    )

    # Qwen Translator Dashscope settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "qwen-mt-turbo"

    # Additional translator settings
    timeout: float = 30.0
    max_tokens: int = 2048
    temperature: float = 0.1

    @model_validator(mode="after")
    def validate_translator_config(self) -> "QwenTranslatorConfig":
        """Validate that required fields are present for Qwen translator"""
        if not self.api_key:
            raise ValueError("api_key is required for Qwen translator")
        if not self.base_url:
            raise ValueError("base_url is required for Qwen translator")
        return self

    # Remove the property methods since we're using direct field names now
