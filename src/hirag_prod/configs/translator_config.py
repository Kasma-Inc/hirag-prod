from typing import Literal, Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class TranslatorConfig(BaseSettings):
    """Translator configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"translator_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    service_type: Literal["openai", "local"]

    # Translator settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None

    # Additional translator settings
    entry_point: Optional[str] = None
    timeout: Optional[float] = 3600.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
