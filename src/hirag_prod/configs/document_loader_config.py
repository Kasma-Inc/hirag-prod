from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class DotsOCRConfig(BaseSettings):
    """Dots OCR configuration"""

    base_url: str = Field(description="The base URL of the Dots OCR API.")
    api_key: SecretStr = Field(description="The API key for the Dots OCR API.")
    model_name: str = Field("DotsOCR", description="The DotsOCR model name.")
    entry_point: str = Field(
        "/parse/file", description="The entry point for the Dots OCR API."
    )
    timeout: int = Field(
        300, description="The timeout in seconds for the Dots OCR requests."
    )
    polling_interval: int = Field(
        5, description="The polling interval in seconds for async jobs"
    )
    polling_retries: int = Field(
        3, description="The number of retries for polling requests."
    )
    rate_limit: int = Field(60, description="The max number of requests per unit time.")
    rate_limit_time_unit: Literal["second", "minute", "hour"] = Field(
        "minute", description="The time unit for the rate limit."
    )
    rate_limit_min_interval_seconds: float = Field(
        0.1, description="The min interval in seconds between requests to Dots OCR."
    )

    class Config:
        @staticmethod
        def alias_generator(x: str) -> str:
            return f"dots_ocr_{x}".upper()

        populate_by_name = True
        extra = "ignore"
