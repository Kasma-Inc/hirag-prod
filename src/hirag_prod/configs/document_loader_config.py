from pydantic_settings import BaseSettings


class DotsOCRConfig(BaseSettings):
    """Dots OCR configuration"""

    base_url: str
    api_key: str
    model_name: str = "DotsOCR"
    entry_point: str = "/parse/file"
    timeout: int = 300
    polling_interval: int = 5  # Polling interval in seconds for async jobs
    polling_retries: int = 3  # Number of retries for polling requests

    class Config:
        alias_generator = lambda x: f"dots_ocr_{x}".upper()
        populate_by_name = True
        extra = "ignore"
