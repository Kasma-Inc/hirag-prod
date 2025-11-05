
from pydantic_settings import BaseSettings


class OcrConfig(BaseSettings):
    """OCR configuration"""

    class Config:
        extra = "allow"
