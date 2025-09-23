from pydantic_settings import BaseSettings


class DotsOCRConfig(BaseSettings):
    """Dots OCR configuration"""

    base_url: str
    api_key: str
    model_name: str = "DotsOCR"
    entry_point: str = "parse/file"
    timeout: int = 300

    class Config:
        alias_generator = lambda x: f"dots_ocr_{x}".upper()
        populate_by_name = True
        extra = "ignore"
