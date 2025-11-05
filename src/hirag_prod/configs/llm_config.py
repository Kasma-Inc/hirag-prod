from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM configuration"""

    class Config:
        extra = "allow"
