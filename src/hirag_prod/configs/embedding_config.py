from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Embedding configuration"""

    class Config:
        extra = "allow"
