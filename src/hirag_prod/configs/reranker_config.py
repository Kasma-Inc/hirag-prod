from pydantic_settings import BaseSettings


class RerankConfig(BaseSettings):
    """Reranker configuration"""

    class Config:
        extra = "allow"
