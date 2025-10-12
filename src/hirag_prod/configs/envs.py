from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class InitEnvs(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    EMBEDDING_DIMENSION: int
    USE_HALF_VEC: bool = True
    HIRAG_QUERY_TIMEOUT: int = 100  # seconds


class Envs(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    ENV: Literal["dev", "prod"] = "dev"
    HI_RAG_LANGUAGE: str = "en"
    POSTGRES_URL_NO_SSL: str
    POSTGRES_URL_NO_SSL_DEV: str
    POSTGRES_TABLE_NAME: str = "KnowledgeBaseJobs"
    POSTGRES_SCHEMA: str = "public"
    REDIS_URL: str = "redis://redis:6379/2"
    REDIS_KEY_PREFIX: str = "hirag"
    REDIS_EXPIRE_TTL: int = 3600 * 24
    EMBEDDING_DIMENSION: int
    USE_HALF_VEC: bool = True
    ENABLE_TOKEN_COUNT: bool = False
    CONSTRUCT_GRAPH: bool = False

    EMBEDDING_SERVICE_TYPE: Literal["openai", "local"] = "openai"
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_BASE_URL: Optional[str] = None
    OPENAI_EMBEDDING_API_KEY: Optional[str] = None
    LOCAL_EMBEDDING_BASE_URL: Optional[str] = None
    LOCAL_EMBEDDING_API_KEY: Optional[str] = None

    LLM_SERVICE_TYPE: Literal["openai", "local"] = "openai"
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    OPENAI_LLM_BASE_URL: Optional[str] = None
    OPENAI_LLM_API_KEY: Optional[str] = None
    LOCAL_LLM_BASE_URL: Optional[str] = None
    LOCAL_LLM_API_KEY: Optional[str] = None

    TRANSLATOR_SERVICE_TYPE: Literal["openai", "local"] = "local"
    TRANSLATOR_BASE_URL: Optional[str] = None
    TRANSLATOR_API_KEY: Optional[str] = None
    TRANSLATOR_MODEL_NAME: Optional[str] = None
    TRANSLATOR_ENTRY_POINT: Optional[str] = None
    TRANSLATOR_TIME_OUT: Optional[str] = None
    TRANSLATOR_MAX_TOKENS: Optional[str] = None
    TRANSLATOR_TEMPERATURE: Optional[str] = None
    OPENAI_TRANSLATOR_BASE_URL: Optional[str] = None
    OPENAI_TRANSLATOR_API_KEY: Optional[str] = None
    OPENAI_TRANSLATOR_MODEL_NAME: Optional[str] = None
    OPENAI_TRANSLATOR_ENTRY_POINT: Optional[str] = None
    OPENAI_TRANSLATOR_TIME_OUT: Optional[str] = None
    OPENAI_TRANSLATOR_MAX_TOKENS: Optional[str] = None
    OPENAI_TRANSLATOR_TEMPERATURE: Optional[str] = None
    LOCAL_TRANSLATOR_BASE_URL: Optional[str] = None
    LOCAL_TRANSLATOR_API_KEY: Optional[str] = None
    LOCAL_TRANSLATOR_MODEL_NAME: Optional[str] = None
    LOCAL_TRANSLATOR_ENTRY_POINT: Optional[str] = None
    LOCAL_TRANSLATOR_TIME_OUT: Optional[str] = None
    LOCAL_TRANSLATOR_MAX_TOKENS: Optional[str] = None
    LOCAL_TRANSLATOR_TEMPERATURE: Optional[str] = None

    # Qwen Translator Configuration
    DASHSCOPE_TRANSLATOR_API_KEY: Optional[str] = None
    DASHSCOPE_TRANSLATOR_BASE_URL: Optional[str] = None
    DASHSCOPE_TRANSLATOR_MODEL_NAME: str = "qwen-mt-plus"

    RERANKER_TYPE: Literal["api", "local"] = "api"

    # API reranker (Voyage AI) settings
    VOYAGE_API_KEY: Optional[str] = None
    VOYAGE_RERANKER_MODEL_NAME: str = "rerank-2"
    VOYAGE_RERANKER_MODEL_BASE_URL: str = "https://api.voyageai.com/v1/rerank"

    # Local reranker settings
    LOCAL_RERANKER_MODEL_BASE_URL: Optional[str] = None
    LOCAL_RERANKER_MODEL_NAME: str = "Qwen3-Reranker-8B"
    LOCAL_RERANKER_MODEL_ENTRY_POINT: str = "/rerank"
    LOCAL_RERANKER_MODEL_AUTHORIZATION: Optional[str] = None

    KNOWLEDGE_BASE_SEARCH_BATCH_SIZE: int = 10000

    LLM_RATE_LIMIT: int = 60
    LLM_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    LLM_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1
    EMBEDDING_RATE_LIMIT: int = 60
    EMBEDDING_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    EMBEDDING_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1
    RERANKER_RATE_LIMIT: int = 60
    RERANKER_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    RERANKER_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1
    TRANSLATOR_RATE_LIMIT: int = 60
    TRANSLATOR_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    TRANSLATOR_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "Envs":
        if self.EMBEDDING_SERVICE_TYPE == "openai":
            if self.OPENAI_EMBEDDING_BASE_URL:
                self.EMBEDDING_BASE_URL = self.OPENAI_EMBEDDING_BASE_URL
            else:
                raise ValueError(
                    "OPENAI_EMBEDDING_BASE_URL is required when EMBEDDING_SERVICE_TYPE is openai"
                )
            if self.OPENAI_EMBEDDING_API_KEY:
                self.EMBEDDING_API_KEY = self.OPENAI_EMBEDDING_API_KEY
            else:
                raise ValueError(
                    "OPENAI_EMBEDDING_API_KEY is required when EMBEDDING_SERVICE_TYPE is openai"
                )
        elif self.EMBEDDING_SERVICE_TYPE == "local":
            if self.LOCAL_EMBEDDING_BASE_URL:
                self.EMBEDDING_BASE_URL = self.LOCAL_EMBEDDING_BASE_URL
            else:
                raise ValueError(
                    "LOCAL_EMBEDDING_BASE_URL is required when EMBEDDING_SERVICE_TYPE is local"
                )
            if self.LOCAL_EMBEDDING_API_KEY:
                self.EMBEDDING_API_KEY = self.LOCAL_EMBEDDING_API_KEY
            else:
                raise ValueError(
                    "LOCAL_EMBEDDING_API_KEY is required when EMBEDDING_SERVICE_TYPE is local"
                )
        if self.LLM_SERVICE_TYPE == "openai":
            if self.OPENAI_LLM_BASE_URL:
                self.LLM_BASE_URL = self.OPENAI_LLM_BASE_URL
            else:
                raise ValueError(
                    "OPENAI_LLM_BASE_URL is required when LLM_SERVICE_TYPE is openai"
                )
            if self.OPENAI_LLM_API_KEY:
                self.LLM_API_KEY = self.OPENAI_LLM_API_KEY
            else:
                raise ValueError(
                    "OPENAI_LLM_API_KEY is required when LLM_SERVICE_TYPE is openai"
                )
        elif self.LLM_SERVICE_TYPE == "local":
            if self.LOCAL_LLM_BASE_URL:
                self.LLM_BASE_URL = self.LOCAL_LLM_BASE_URL
            else:
                raise ValueError(
                    "LOCAL_LLM_BASE_URL is required when LLM_SERVICE_TYPE is local"
                )
            if self.LOCAL_LLM_API_KEY:
                self.LLM_API_KEY = self.LOCAL_LLM_API_KEY
            else:
                raise ValueError(
                    "LOCAL_LLM_API_KEY is required when LLM_SERVICE_TYPE is local"
                )
        if self.TRANSLATOR_SERVICE_TYPE == "openai":
            if self.OPENAI_TRANSLATOR_BASE_URL:
                self.TRANSLATOR_BASE_URL = self.OPENAI_TRANSLATOR_BASE_URL
            else:
                raise ValueError(
                    "OPENAI_TRANSLATOR_BASE_URL is required when TRANSLATOR_SERVICE_TYPE is openai"
                )
            if self.OPENAI_TRANSLATOR_API_KEY:
                self.TRANSLATOR_API_KEY = self.OPENAI_TRANSLATOR_API_KEY
            else:
                raise ValueError(
                    "OPENAI_TRANSLATOR_API_KEY is required when TRANSLATOR_SERVICE_TYPE is openai"
                )
            if self.OPENAI_TRANSLATOR_MODEL_NAME:
                self.TRANSLATOR_MODEL_NAME = self.OPENAI_TRANSLATOR_MODEL_NAME
            else:
                raise ValueError(
                    "OPENAI_TRANSLATOR_MODEL_NAME is required when TRANSLATOR_SERVICE_TYPE is openai"
                )
            self.TRANSLATOR_TIME_OUT = self.OPENAI_TRANSLATOR_TIME_OUT
            self.TRANSLATOR_MAX_TOKENS = self.OPENAI_TRANSLATOR_MAX_TOKENS
            self.TRANSLATOR_TEMPERATURE = self.OPENAI_TRANSLATOR_TEMPERATURE
        elif self.TRANSLATOR_SERVICE_TYPE == "local":
            if self.LOCAL_TRANSLATOR_BASE_URL:
                self.TRANSLATOR_BASE_URL = self.LOCAL_TRANSLATOR_BASE_URL
            else:
                raise ValueError(
                    "LOCAL_TRANSLATOR_BASE_URL is required when TRANSLATOR_SERVICE_TYPE is local"
                )
            if self.LOCAL_TRANSLATOR_API_KEY:
                self.TRANSLATOR_API_KEY = self.LOCAL_TRANSLATOR_API_KEY
            else:
                raise ValueError(
                    "LOCAL_TRANSLATOR_API_KEY is required when TRANSLATOR_SERVICE_TYPE is local"
                )
            if self.LOCAL_TRANSLATOR_MODEL_NAME:
                self.TRANSLATOR_MODEL_NAME = self.LOCAL_TRANSLATOR_MODEL_NAME
            else:
                raise ValueError(
                    "LOCAL_TRANSLATOR_MODEL_NAME is required when TRANSLATOR_SERVICE_TYPE is local"
                )
            if self.LOCAL_TRANSLATOR_ENTRY_POINT:
                self.TRANSLATOR_ENTRY_POINT = self.LOCAL_TRANSLATOR_ENTRY_POINT
            else:
                raise ValueError(
                    "LOCAL_TRANSLATOR_ENTRY_POINT is required when TRANSLATOR_SERVICE_TYPE is local"
                )
            self.TRANSLATOR_TIME_OUT = self.LOCAL_TRANSLATOR_TIME_OUT
        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
