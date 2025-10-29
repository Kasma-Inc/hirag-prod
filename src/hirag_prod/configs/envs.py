from typing import Literal, Optional, Union, get_args, get_origin

from pydantic import ConfigDict, Field, PostgresDsn, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings


class InitEnvs(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    EMBEDDING_DIMENSION: int = Field(
        group="Embedding", description="The dimension of the embedding."
    )
    USE_HALF_VEC: bool = Field(
        group="Embedding",
        description="Whether to use half vector for embedding.",
        default=True,
    )
    HIRAG_QUERY_TIMEOUT: int = Field(
        default=100, description="The timeout in seconds for rag queries."
    )


class Envs(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow", env_ignore_empty=True
    )

    CONSTRUCT_GRAPH: bool = Field(
        group="Index configuration",
        description="Whether to construct graph for indexing.",
        default=False,
    )

    HI_RAG_LANGUAGE: str = "en"
    ENABLE_TOKEN_COUNT: bool = False

    POSTGRES_URL: PostgresDsn = Field(
        group="PostgresDB",
        description="Postgres URL",
        examples=["postgres://user:password@postgresdb:5432/database"],
    )
    POSTGRES_TABLE_NAME: str = Field(
        group="PostgresDB",
        description="Postgres table name for storing knowledge base catalog",
        default="KnowledgeBaseCatalog",
    )
    POSTGRES_SCHEMA: str = Field(
        group="PostgresDB", description="Postgres schema", default="public"
    )

    # TODO(tatiana): deprecated now, remove later
    REDIS_URL: str = Field(
        deprecated=True,
        group="Redis",
        description="Redis URL. Deprecated.",
        default="redis://redis:6379/2",
    )

    LLM_SERVICE_TYPE: Literal["openai", "local"] = "openai"
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    OPENAI_LLM_BASE_URL: Optional[str] = None
    OPENAI_LLM_API_KEY: Optional[str] = None
    LOCAL_LLM_BASE_URL: Optional[str] = None
    LOCAL_LLM_API_KEY: Optional[str] = None

    # TODO(tatiana): consolidate the timeout logic
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

    RERANKER_TYPE: Literal["api", "local"] = "local"

    # Local reranker settings
    LOCAL_RERANKER_MODEL_BASE_URL: Optional[str] = None
    LOCAL_RERANKER_MODEL_NAME: str = "Qwen3-Reranker-8B"
    LOCAL_RERANKER_MODEL_ENTRY_POINT: str = "/rerank"
    LOCAL_RERANKER_MODEL_AUTHORIZATION: Optional[str] = None

    LLM_RATE_LIMIT: int = 60
    LLM_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    LLM_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1
    RERANKER_RATE_LIMIT: int = 6000
    RERANKER_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    RERANKER_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1
    TRANSLATOR_RATE_LIMIT: int = 60
    TRANSLATOR_RATE_LIMIT_TIME_UNIT: Literal["second", "minute", "hour"] = "minute"
    TRANSLATOR_RATE_LIMIT_MIN_INTERVAL_SECONDS: float = 0.1

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "Envs":
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
            self.TRANSLATOR_MAX_TOKENS = self.LOCAL_TRANSLATOR_MAX_TOKENS
            self.TRANSLATOR_TEMPERATURE = self.LOCAL_TRANSLATOR_TEMPERATURE
        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def to_dotenv_example(cls, file_path: str = ".env.example", only_no_default=False):
        """Generate a template .env file that can be used to configure this settings model class.
        If only_no_default is True, only fields with no default value will be included.
        """
        group = None
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                "# Generated by configs."
                f'{cls.__name__}.to_dotenv_example("{file_path}", {only_no_default})\n'
            )
            for env_name, field in cls.model_fields.items():
                if only_no_default and field.default is not PydanticUndefined:
                    continue
                current_group = (
                    field.json_schema_extra.get("group")
                    if field.json_schema_extra
                    else None
                )
                if current_group != group:
                    group = current_group
                    if current_group is not None:
                        f.write(f"\n##### {current_group.upper()}\n")
                    else:
                        f.write("\n")
                f.write(cls._to_dotenv_doc(env_name, field))
                default = (
                    field.default
                    if field.default is not None
                    and field.default is not PydanticUndefined
                    else ""
                )
                f.write(f"{env_name}={default}\n")

    @classmethod
    def _to_dotenv_doc(cls, name: str, field: FieldInfo):
        field = cls.model_fields.get(name)
        args_str = f"Required {field.annotation.__name__}"
        origin = get_origin(field.annotation)
        if origin is Literal:
            args = [str(arg) for arg in get_args(field.annotation)]
            args_str = f"Choices: [{', '.join(args)}]"
        elif origin is Union:
            args = get_args(field.annotation)
            if args[-1] is type(None) and len(args) == 2:
                args_str = f"Optional {args[0].__name__}"
        examples_str = ""
        if field.examples:
            examples_str = f"# Examples: {', '.join(field.examples)}\n"
        description_str = field.description if field.description else ""
        return f"# {args_str}. {description_str}\n{examples_str}"
