from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InitEnvs(BaseSettings):
    model_config = SettingsConfigDict(
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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        env_ignore_empty=True,
        title="Configurations",
    )

    # TODO(tatiana): remove after token usage is included in API response
    ENABLE_TOKEN_COUNT: bool = False

    # TODO(tatiana): consolidate the timeout logic of models
