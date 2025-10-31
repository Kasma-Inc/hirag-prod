from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresDBConfig(BaseSettings):
    """PostgresDB configuration"""

    model_config = SettingsConfigDict(
        alias_generator=lambda x: f"postgres_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    url: PostgresDsn = Field(
        group="PostgresDB",
        description="Postgres URL",
        examples=["postgres://user:password@postgresdb:5432/database"],
    )
    table_name: str = Field(
        "KnowledgeBaseCatalog",
        description="Postgres table name for storing knowledge base catalog",
    )
    table_schema: str = Field(
        "public", description="Postgres schema", alias="POSTGRES_SCHEMA"
    )

    @computed_field
    def postgres_url_async(self) -> str:
        dsn = PostgresDsn(self.url)
        return str(self.url).replace(f"{dsn.scheme}://", "postgresql+asyncpg://", 1)

    @computed_field
    def postgres_url_sync(self) -> str:
        dsn = PostgresDsn(self.url)
        return str(self.url).replace(f"{dsn.scheme}://", "postgresql://", 1)
