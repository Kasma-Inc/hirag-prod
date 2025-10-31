from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseSettings):
    """S3 configuration"""

    region: str
    access_key_id: str
    secret_access_key: SecretStr
    bucket_name: str
    endpoint: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="AWS_",  # for compatibility with common practices
        alias_generator=lambda x: f"ofnil_stage_{x}".upper(),  # for integration with ofnil
        populate_by_name=True,
        extra="ignore",
    )
