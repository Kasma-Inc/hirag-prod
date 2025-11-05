import re
from typing import Any, Dict

from pydantic import ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class ProviderConfig(BaseSettings):
    base_url: str = Field(..., description="API base URL")
    api_key: SecretStr = Field(..., description="API key")


class ProviderKeyConfigs(BaseSettings):
    provider_keys: Dict[str, ProviderConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def load_dynamic_providers(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            values = {}

        pattern = re.compile(
            r"^(?P<service>[a-z0-9]+)_(?P<suffix>base_url|api_key)$", re.IGNORECASE
        )
        raw: Dict[str, Dict[str, str]] = {}

        for key, value in values.items():
            if not isinstance(key, str):
                continue
            if m := pattern.match(key):
                service = m.group("service").lower()
                suffix = m.group("suffix").lower()
                raw.setdefault(service, {})[suffix] = value

        # initialize provider_keys
        values["provider_keys"] = values.get("provider_keys", {})

        for service, config_dict in raw.items():
            if "base_url" in config_dict and "api_key" in config_dict:
                try:
                    provider = ProviderConfig(
                        base_url=config_dict["base_url"], api_key=config_dict["api_key"]
                    )
                    values["provider_keys"][service] = provider
                    values[service] = provider
                except Exception as e:
                    raise ValueError(f"Invalid config for {service}: {e}")

        return values
