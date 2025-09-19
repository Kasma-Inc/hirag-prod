import threading
from typing import Dict, List, Optional

from hirag_prod.configs.cloud_storage_config import AWSConfig, OSSConfig
from hirag_prod.configs.document_loader_config import DotsOCRConfig
from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.envs import Envs
from hirag_prod.configs.hi_rag_config import HiRAGConfig
from hirag_prod.configs.llm_config import LLMConfig
from hirag_prod.configs.qwen_translator_config import QwenTranslatorConfig
from hirag_prod.configs.reranker_config import RerankConfig


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        cli_options_dict: Optional[Dict] = None,
        config_dict: Optional[Dict] = None,
    ) -> None:
        if getattr(self, "_created", False):
            return

        self.debug: bool = cli_options_dict["debug"]
        self.envs: Envs = Envs(**config_dict if config_dict else {})
        self.hi_rag_config = HiRAGConfig(**self.envs.model_dump())
        self.embedding_config: EmbeddingConfig = EmbeddingConfig(
            **self.envs.model_dump()
        )
        self.llm_config: LLMConfig = LLMConfig(**self.envs.model_dump())
        self.reranker_config: RerankConfig = RerankConfig(**self.envs.model_dump())
        self._qwen_translator_config: Optional[QwenTranslatorConfig] = None
        self._dots_ocr_config: Optional[DotsOCRConfig] = None
        self._aws_config: Optional[AWSConfig] = None
        self._oss_config: Optional[OSSConfig] = None

        self.supported_languages: List[str] = ["en", "cn-s", "cn-t"]
        self.language: str = (
            config_dict["language"]
            if (config_dict and "language" in config_dict)
            else self.envs.HI_RAG_LANGUAGE
        )
        if self.language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language {self.language}. Supported languages are {self.supported_languages}"
            )

        self.postgres_url_env: Optional[str] = (
            self.envs.POSTGRES_URL_NO_SSL_DEV
            if self.envs.ENV == "dev"
            else self.envs.POSTGRES_URL_NO_SSL
        )
        self.postgres_url_async: Optional[str] = self.postgres_url_env
        self.postgres_url_sync: Optional[str] = self.postgres_url_env
        # Replace postgres:// with postgresql+asyncpg:// for async connections
        if self.postgres_url_env.startswith("postgres://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgres://", "postgresql://", 1
            )
        elif self.postgres_url_env.startswith("postgresql://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif self.postgres_url_env.startswith("postgres+asyncpg://"):
            self.postgres_url_async = self.postgres_url_env.replace(
                "postgres+asyncpg://", "postgresql+asyncpg://", 1
            )
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgres+asyncpg://", "postgresql://", 1
            )
        elif self.postgres_url_env.startswith("postgresql+asyncpg://"):
            self.postgres_url_sync = self.postgres_url_env.replace(
                "postgresql+asyncpg://", "postgresql://", 1
            )

        self._created: bool = True

    @property
    def qwen_translator_config(self) -> QwenTranslatorConfig:
        """Getter for qwen_translator_config"""
        if not self._qwen_translator_config:
            self._qwen_translator_config = QwenTranslatorConfig(
                **self.envs.model_dump()
            )
        return self._qwen_translator_config

    @property
    def dots_ocr_config(self) -> DotsOCRConfig:
        """Getter for dots_ocr_config"""
        if not self._dots_ocr_config:
            self._dots_ocr_config = DotsOCRConfig(**self.envs.model_dump())
        return self._dots_ocr_config

    @property
    def aws_config(self) -> AWSConfig:
        """Getter for aws_config"""
        if not self._aws_config:
            self._aws_config = AWSConfig(**self.envs.model_dump())
        return self._aws_config

    @property
    def oss_config(self) -> OSSConfig:
        """Getter for oss_config"""
        if not self._oss_config:
            self._oss_config = OSSConfig(**self.envs.model_dump())
        return self._oss_config
