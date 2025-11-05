from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from hirag_prod.configs.cloud_storage_config import S3Config
from hirag_prod.configs.document_loader_config import DotsOCRConfig
from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.envs import Envs, InitEnvs
from hirag_prod.configs.hi_rag_config import HiRAGConfig
from hirag_prod.configs.llm_config import LLMConfig
from hirag_prod.configs.ocr_config import OcrConfig
from hirag_prod.configs.postgres_db_config import PostgresDBConfig
from hirag_prod.configs.provider_key_config import ProviderKeyConfigs
from hirag_prod.configs.reranker_config import RerankConfig
from hirag_prod.configs.translator_config import TranslatorConfig

if TYPE_CHECKING:
    from hirag_prod.configs.config_manager import ConfigManager
    from hirag_prod.configs.shared_variables import SharedVariables

INIT_CONFIG = InitEnvs()


def initialize_config_manager(
    cli_options_dict: Optional[Dict] = None,
    config_dict: Optional[Dict] = None,
    shared_variable_dict: Optional[Dict] = None,
) -> None:
    from hirag_prod.configs.config_manager import ConfigManager

    ConfigManager(cli_options_dict, config_dict, shared_variable_dict)


def get_config_manager() -> "ConfigManager":
    from hirag_prod.configs.config_manager import ConfigManager

    return ConfigManager()


def is_main_process() -> bool:
    return get_config_manager().is_main_process


def get_hi_rag_config() -> HiRAGConfig:
    return get_config_manager().hi_rag_config


def get_provider_key_configs() -> ProviderKeyConfigs:
    return get_config_manager().provider_key_configs


def get_init_config() -> InitEnvs:
    return INIT_CONFIG


def get_document_converter_config(
    converter_type: Literal["dots_ocr"],
) -> Optional[DotsOCRConfig]:
    if converter_type == "dots_ocr":
        return get_config_manager().dots_ocr_config
    else:
        return None


def get_cloud_storage_config(
    _: Literal["s3", "oss"],
) -> S3Config:
    return get_config_manager().s3_config


def get_postgres_config() -> "PostgresDBConfig":
    return get_config_manager().postgres_config


def get_envs() -> Envs:
    return get_config_manager().envs


def get_shared_variables() -> "SharedVariables":
    return get_config_manager().shared_variables


def get_llm_configs() -> Dict[str, LLMConfig]:
    return get_config_manager().llm_configs


def get_embedding_configs() -> Dict[str, EmbeddingConfig]:
    return get_config_manager().embedding_configs


def get_reranker_configs() -> Dict[str, RerankConfig]:
    return get_config_manager().reranker_configs


def get_translator_configs() -> Dict[str, TranslatorConfig]:
    return get_config_manager().translator_configs


def get_ocr_configs() -> Dict[str, OcrConfig]:
    return get_config_manager().ocr_configs


def get_llm_api_provider(model_id: str) -> str:
    """Get llm provider by model id."""
    return get_llm_configs()[model_id].db_table["api_provider"]


def get_embedding_api_provider(model_id: str) -> str:
    """Get embedding provider by model id."""
    return get_embedding_configs()[model_id].db_table["api_provider"]


def get_kb_model_configs() -> Dict[str, Any]:
    return get_config_manager().kb_model_configs
