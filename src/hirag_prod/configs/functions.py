from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

from hirag_prod.configs.cloud_storage_config import AWSConfig, OSSConfig
from hirag_prod.configs.document_loader_config import DotsOCRConfig
from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.envs import Envs, InitEnvs
from hirag_prod.configs.hi_rag_config import HiRAGConfig
from hirag_prod.configs.llm_config import LLMConfig
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


def get_embedding_config() -> EmbeddingConfig:
    return get_config_manager().embedding_config


def get_llm_config() -> LLMConfig:
    return get_config_manager().llm_config


def get_reranker_config() -> RerankConfig:
    return get_config_manager().reranker_config


def get_translator_config() -> TranslatorConfig:
    return get_config_manager().translator_config


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
    storage_type: Literal["s3", "oss"],
) -> Union[AWSConfig, OSSConfig]:
    if storage_type == "s3":
        return get_config_manager().aws_config
    else:
        return get_config_manager().oss_config


def get_envs() -> Envs:
    return get_config_manager().envs


def get_shared_variables() -> "SharedVariables":
    return get_config_manager().shared_variables
