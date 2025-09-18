from typing import Dict, Literal, Optional, Union

from hirag_prod.configs.cloud_storage_config import AWSConfig, OSSConfig
from hirag_prod.configs.config_manager import ConfigManager
from hirag_prod.configs.document_loader_config import DoclingCloudConfig, DotsOCRConfig
from hirag_prod.configs.embedding_config import EmbeddingConfig
from hirag_prod.configs.envs import Envs, InitEnvs
from hirag_prod.configs.hi_rag_config import HiRAGConfig
from hirag_prod.configs.llm_config import LLMConfig
from hirag_prod.configs.reranker_config import RerankConfig

INIT_CONFIG = InitEnvs()


def initialize_config_manager(
    cli_options_dict: Optional[Dict] = None, config_dict: Optional[Dict] = None
) -> None:
    ConfigManager(cli_options_dict, config_dict)


def get_config_manager() -> ConfigManager:
    return ConfigManager()


def get_hi_rag_config() -> HiRAGConfig:
    return ConfigManager().hi_rag_config


def get_embedding_config() -> EmbeddingConfig:
    return ConfigManager().embedding_config


def get_llm_config() -> LLMConfig:
    return ConfigManager().llm_config


def get_reranker_config() -> RerankConfig:
    return ConfigManager().reranker_config


def get_init_config() -> InitEnvs:
    return INIT_CONFIG


def get_document_converter_config(
    converter_type: Literal["dots_ocr", "docling_cloud"],
) -> Union[DotsOCRConfig, DoclingCloudConfig]:
    if converter_type == "dots_ocr":
        return ConfigManager().dots_ocr_config
    else:
        return ConfigManager().docling_cloud_config


def get_cloud_storage_config(
    storage_type: Literal["s3", "oss"],
) -> Union[AWSConfig, OSSConfig]:
    if storage_type == "s3":
        return ConfigManager().aws_config
    else:
        return ConfigManager().oss_config


def get_envs() -> Envs:
    return ConfigManager().envs
