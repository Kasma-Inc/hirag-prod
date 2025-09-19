import asyncio
import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from hirag_prod._utils import log_error_info

if TYPE_CHECKING:
    from hirag_prod.resources.resource_manager import ResourceManager


def timing_logger(operation_name: str):
    """Decorator to log execution time of operations."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(
                    f"⏱ [TIMING: {os.getpid()}] {operation_name} completed in {duration:.3f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_error_info(
                    logging.ERROR,
                    f"⏱ [TIMING: {os.getpid()}] {operation_name} failed after {duration:.3f}s",
                    e,
                    raise_error=True,
                )

        return wrapper

    return decorator


async def initialize_resource_manager(resource_dict: Optional[Dict] = None) -> None:
    from hirag_prod.resources.resource_manager import ResourceManager

    resource_manager: ResourceManager = ResourceManager(resource_dict)
    await resource_manager.initialize()


def get_resource_manager() -> "ResourceManager":
    from hirag_prod.resources.resource_manager import ResourceManager

    return ResourceManager()


def get_db_engine():
    return get_resource_manager().get_db_engine()


def get_db_session_maker():
    return get_resource_manager().get_session_maker()


def get_db_session():
    session_maker = get_resource_manager().get_session_maker()
    return session_maker()


def get_redis(**kwargs: Any):
    return get_resource_manager().get_redis_client(**kwargs)


def get_chinese_convertor(convertor_type: str = "s2hk"):
    return get_resource_manager().get_chinese_convertor(convertor_type)


def get_sentence_tokenizer():
    return get_resource_manager().get_sentence_tokenizer()


def tokenize_sentence(sentence: str) -> Tuple[List[str], List[int], List[int]]:
    if len(sentence.strip()) == 0:
        return [], [], []
    else:
        result_list: List[str] = []
        token_start_index_list: List[int] = []
        token_end_index_list: List[int] = []
        current_text_index: int = 0
        current_result_list_index: int = 0
        finish: bool = False
        while not finish:
            if current_text_index + 510 < len(sentence):
                result_list.extend(
                    get_sentence_tokenizer()(
                        sentence[current_text_index : current_text_index + 510]
                    )[:-1]
                )
            else:
                result_list.extend(
                    get_sentence_tokenizer()(sentence[current_text_index:])
                )
                finish = True
            for token in result_list[current_result_list_index:]:
                token_start_index_list.append(
                    current_text_index + sentence[current_text_index:].find(token[0])
                )
                for char in token:
                    current_text_index += sentence[current_text_index:].find(char) + 1
                token_end_index_list.append(current_text_index)
            current_result_list_index = len(result_list)
        return result_list, token_start_index_list, token_end_index_list


def get_translator():
    return get_resource_manager().get_translator()


def get_qwen_translator():
    return get_resource_manager().get_qwen_translator()


def get_reranker():
    return get_resource_manager().get_reranker()


def get_chat_service():
    return get_resource_manager().get_chat_service()


def get_embedding_service():
    return get_resource_manager().get_embedding_service()
