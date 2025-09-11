import asyncio
import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

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


def get_translator():
    return get_resource_manager().get_translator()


def get_chat_service():
    return get_resource_manager().get_chat_service()


def get_embedding_service():
    return get_resource_manager().get_embedding_service()
