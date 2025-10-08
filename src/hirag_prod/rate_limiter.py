import asyncio
import functools
import inspect
import multiprocessing.synchronize
import threading
import time
from asyncio import AbstractEventLoop
from multiprocessing.sharedctypes import Synchronized
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Union

from hirag_prod.configs.functions import get_envs, get_shared_variables, is_main_process

RATE_LIMITER_NAME_SET: Set[str] = set()


async def acquire_multiprocessing_lock_using_thread(
    lock: multiprocessing.synchronize.Lock,
):
    loop: AbstractEventLoop = asyncio.get_event_loop()
    future = loop.create_future()

    def acquire_and_set_result():
        lock.acquire()
        loop.call_soon_threadsafe(future.set_result, None)

    thread: threading.Thread = threading.Thread(
        target=acquire_and_set_result, daemon=True
    )
    thread.start()
    await future
    del thread


class RateLimiter:

    def __init__(
        self,
        name: Optional[str] = None,
        rate_limit: Optional[Union[int, float, str]] = None,
        time_unit: Optional[str] = None,
    ):
        if name is None:
            self.rate_limiter_name_set: Set[str] = set()
        else:
            self.name: str = name

            self.second_number_dict: Dict[str, int] = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
            }
            self.min_interval: Optional[float] = None
            if isinstance(rate_limit, Union[int, float]) and (
                time_unit in ["second", "minute", "hour"]
            ):
                self.min_interval = self.second_number_dict[time_unit] / rate_limit
            else:
                self.rate_limit_env_name: str = rate_limit
                self.time_unit_env_name: str = time_unit

            if is_main_process():
                get_shared_variables().rate_limiter_last_call_time_dict[self.name] = (
                    multiprocessing.Value("d", 0.0)
                )
                get_shared_variables().rate_limiter_wait_lock_dict[
                    self.name
                ] = multiprocessing.Lock()
            self.last_call_time: Synchronized[float] = (
                get_shared_variables().rate_limiter_last_call_time_dict[self.name]
            )
            self.wait_lock: multiprocessing.synchronize.Lock = (
                get_shared_variables().rate_limiter_wait_lock_dict[self.name]
            )

    def limit(self, name: str, rate_limit: Union[int, float, str], time_unit: str):
        def decorator(func: Callable) -> Callable:
            self.rate_limiter_name_set.add(name)
            second_number_dict: Dict[str, int] = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
            }
            min_interval_list: List[float] = []
            if isinstance(rate_limit, Union[int, float]) and (
                time_unit in ["second", "minute", "hour"]
            ):
                min_interval_list.append(second_number_dict[time_unit] / rate_limit)
            else:
                rate_limit_env_name: str = rate_limit
                time_unit_env_name: str = time_unit
            last_call_time_list: List[Synchronized[float]] = []
            wait_lock_list: List[multiprocessing.synchronize.Lock] = []

            def initialize():
                if len(min_interval_list) == 0:
                    min_interval_list.append(
                        second_number_dict[getattr(get_envs(), time_unit_env_name)]
                        / getattr(get_envs(), rate_limit_env_name)
                    )
                if len(last_call_time_list) == 0:
                    last_call_time_list.append(
                        get_shared_variables().rate_limiter_last_call_time_dict[name]
                    )
                if len(wait_lock_list) == 0:
                    wait_lock_list.append(
                        get_shared_variables().rate_limiter_wait_lock_dict[name]
                    )

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    initialize()
                    await acquire_multiprocessing_lock_using_thread(wait_lock_list[0])
                    try:
                        elapsed = time.time() - last_call_time_list[0].value
                        left_to_wait = min_interval_list[0] - elapsed
                        if left_to_wait > 0:
                            await asyncio.sleep(left_to_wait)
                        last_call_time_list[0].value = time.time()
                    finally:
                        wait_lock_list[0].release()
                    return await func(*args, **kwargs)

                return wrapper
            elif inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    initialize()
                    await acquire_multiprocessing_lock_using_thread(wait_lock_list[0])
                    try:
                        elapsed = time.time() - last_call_time_list[0].value
                        left_to_wait = min_interval_list[0] - elapsed
                        if left_to_wait > 0:
                            await asyncio.sleep(left_to_wait)
                        last_call_time_list[0].value = time.time()
                    finally:
                        wait_lock_list[0].release()
                    async for item in func(*args, **kwargs):
                        yield item

                return wrapper
            else:

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    initialize()
                    wait_lock_list[0].acquire()
                    try:
                        elapsed = time.time() - last_call_time_list[0].value
                        left_to_wait = min_interval_list[0] - elapsed
                        if left_to_wait > 0:
                            time.sleep(left_to_wait)
                        last_call_time_list[0].value = time.time()
                    finally:
                        wait_lock_list[0].release()
                    return func(*args, **kwargs)

                return wrapper

        return decorator

    def initialize(self):
        if self.min_interval is None:
            self.min_interval = self.second_number_dict[
                getattr(get_envs(), self.time_unit_env_name)
            ] / getattr(get_envs(), self.rate_limit_env_name)

    async def run_function_async(self, func: Callable, *args, **kwargs) -> Any:
        self.initialize()
        await acquire_multiprocessing_lock_using_thread(self.wait_lock)
        try:
            elapsed = time.time() - self.last_call_time.value
            left_to_wait = self.min_interval - elapsed
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            self.last_call_time.value = time.time()
        finally:
            self.wait_lock.release()
        return await func(*args, **kwargs)

    async def run_async_generator(
        self, func: Callable, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        self.initialize()
        await acquire_multiprocessing_lock_using_thread(self.wait_lock)
        try:
            elapsed = time.time() - self.last_call_time.value
            left_to_wait = self.min_interval - elapsed
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            self.last_call_time.value = time.time()
        finally:
            self.wait_lock.release()
        async for item in func(*args, **kwargs):
            yield item

    def run_function_sync(self, func: Callable, *args, **kwargs) -> Any:
        self.initialize()
        self.wait_lock.acquire()
        try:
            elapsed = time.time() - self.last_call_time.value
            left_to_wait = self.min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            self.last_call_time.value = time.time()
        finally:
            self.wait_lock.release()
        return func(*args, **kwargs)
