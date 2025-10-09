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
        min_interval_seconds: Optional[Union[float, str]] = None,
        rate_limit: Optional[Union[int, str]] = None,
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
            self.min_interval_seconds: Optional[float] = None
            self.max_request_number: Optional[int] = None
            self.time_interval_seconds: Optional[int] = None
            if isinstance(min_interval_seconds, float):
                self.min_interval_seconds = min_interval_seconds
            elif min_interval_seconds is not None:
                self.min_interval_seconds_env_name: str = min_interval_seconds
            if (rate_limit is not None) and (time_unit is not None):
                if isinstance(rate_limit, int):
                    self.max_request_number = rate_limit
                else:
                    self.rate_limit_env_name: str = rate_limit
                if time_unit in ["second", "minute", "hour"]:
                    self.time_interval_seconds = self.second_number_dict[time_unit]
                else:
                    self.time_unit_env_name: str = time_unit

            if is_main_process():
                get_shared_variables().rate_limiter_last_call_time_dict[self.name] = (
                    multiprocessing.Value("d", 0.0)
                )
                get_shared_variables().rate_limiter_call_time_queue_dict[
                    self.name
                ] = multiprocessing.Queue()
                get_shared_variables().rate_limiter_wait_lock_dict[
                    self.name
                ] = multiprocessing.Lock()
            self.last_call_time: Synchronized[float] = (
                get_shared_variables().rate_limiter_last_call_time_dict[self.name]
            )
            self.call_time_queue: multiprocessing.Queue[float] = (
                get_shared_variables().rate_limiter_call_time_queue_dict[self.name]
            )
            self.wait_lock: multiprocessing.synchronize.Lock = (
                get_shared_variables().rate_limiter_wait_lock_dict[self.name]
            )

    def limit(
        self,
        name: str,
        min_interval_seconds: Optional[Union[float, str]] = None,
        rate_limit: Optional[Union[int, str]] = None,
        time_unit: Optional[str] = None,
    ):
        def decorator(func: Callable) -> Callable:
            self.rate_limiter_name_set.add(name)
            second_number_dict: Dict[str, int] = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
            }
            min_interval_seconds_list: List[float] = []
            max_request_number_list: List[int] = []
            time_interval_seconds_list: List[int] = []
            if isinstance(min_interval_seconds, float):
                min_interval_seconds_list.append(min_interval_seconds)
            elif min_interval_seconds is not None:
                min_interval_seconds_env_name: str = min_interval_seconds
            if (rate_limit is not None) and (time_unit is not None):
                if isinstance(rate_limit, int):
                    max_request_number_list.append(rate_limit)
                else:
                    rate_limit_env_name: str = rate_limit
                if time_unit in ["second", "minute", "hour"]:
                    time_interval_seconds_list.append(second_number_dict[time_unit])
                else:
                    time_unit_env_name: str = time_unit
            last_call_time_list: List[Synchronized[float]] = []
            call_time_queue_list: List[multiprocessing.Queue[float]] = []
            wait_lock_list: List[multiprocessing.synchronize.Lock] = []

            def initialize():
                if (min_interval_seconds is not None) and (
                    len(min_interval_seconds_list) == 0
                ):
                    min_interval_seconds_list.append(
                        getattr(get_envs(), min_interval_seconds_env_name)
                    )
                if (rate_limit is not None) and (time_unit is not None):
                    if len(max_request_number_list) == 0:
                        max_request_number_list.append(
                            getattr(get_envs(), rate_limit_env_name)
                        )
                    if len(time_interval_seconds_list) == 0:
                        time_interval_seconds_list.append(
                            second_number_dict[getattr(get_envs(), time_unit_env_name)]
                        )
                if len(last_call_time_list) == 0:
                    last_call_time_list.append(
                        get_shared_variables().rate_limiter_last_call_time_dict[name]
                    )
                if len(call_time_queue_list) == 0:
                    call_time_queue_list.append(
                        get_shared_variables().rate_limiter_call_time_queue_dict[name]
                    )
                if len(wait_lock_list) == 0:
                    wait_lock_list.append(
                        get_shared_variables().rate_limiter_wait_lock_dict[name]
                    )

            async def check_rate_limit_async():
                initialize()
                await acquire_multiprocessing_lock_using_thread(wait_lock_list[0])
                try:
                    if len(min_interval_seconds_list) != 0:
                        elapsed = time.time() - last_call_time_list[0].value
                        left_to_wait = min_interval_seconds_list[0] - elapsed
                        if left_to_wait > 0:
                            await asyncio.sleep(left_to_wait)
                    if (len(max_request_number_list) != 0) and (
                        len(time_interval_seconds_list) != 0
                    ):
                        if (
                            call_time_queue_list[0].qsize()
                            >= max_request_number_list[0]
                        ):
                            elapsed = time.time() - call_time_queue_list[0].get()
                            left_to_wait = time_interval_seconds_list[0] - elapsed
                            if left_to_wait > 0:
                                await asyncio.sleep(left_to_wait)
                    call_time: float = time.time()
                    if len(min_interval_seconds_list) != 0:
                        last_call_time_list[0].value = call_time
                    if (len(max_request_number_list) != 0) and (
                        len(time_interval_seconds_list) != 0
                    ):
                        call_time_queue_list[0].put(call_time)
                finally:
                    wait_lock_list[0].release()

            def check_rate_limit_sync():
                initialize()
                wait_lock_list[0].acquire()
                try:
                    if len(min_interval_seconds_list) != 0:
                        elapsed = time.time() - last_call_time_list[0].value
                        left_to_wait = min_interval_seconds_list[0] - elapsed
                        if left_to_wait > 0:
                            time.sleep(left_to_wait)
                    if (len(max_request_number_list) != 0) and (
                        len(time_interval_seconds_list) != 0
                    ):
                        if (
                            call_time_queue_list[0].qsize()
                            >= max_request_number_list[0]
                        ):
                            elapsed = time.time() - call_time_queue_list[0].get()
                            left_to_wait = time_interval_seconds_list[0] - elapsed
                            if left_to_wait > 0:
                                time.sleep(left_to_wait)
                    call_time: float = time.time()
                    if len(min_interval_seconds_list) != 0:
                        last_call_time_list[0].value = call_time
                    if (len(max_request_number_list) != 0) and (
                        len(time_interval_seconds_list) != 0
                    ):
                        call_time_queue_list[0].put(call_time)
                finally:
                    wait_lock_list[0].release()

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    await check_rate_limit_async()
                    return await func(*args, **kwargs)

                return wrapper
            elif inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    await check_rate_limit_async()
                    async for item in func(*args, **kwargs):
                        yield item

                return wrapper
            else:

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    check_rate_limit_sync()
                    return func(*args, **kwargs)

                return wrapper

        return decorator

    def initialize(self):
        if (self.min_interval_seconds is None) and hasattr(
            self, "min_interval_seconds_env_name"
        ):
            self.min_interval_seconds = getattr(
                get_envs(), self.min_interval_seconds_env_name
            )
        if hasattr(self, "rate_limit_env_name") and hasattr(self, "time_unit_env_name"):
            if self.max_request_number is None:
                self.max_request_number = getattr(get_envs(), self.rate_limit_env_name)
            if self.time_interval_seconds is None:
                self.time_interval_seconds = self.second_number_dict[
                    getattr(get_envs(), self.time_unit_env_name)
                ]

    async def check_rate_limit_async(self):
        self.initialize()
        await acquire_multiprocessing_lock_using_thread(self.wait_lock)
        try:
            if self.min_interval_seconds is not None:
                elapsed = time.time() - self.last_call_time.value
                left_to_wait = self.min_interval_seconds - elapsed
                if left_to_wait > 0:
                    await asyncio.sleep(left_to_wait)
            if (self.max_request_number is not None) and (
                self.time_interval_seconds is not None
            ):
                if self.call_time_queue.qsize() >= self.max_request_number:
                    elapsed = time.time() - self.call_time_queue.get()
                    left_to_wait = self.time_interval_seconds - elapsed
                    if left_to_wait > 0:
                        await asyncio.sleep(left_to_wait)
            call_time: float = time.time()
            if self.min_interval_seconds is not None:
                self.last_call_time.value = call_time
            if (self.max_request_number is not None) and (
                self.time_interval_seconds is not None
            ):
                self.call_time_queue.put(call_time)
        finally:
            self.wait_lock.release()

    def check_rate_limit_sync(self):
        self.initialize()
        self.wait_lock.acquire()
        try:
            if self.min_interval_seconds is not None:
                elapsed = time.time() - self.last_call_time.value
                left_to_wait = self.min_interval_seconds - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
            if (self.max_request_number is not None) and (
                self.time_interval_seconds is not None
            ):
                if self.call_time_queue.qsize() >= self.max_request_number:
                    elapsed = time.time() - self.call_time_queue.get()
                    left_to_wait = self.time_interval_seconds - elapsed
                    if left_to_wait > 0:
                        time.sleep(left_to_wait)
            call_time: float = time.time()
            if self.min_interval_seconds is not None:
                self.last_call_time.value = call_time
            if (self.max_request_number is not None) and (
                self.time_interval_seconds is not None
            ):
                self.call_time_queue.put(call_time)
        finally:
            self.wait_lock.release()

    async def run_function_async(self, func: Callable, *args, **kwargs) -> Any:
        await self.check_rate_limit_async()
        return await func(*args, **kwargs)

    async def run_async_generator(
        self, func: Callable, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        await self.check_rate_limit_async()
        async for item in func(*args, **kwargs):
            yield item

    def run_function_sync(self, func: Callable, *args, **kwargs) -> Any:
        self.check_rate_limit_sync()
        return func(*args, **kwargs)
