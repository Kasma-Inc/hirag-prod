import multiprocessing.synchronize
from multiprocessing.sharedctypes import Synchronized
from typing import Dict


class SharedVariables:
    def __init__(self, is_main_process: bool = True, **kwargs) -> None:
        self.rate_limiter_last_call_time_dict: Dict[str, Synchronized[float]] = (
            kwargs.get("rate_limiter_last_call_time_dict", {})
        )
        self.rate_limiter_call_time_queue_dict: Dict[
            str, multiprocessing.Queue[float]
        ] = kwargs.get("rate_limiter_call_time_queue_dict", {})
        self.rate_limiter_wait_lock_dict: Dict[
            str, multiprocessing.synchronize.Lock
        ] = kwargs.get("rate_limiter_wait_lock_dict", {})
        self.input_token_count_dict: Dict[str, Synchronized[int]] = kwargs.get(
            "input_token_count_dict", {}
        )
        self.output_token_count_dict: Dict[str, Synchronized[int]] = kwargs.get(
            "output_token_count_dict", {}
        )

    def to_dict(self):
        return {
            k if not k.startswith("_") else k[1:]: v for k, v in self.__dict__.items()
        }
