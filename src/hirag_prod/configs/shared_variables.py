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

        if is_main_process:
            from hirag_prod import _llm
            from hirag_prod.rate_limiter import RATE_LIMITER_NAME_SET
            from hirag_prod.reranker import api_reranker, local_reranker
            from hirag_prod.translator import qwen_translator

            RATE_LIMITER_NAME_SET.update(_llm.rate_limiter.rate_limiter_name_set)
            RATE_LIMITER_NAME_SET.update(
                api_reranker.rate_limiter.rate_limiter_name_set
            )
            RATE_LIMITER_NAME_SET.update(
                local_reranker.rate_limiter.rate_limiter_name_set
            )
            RATE_LIMITER_NAME_SET.update(
                qwen_translator.rate_limiter.rate_limiter_name_set
            )

            for rate_limiter_name in RATE_LIMITER_NAME_SET:
                if rate_limiter_name not in self.rate_limiter_last_call_time_dict:
                    self.rate_limiter_last_call_time_dict[rate_limiter_name] = (
                        multiprocessing.Value("d", 0.0)
                    )
                if rate_limiter_name not in self.rate_limiter_call_time_queue_dict:
                    self.rate_limiter_call_time_queue_dict[rate_limiter_name] = (
                        multiprocessing.Queue()
                    )
                if rate_limiter_name not in self.rate_limiter_wait_lock_dict:
                    self.rate_limiter_wait_lock_dict[rate_limiter_name] = (
                        multiprocessing.Lock()
                    )
                if rate_limiter_name not in self.input_token_count_dict:
                    self.input_token_count_dict[rate_limiter_name] = (
                        multiprocessing.Value("i", 0)
                    )
                if rate_limiter_name not in self.output_token_count_dict:
                    self.output_token_count_dict[rate_limiter_name] = (
                        multiprocessing.Value("i", 0)
                    )

    def to_dict(self):
        return {
            k if not k.startswith("_") else k[1:]: v for k, v in self.__dict__.items()
        }
