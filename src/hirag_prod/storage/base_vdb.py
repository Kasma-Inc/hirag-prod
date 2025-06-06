from abc import ABC, abstractmethod
from typing import Callable, List, Literal


class BaseVDB(ABC):
    embedding_func: Callable

    @abstractmethod
    async def upsert_text(
        self,
        text_to_embed: str,
        properties: dict,
        mode: Literal["append", "overwrite"] = "append",
    ):
        raise NotImplementedError

    @abstractmethod
    async def query(self, query: str) -> List[dict]:
        raise NotImplementedError
