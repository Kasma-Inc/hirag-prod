from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Reranker(ABC):

    @abstractmethod
    async def rerank(
        self,
        query: Union[str, List[str]],
        items: List[Dict],
        rerank_with_time=False,
        key: str = "text",
    ) -> List[Dict]:
        pass
