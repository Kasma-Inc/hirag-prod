from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Reranker(ABC):

    @abstractmethod
    async def rerank(
        self,
        query: Union[str, List[str]],
        items: List[Dict],
        key: str = "text",
    ) -> List[Dict]:
        pass
