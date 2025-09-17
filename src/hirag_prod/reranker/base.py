from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Reranker(ABC):

    @abstractmethod
    async def rerank(
        self, query: Union[str, List[str]], items: List[Dict], topn: int
    ) -> List[Dict]:
        pass
