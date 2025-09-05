from abc import ABC, abstractmethod
from typing import Dict, List


class Reranker(ABC):

    @abstractmethod
    async def rerank(self, query: str, items: List[Dict], topn: int) -> List[Dict]:
        pass
