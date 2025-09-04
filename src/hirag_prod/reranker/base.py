from typing import List, Dict
from abc import ABC, abstractmethod

class Reranker(ABC):
    
    @abstractmethod
    async def rerank(self, query: str, items: List[Dict], topn: int) -> List[Dict]:
        pass