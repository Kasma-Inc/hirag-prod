from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from hirag_prod.schema import Entity, Relation


class BaseGDB(ABC):
    @abstractmethod
    async def upsert_relation(self, relation: Relation):
        raise NotImplementedError

    @abstractmethod
    async def query_one_hop(self, node_id: str) -> Tuple[List[Entity], List[Relation]]:
        raise NotImplementedError

    @abstractmethod
    async def query_node(self, node_id: str) -> Entity:
        raise NotImplementedError

    @abstractmethod
    async def pagerank_top_chunks_with_reset(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    async def health_check(self) -> None:
        return None

    async def clean_up(self) -> None:
        return None

    async def dump(self) -> None:
        return None
