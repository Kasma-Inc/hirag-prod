from abc import ABC, abstractmethod
from typing import List

from ..schema import Entity, Relation


class BaseGDB(ABC):
    @abstractmethod
    async def upsert_relation(self, relation: Relation):
        raise NotImplementedError

    @abstractmethod
    async def query_one_hop(self, query: str) -> (List[Entity], List[Relation]):
        raise NotImplementedError
