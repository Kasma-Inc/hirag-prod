from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from hirag_prod.schema import Chunk, Entity, Relation


@dataclass
class BaseKG(ABC):
    entity_extract_prompt: str

    @abstractmethod
    async def construct_kg(
        self, chunks: List[Chunk]
    ) -> Tuple[List[Entity], List[Relation]]:
        pass
