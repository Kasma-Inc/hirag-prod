from dataclasses import dataclass

from ..schema import Entity


@dataclass
class Relation:
    source: Entity
    target: Entity
    properties: dict
