from dataclasses import dataclass


@dataclass
class Relation:
    source: str
    target: str
    properties: dict
