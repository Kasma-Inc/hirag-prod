from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hirag import HiRAG

__all__ = ["HiRAG"]


def __getattr__(name):
    if name == "HiRAG":
        from .hirag import HiRAG

        return HiRAG
    raise AttributeError(name)
