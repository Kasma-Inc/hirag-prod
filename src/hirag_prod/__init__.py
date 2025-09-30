from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hirag_prod.hirag import HiRAG

__all__ = ["HiRAG", "server"]


def __getattr__(name):
    if name == "HiRAG":
        from hirag_prod.hirag import HiRAG

        return HiRAG
    if name == "server":
        from hirag_prod import server

        return server
    raise AttributeError(name)
