__all__ = ["HiRAG", "server"]


def __getattr__(name):
    if name == "HiRAG":
        from .hirag import HiRAG

        return HiRAG
    if name == "server":
        from . import server

        return server
    raise AttributeError(name)
