# This is a base class for parsers in the HiRAG system.
from abc import ABC, abstractmethod

class BaseParser(ABC):
    """
    Base class for parsers in the HiRAG system.
    This class defines the interface for parsers that handle the RAG retrieved context or chat history.
    """
    
    @abstractmethod
    def parse(self, data) -> str:
        """
        Parse the given data.

        Args:
            data: The data to be parsed.

        Returns:
            Parsed data string.
        """
        pass