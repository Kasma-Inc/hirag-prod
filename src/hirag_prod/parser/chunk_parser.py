# This is a parser designed to handle lists of items
from .base_parser import BaseParser

class ChunkParser(BaseParser):
    """
    A parser that formats a list of items into a string representation.
    This is useful for displaying lists in a more readable format.
    """

    def parse(self, data: list[dict], keep_attr: list[str] = []) -> str:
        """
        Parse the given list of items into a formatted string.

        Args:
            data (list): The list of items to be parsed.
            keep_attr (list): The attributes to keep from each item.

        Returns:
            str: A formatted string representation of the list.
        """

        if not data:
            return "No items found."

        if not keep_attr:
            keep_attr = list(data[0].keys())
        
        formatted_items = []
        for item in data:
            item_parts = []
            for attr in keep_attr:
                if attr in item:
                    item_parts.append(f"{attr}: {item[attr]}")
            formatted_items.append(", ".join(item_parts))
        return "\n".join(formatted_items)