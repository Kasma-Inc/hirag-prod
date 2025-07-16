from pandas import DataFrame
from typing import Dict, List, Any

class DictParser:
    """Parser for parsing dictionary data into LLM friendly format."""

    def parse_to_string(self, data: dict) -> str:
        """
        Convert dictionary data to a string format suitable for LLM processing.

        Args:
            data (dict): The dictionary data to parse.

        Returns:
            str: The parsed string representation of the dictionary.
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        return "\n".join(f"{key}: {value}" for key, value in data.items())

    def parse_to_table(self, data: dict) -> DataFrame:
        """
        Convert dictionary data to a table format suitable for LLM processing.

        Args:
            data (dict): The dictionary data to parse.

        Returns:
            str: The parsed table representation of the dictionary.
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        df = DataFrame(list(data.items()), columns=['Key', 'Value'])
        return df

    def parse_list_of_dicts(self, data: List[Dict[str, Any]], method: str = "string") -> str:
        """
        Convert a list of dictionaries to a string or table format.

        Args:
            data (List[Dict[str, Any]]): The list of dictionaries to parse.
            method (str): The method of parsing ('string' or 'table').

        Returns:
            str: The parsed representation of the list of dictionaries.
        """
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Input must be a list of dictionaries")

        # transform each dictionary to a df and concatenate them
        if method == "string":
            return "\n".join(self.parse_to_string(item) for item in data)
        elif method == "table":
            return DataFrame(data).to_csv(index=False)

        else:
            raise ValueError("Method must be either 'string' or 'table'")