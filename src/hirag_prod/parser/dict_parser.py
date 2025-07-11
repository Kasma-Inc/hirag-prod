# try to parse dictionary data into a token efficient XML-like format.
from dicttoxml2 import dicttoxml
from xml.dom.minidom import parseString
from .base_parser import BaseParser

class DictParser(BaseParser):
    """
    A parser that converts a dictionary into an XML string representation.
    """

    def parse(self, data: dict) -> str:
        """
        Parse a dictionary into an XML string.

        Args:
            data (dict): The dictionary to be parsed.

        Returns:
            str: The XML representation of the dictionary.
        """

        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        # Convert the dictionary to an XML string
        xml_string = dicttoxml(data, custom_root='root', attr_type=False, root=False)

        # Parse the XML string to a pretty format
        xml_string = parseString(xml_string).toprettyxml()

        return xml_string.strip()