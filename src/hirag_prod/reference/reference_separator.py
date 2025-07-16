class ReferenceSeparator:
    """
    A class to separate references in a text based on a specific placeholder.
    """

    def __init__(self, place_holder_begin="", place_holder_end="", separator_type="double"):
        """
        Initialize the ReferenceSeparator with the placeholder strings and separator type.

        Args:
            place_holder_begin (str): The beginning placeholder string.
            place_holder_end (str): The ending placeholder string.
            separator_type (str): The type of separator to use
                                    "double" for wrapping the text that references with placeholders,
                                    "single" for ending the text that references with a placeholder.

        Raises:
            ValueError: If the separator type is not valid.
        """
        self.separator_type = separator_type
        if separator_type == "double":
            self.place_holder_begin = place_holder_begin
            self.place_holder_end = place_holder_end
        elif separator_type == "single":
            self.place_holder_end = place_holder_end
        else:
            raise ValueError("Invalid separator type. Use 'double' or 'single'.")

    def separate(self, text: str) -> list[str]:
        """
        Separate the text into parts based on the placeholder.

        Args:
            text (str): The text to be separated.
        
        Returns:
            list[str]: A list of text parts separated by the placeholder, only containing the parts between the placeholders.
        """

        if self.separator_type == "double":
            parts = []
            start = 0

            while True:
                start = text.find(self.place_holder_begin, start)
                if start == -1:
                    break
                start += len(self.place_holder_begin)

                end = text.find(self.place_holder_end, start)
                if end == -1:
                    break
                
                parts.append(text[start:end].strip())

                start = end + len(self.place_holder_end)

            return parts
        elif self.separator_type == "single":
            # separate the text by period, newline, tab, question mark, exclamation mark, or semicolon
            potential_separators = [".", "\n", "\t", "?", "!", ";"]
            parts = []
            start = 0

            # for each place holder end, find the sentence before it
            while True:
                end = text.find(self.place_holder_end, start)
                if end == -1:
                    break
                
                # Find the last potential separator before the end
                last_separator = max(text.rfind(sep, start, end) for sep in potential_separators)

                if last_separator == -1 or last_separator < start:
                    parts.append(text[start:end].strip())
                else:
                    parts.append(text[last_separator + 1:end].strip())

                start = end + len(self.place_holder_end)
            
            return parts

    def fill_placeholders(self, text: str, references: list[str], format_prompt: str = "{document_key}") -> str:
        """
        Fill the placeholders in the text with the provided references.

        Args:
            text (str): The text containing placeholders.
            references (list[str]): The references to fill in the placeholders.
        
        Returns:
            str: The text with placeholders filled with references.
        """
        
        if self.separator_type == "double":
            # Remove all placeholder begin from the text
            # if " <PH> " in text, remove a space
            text = text.replace(" " + self.place_holder_begin + " ", " ")
            text = text.replace(self.place_holder_begin, "")

            # Substitute the placeholder end with the references
            for ref in references:
                if ref != "":
                    text = text.replace(self.place_holder_end, format_prompt.format(document_key=ref), 1)
                else:
                    text = text.replace(self.place_holder_end, "", 1)

            # Remove any remaining placeholder end
            text = text.replace(self.place_holder_end, "")

            return text

        elif self.separator_type == "single":

            for ref in references:
                if ref != "":
                    text = text.replace(self.place_holder_end, format_prompt.format(document_key=ref), 1)
                else:
                    text = text.replace(self.place_holder_end, "", 1)

            return text