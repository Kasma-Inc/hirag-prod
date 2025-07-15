class ReferenceSeparator:
    """
    A class to separate references in a text based on a specific placeholder.
    """

    def __init__(self, place_holder_begin, place_holder_end):
        self.place_holder_begin = place_holder_begin
        self.place_holder_end = place_holder_end

    def separate(self, text: str) -> list[str]:
        """
        Separate the text into parts based on the placeholder.

        Args:
            text (str): The text to be separated.
        
        Returns:
            list[str]: A list of text parts separated by the placeholder, only containing the parts between the placeholders.
        """
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

    def fill_placeholders(self, text: str, references: list[str], format_prompt: str = "{document_key}") -> str:
        """
        Fill the placeholders in the text with the provided references.

        Args:
            text (str): The text containing placeholders.
            references (list[str]): The references to fill in the placeholders.
        
        Returns:
            str: The text with placeholders filled with references.
        """
        
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