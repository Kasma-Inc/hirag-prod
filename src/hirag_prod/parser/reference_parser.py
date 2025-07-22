from typing import List


class ReferenceParser:
    """
    ReferenceParser class for parsing references in text.
    """

    def __init__(self):
        pass

    async def parse_references(
        self, text: str, reference_placeholder: str
    ) -> List[str]:
        """
        Parse references from the given text.
        """
        if not text:
            return []

        # separate the text by period, newline, tab, question mark, exclamation mark, or semicolon
        potential_separators = [".", "\n", "\t", "?", "!", ";"]
        parts = []
        start = 0

        # for each place holder end, find the sentence before it
        while True:
            end = text.find(reference_placeholder, start)
            if end == -1:
                break

            # Find the last potential separator before the end
            last_separator = max(
                text.rfind(sep, start, end) for sep in potential_separators
            )

            if last_separator == -1 or last_separator < start:
                parts.append(text[start:end].strip())
            else:
                parts.append(text[last_separator + 1 : end].strip())

            start = end + len(reference_placeholder)

        return parts

    async def fill_placeholders(
        self,
        text: str,
        references: list[str],
        reference_placeholder: str,
        format_prompt: str = "{document_key}",
    ) -> str:
        """
        Fill the placeholders in the text with the provided references.

        Args:
            text (str): The text containing placeholders.
            references (list[str]): The references to fill in the placeholders.
            format_prompt (str): The format string for the references.

        Returns:
            str: The text with placeholders filled with references.
        """

        for ref in references:
            if ref != "":
                text = text.replace(
                    reference_placeholder, format_prompt.format(document_key=ref), 1
                )
            else:
                text = text.replace(reference_placeholder, "", 1)

        return text
