import logging
import time
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger("HiRAG")


class ReferenceParser:
    """
    ReferenceParser class for parsing references in text.
    """

    def __init__(self):
        logger.info("🔧 Initializing ReferenceParser...")
        logger.info("📥 Downloading NLTK data (this may take a moment on first run)...")
        start_time = time.perf_counter()

        nltk.download("punkt_tab", quiet=True)

        total_time = time.perf_counter() - start_time
        logger.info(f"✅ NLTK data download completed in {total_time:.3f}s")

    async def parse_references(
        self, text: str, reference_placeholder: str, omit_length: int = 5
    ) -> List[str]:
        """
        Parse references from the given text using NLTK sentence tokenization.
        Returns the sentence containing each reference placeholder, with empty sentences discarded.
        """
        if not text:
            return []

        references = []
        remaining_text = text

        while reference_placeholder in remaining_text:
            # Find the position of the next placeholder
            placeholder_pos = remaining_text.find(reference_placeholder)

            # Extract text up to and including the placeholder
            text_up_to_placeholder = remaining_text[:placeholder_pos]

            # Use NLTK to tokenize sentences in this portion
            if text_up_to_placeholder.strip():
                sentences = sent_tokenize(text_up_to_placeholder)
                if sentences:
                    # Get the last sentence (the one containing the reference)
                    last_sentence = sentences[-1].strip()
                    # Only add if not empty after stripping
                    if last_sentence and len(last_sentence) > omit_length:
                        references.append(last_sentence)

                    # All errors fall back to appending empty string, and handled later
                    else:
                        references.append("")
                else:
                    references.append("")
            else:
                references.append("")

            # Move past this placeholder for next iteration
            remaining_text = remaining_text[
                placeholder_pos + len(reference_placeholder) :
            ]

        return references

    async def fill_placeholders(
        self,
        text: str,
        references: list[list[str]],
        reference_placeholder: str,
        format_prompt: str = "{documentKey}",
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
            if ref != []:
                formatted_refs = [format_prompt.format(documentKey=r) for r in ref]
                replace_text = "".join(formatted_refs)
                text = text.replace(reference_placeholder, replace_text, 1)
            else:
                text = text.replace(reference_placeholder, "", 1)

        return text
