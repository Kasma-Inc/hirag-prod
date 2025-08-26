import random
from typing import Callable, List

from hirag_prod._utils import decode_tokens_by_tiktoken, encode_string_by_tiktoken
from hirag_prod.prompt import PROMPTS
from hirag_prod.summarization.base import BaseSummarizer


class TrancatedAggregateSummarizer(BaseSummarizer):
    def __init__(
        self,
        extract_func: Callable,
        llm_model_name: str = "gpt-4o-mini",
        tiktoken_model_name: str = "gpt-4o-mini",
        input_max_tokens: int = 16000,
        output_max_tokens: int = 1000,
    ):
        self.tiktoken_model_name = tiktoken_model_name
        self.input_max_tokens = input_max_tokens
        self.output_max_tokens = output_max_tokens
        self.extract_func = extract_func
        self.llm_model_name = llm_model_name

    async def summarize_entity(
        self,
        entity_name: str,
        descriptions: List[str],
    ) -> str:
        """
        Summarize the entity descriptions.
        Args:
            entity_name: The name of the entity.
            descriptions: The descriptions of the entity.
        Returns:
            The summary of the entity.
        """
        random.shuffle(descriptions)
        sep = "<SEP>"
        descriptions = sep.join(descriptions)
        # Truncate the descriptions to the input_max_tokens
        tokens = encode_string_by_tiktoken(
            descriptions, model_name=self.tiktoken_model_name
        )
        if len(tokens) < self.input_max_tokens:
            descriptions_for_prompt = descriptions
        else:
            descriptions_for_prompt = decode_tokens_by_tiktoken(
                tokens[: self.input_max_tokens], model_name=self.tiktoken_model_name
            ).split(sep)

        summary_prompt_template = PROMPTS["summarize_entity_descriptions_en"]

        context_for_prompt = {
            "entity_name": entity_name,
            "description_list": descriptions_for_prompt,
        }

        use_prompt = summary_prompt_template.format(**context_for_prompt)
        summary = await self.extract_func(
            model=self.llm_model_name,
            prompt=use_prompt,
            max_tokens=self.output_max_tokens,
        )
        return summary
