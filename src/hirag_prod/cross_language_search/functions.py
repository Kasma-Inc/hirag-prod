import re
import string
from typing import List, Set, Tuple

import numpy as np

from hirag_prod.configs.functions import get_llm_config
from hirag_prod.cross_language_search.types import ProcessSearchResponse
from hirag_prod.resources.functions import (
    get_chat_service,
    get_chinese_convertor,
    get_embedding_service,
    tokenize_sentence,
)


def has_traditional_chinese(text: str) -> bool:
    return get_chinese_convertor("hk2s").convert(text) != text


def normalize_text(text: str) -> str:
    return get_chinese_convertor("hk2s").convert(
        re.sub(f"[{re.escape(string.punctuation)}]", "", text).strip().lower()
    )


def normalize_tokenize_text(text: str) -> Tuple[str, List[str], List[int], List[int]]:
    normalized_text: str = normalize_text(text)
    token_list, token_start_index_list, token_end_index_list = tokenize_sentence(
        normalized_text
    )
    return normalized_text, token_list, token_start_index_list, token_end_index_list


async def get_synonyms_and_validate_and_translate(
    search: str,
) -> Tuple[List[str], np.ndarray, bool, List[str], np.ndarray]:
    synonym_set: Set[str] = set()

    process_search_response: ProcessSearchResponse = await get_chat_service().complete(
        prompt=f"""Please complete the following two tasks according to the search keyword or sentence **{search}**, then output the final result according to the format provided below:
Task 1: Please provide some synonyms for the search keyword or sentence **{search}**. The synonyms need to be **in the same language with the search**. Please give at least 5 different synonyms and output them as a JSON list.
Task 2: Please identify if the search only includes English, return a JSON value of **true** or **false**.
Task 3: Please translate **{search}** into English ** only if it is not in English**, return **an empty JSON list** if the search is in English. Please translate as briefly as possible. Please give at least 6 different possible translations and output them as a JSON list.
The final result need to be **a JSON object with the following structure**:
{{
  "synonym_list": ["synonym1", "synonym2", "synonym3", "synonym4", "synonym5", ...],
  "is_english": true or false,
  "translation_list": ["translation1", "translation2", "translation3", "translation4", "translation5", "translation6", ...]
}}""",
        model=get_llm_config().model_name,
        max_tokens=get_llm_config().max_tokens,
        response_format=ProcessSearchResponse,
        timeout=get_llm_config().timeout,
    )

    synonym_set.update(process_search_response.synonym_list)
    try:
        synonym_set.remove(search)
    except KeyError:
        pass
    synonym_list: List[str] = list(synonym_set)
    embedding_np_array: np.ndarray = await get_embedding_service().create_embeddings(
        synonym_list + process_search_response.translation_list + [search]
    )
    if len(synonym_list) == 0:
        return (
            [search],
            embedding_np_array[-1:],
            process_search_response.is_english,
            process_search_response.translation_list,
            embedding_np_array[:-1],
        )
    synonym_tuple_list: List[Tuple[str, np.ndarray]] = [
        (synonym_list[i], embedding_np_array[i : i + 1])
        for i in range(len(synonym_list))
    ]
    synonym_tuple_list.sort(key=lambda x: x[0], reverse=True)
    synonym_tuple_list.insert(0, (search, embedding_np_array[-1:]))

    return (
        [synonym_tuple[0] for synonym_tuple in synonym_tuple_list],
        np.concatenate(
            [synonym_tuple[1] for synonym_tuple in synonym_tuple_list], axis=0
        ),
        process_search_response.is_english,
        process_search_response.translation_list,
        embedding_np_array[len(synonym_list) - 1 : -1],
    )
