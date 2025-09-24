import re
import string
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import ScoreAlignment
from sklearn.metrics.pairwise import cosine_similarity

from hirag_prod.configs.functions import get_llm_config
from hirag_prod.cross_language_search.types import ProcessSearchResponse
from hirag_prod.resources.functions import (
    get_chat_service,
    get_chinese_convertor,
    get_embedding_service,
    tokenize_sentence,
)


def classify_search(
    search_list: List[str], embedding_np_array: np.ndarray
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    search_keyword_list: List[str] = []
    keyword_embedding_list: List[np.ndarray] = []
    keyword_embedding_np_array: np.ndarray = np.empty((0, embedding_np_array.shape[1]))
    search_sentence_list: List[str] = []
    sentence_embedding_list: List[np.ndarray] = []
    sentence_embedding_np_array: np.ndarray = np.empty((0, embedding_np_array.shape[1]))
    for i, search in enumerate(search_list):
        search_normalized: str = get_chinese_convertor("hk2s").convert(
            re.sub(f"[{re.escape(string.punctuation)}]", "", search).lower()
        )
        token_list, _, _ = tokenize_sentence(search_normalized)
        if len(token_list) == 1:
            search_keyword_list.append(search_normalized)
            keyword_embedding_list.append(embedding_np_array[i : i + 1])
        else:
            search_sentence_list.append(search_normalized)
            sentence_embedding_list.append(embedding_np_array[i : i + 1])
    return (
        search_keyword_list,
        np.concatenate([keyword_embedding_np_array] + keyword_embedding_list, axis=0),
        search_sentence_list,
        np.concatenate([sentence_embedding_np_array] + sentence_embedding_list, axis=0),
    )


def has_traditional_chinese(text: str) -> bool:
    return get_chinese_convertor("hk2s").convert(text) != text


def normalize_text(text: str) -> str:
    return get_chinese_convertor("hk2s").convert(
        re.sub(f"[{re.escape(string.punctuation)}]", "", text).strip().lower()
    )


def normalize_tokenize_text(text: str) -> Tuple[List[str], List[int], List[int]]:
    normalized_text: str = normalize_text(text)
    return tokenize_sentence(normalized_text)


async def create_embeddings_batch(
    str_list_dict: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    if len(str_list_dict) == 0:
        return {}
    item_list: List[Tuple[str, List[str]]] = list(str_list_dict.items())
    input_str_list: List[str] = []
    for item in item_list:
        input_str_list.extend(item[1])
    embedding_np_array: np.ndarray = await get_embedding_service().create_embeddings(
        input_str_list
    )
    embedding_np_array_dict: Dict[str, np.ndarray] = {}
    current_index: int = 0
    for item in item_list:
        embedding_np_array_dict[item[0]] = embedding_np_array[
            current_index : current_index + len(item[1])
        ]
        current_index += len(item[1])
    return embedding_np_array_dict


async def validate_similarity(
    str_embedding_np_array: np.ndarray,
    search_embedding_np_array: np.ndarray,
    matched_list_dict_batch: List[
        Dict[str, Optional[List[Optional[Union[int, Tuple[int, int]]]]]]
    ],
    threshold: float = 0.8,
) -> None:
    cosine_similarity_list: List[float] = (
        cosine_similarity(
            str_embedding_np_array,
            search_embedding_np_array,
        )
        .max(axis=1)
        .tolist()
    )

    current_index: int = 0
    for matched_list_dict in matched_list_dict_batch:
        if matched_list_dict["original"] is not None:
            for i in range(len(matched_list_dict["original"])):
                if cosine_similarity_list[current_index] <= threshold:
                    matched_list_dict["original"][i] = None
                current_index += 1
        if matched_list_dict["translation"] is not None:
            for i in range(len(matched_list_dict["translation"])):
                if cosine_similarity_list[current_index] <= threshold:
                    matched_list_dict["translation"][i] = None
                current_index += 1


async def get_synonyms_and_validate_and_translate(
    search: str,
) -> Tuple[List[str], np.ndarray, bool, List[str], np.ndarray]:
    synonym_set: Set[str] = set()
    synonym_set.add(search)

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
    synonym_list: List[str] = list(synonym_set)
    embedding_np_array: np.ndarray = await get_embedding_service().create_embeddings(
        synonym_list + process_search_response.translation_list + [search]
    )
    if len(synonym_list) == 1:
        return (
            [search],
            embedding_np_array[:1],
            process_search_response.is_english,
            process_search_response.translation_list,
            embedding_np_array[1:-1],
        )
    cosine_similarity_list: List[float] = (
        cosine_similarity(
            embedding_np_array[: len(synonym_list)],
            embedding_np_array[-1:],
        )
        .flatten()
        .tolist()
    )
    synonym_tuple_list: List[Tuple[str, np.ndarray, float]] = [
        (synonym_list[i], embedding_np_array[i : i + 1], similarity)
        for i, similarity in enumerate(cosine_similarity_list)
        if similarity > 0.75
    ]
    synonym_tuple_list.sort(key=lambda x: x[2], reverse=True)

    return (
        [synonym_tuple[0] for synonym_tuple in synonym_tuple_list],
        np.concatenate(
            [synonym_tuple[1] for synonym_tuple in synonym_tuple_list], axis=0
        ),
        process_search_response.is_english,
        process_search_response.translation_list,
        embedding_np_array[len(synonym_list) : -1],
    )


def find_keyword_matches(
    word_list: List[str],
    search_list: List[str],
    prev_matched_index_list: Optional[List[int]] = None,
) -> Optional[List[int]]:
    matched_index_set: Set[int] = set()
    if prev_matched_index_list is not None:
        matched_index_set.update(prev_matched_index_list)
    for j, word in enumerate(word_list):
        for search in search_list:
            if (fuzz.ratio(word, search) > 90) or (
                (len(word) >= len(search)) and (fuzz.partial_ratio(word, search) > 90)
            ):
                matched_index_set.add(j)
                break

    if len(matched_index_set) > 0:
        return sorted(matched_index_set)
    else:
        return None


async def search_by_search_keyword_list(
    processed_chunk_list: List[Dict[str, Any]],
    search_list_original: List[str],
    search_list: List[str],
) -> Tuple[List[str], List[Dict[str, Optional[List[Optional[int]]]]]]:
    matched_index_list_dict_batch: List[Dict[str, Optional[List[Optional[int]]]]] = [
        {"original": None, "translation": None} for _ in processed_chunk_list
    ]

    for i, processed_chunk in enumerate(processed_chunk_list):
        matched_index_list_dict_batch[i]["original"] = find_keyword_matches(
            processed_chunk["original_token_list"], search_list_original
        )
        if len(search_list) > 0:
            if matched_index_list_dict_batch[i]["original"] is not None:
                matched_index_list_dict_batch[i]["original"] = find_keyword_matches(
                    processed_chunk["original_token_list"],
                    search_list,
                    matched_index_list_dict_batch[i]["original"],
                )
            else:
                matched_index_list_dict_batch[i]["translation"] = find_keyword_matches(
                    processed_chunk["translation_token_list"], search_list
                )

    word_list_to_embed: List[str] = []
    for processed_chunk, matched_index_list_dict in zip(
        processed_chunk_list, matched_index_list_dict_batch
    ):
        if matched_index_list_dict["original"] is not None:
            for matched_index in matched_index_list_dict["original"]:
                word_list_to_embed.append(
                    processed_chunk["original_token_list"][matched_index]
                )
        if matched_index_list_dict["translation"] is not None:
            for matched_index in matched_index_list_dict["translation"]:
                word_list_to_embed.append(
                    processed_chunk["translation_token_list"][matched_index]
                )

    return word_list_to_embed, matched_index_list_dict_batch


def find_sentence_matches(
    text_normalized: str, search_list: List[str]
) -> Optional[List[Tuple[int, int]]]:
    fuzzy_match_list: Optional[List[Optional[Tuple[int, int]]]] = []
    queue: List[Tuple[str, int]] = [(text_normalized, 0)]
    while len(queue) > 0:
        text, start_index = queue.pop(0)
        for search in search_list:
            if fuzz.ratio(text, search) > 80:
                fuzzy_match_list.append((start_index, start_index + len(text)))
                break
            elif len(text) >= len(search):
                match_result: Optional[ScoreAlignment] = fuzz.partial_ratio_alignment(
                    text, search, score_cutoff=80
                )
                if match_result is not None:
                    fuzzy_match_list.append(
                        (
                            start_index + match_result.src_start,
                            start_index + match_result.src_end,
                        )
                    )
                    if match_result.src_start > 0:
                        queue.append(
                            (
                                text[: match_result.src_start],
                                start_index,
                            )
                        )
                    if match_result.src_end < len(text):
                        queue.append(
                            (
                                text[match_result.src_end :],
                                start_index + match_result.src_end,
                            )
                        )
    if len(fuzzy_match_list) > 0:
        return fuzzy_match_list
    else:
        return None


async def precise_search_by_search_sentence_list(
    processed_chunk_list: List[Dict[str, Any]],
    search_list_original: List[str],
    search_list: List[str],
) -> Tuple[List[str], List[Dict[str, Optional[List[Optional[Tuple[int, int]]]]]]]:
    fuzzy_match_list_dict_batch: List[
        Dict[str, Optional[List[Optional[Tuple[int, int]]]]]
    ] = [{"original": None, "translation": None} for _ in processed_chunk_list]

    for i, processed_chunk in enumerate(processed_chunk_list):
        fuzzy_match_list_dict_batch[i]["original"] = find_sentence_matches(
            processed_chunk["original_normalized"], search_list_original
        )
        if len(search_list) > 0:
            if fuzzy_match_list_dict_batch[i]["original"] is not None:
                fuzzy_match_list_dict_batch[i]["original"] += find_sentence_matches(
                    processed_chunk["original_normalized"], search_list
                )
            else:
                fuzzy_match_list_dict_batch[i]["translation"] = find_sentence_matches(
                    processed_chunk["translation_normalized"], search_list
                )

    sentence_list_to_embed: List[str] = []
    for i, fuzzy_match_list_dict in enumerate(fuzzy_match_list_dict_batch):
        if fuzzy_match_list_dict["original"] is not None:
            for fuzzy_match in fuzzy_match_list_dict["original"]:
                sentence_list_to_embed.append(
                    processed_chunk_list[i]["original_normalized"][
                        fuzzy_match[0] : fuzzy_match[1]
                    ]
                )
        if fuzzy_match_list_dict["translation"] is not None:
            for fuzzy_match in fuzzy_match_list_dict["translation"]:
                sentence_list_to_embed.append(
                    processed_chunk_list[i]["translation_normalized"][
                        fuzzy_match[0] : fuzzy_match[1]
                    ]
                )

    return sentence_list_to_embed, fuzzy_match_list_dict_batch


async def embedding_search_by_search_sentence_list(
    processed_chunk_list: List[Dict[str, Any]],
) -> Dict[int, float]:
    embedding_similar_chunk_info_dict: Dict[int, float] = {}
    for i, processed_chunk in enumerate(processed_chunk_list):
        if processed_chunk["search_sentence_cosine_distance"] < 0.4:
            embedding_similar_chunk_info_dict[i] = processed_chunk[
                "search_sentence_cosine_distance"
            ]

    return embedding_similar_chunk_info_dict


def get_token_index(
    token_start_index_list: List[int], token_end_index_list: List[int], char_index: int
) -> Tuple[int, bool]:
    left_index: int = 0
    right_index: int = len(token_start_index_list)
    while left_index < right_index:
        mid_index: int = (left_index + right_index) // 2
        if (char_index >= token_start_index_list[mid_index]) and (
            char_index < token_end_index_list[mid_index]
        ):
            return mid_index, True
        elif token_start_index_list[mid_index] > char_index:
            right_index = mid_index
        else:
            left_index = mid_index + 1
    return left_index, False


def bold_matched_text(
    processed_chunk: Dict[str, Any],
    matched_keyword_index_list_dict: Dict[str, Optional[List[Optional[int]]]],
    matched_sentence_index_list_dict: Dict[
        str, Optional[List[Optional[Tuple[int, int]]]]
    ],
    text_type: Literal["original", "translation"],
) -> None:
    if matched_keyword_index_list_dict[text_type] is not None:
        for matched_keyword_index in matched_keyword_index_list_dict[text_type]:
            if matched_keyword_index is not None:
                processed_chunk[f"{text_type}_token_list"][
                    matched_keyword_index
                ] = f"<mark>{processed_chunk[f"{text_type}_token_list"][matched_keyword_index]}</mark>"

    if matched_sentence_index_list_dict[text_type] is not None:
        for matched_sentence_index in matched_sentence_index_list_dict[text_type]:
            if matched_sentence_index is not None:
                start, _ = get_token_index(
                    processed_chunk[f"{text_type}_token_start_index_list"],
                    processed_chunk[f"{text_type}_token_end_index_list"],
                    matched_sentence_index[0],
                )
                end, in_token = get_token_index(
                    processed_chunk[f"{text_type}_token_start_index_list"],
                    processed_chunk[f"{text_type}_token_end_index_list"],
                    matched_sentence_index[1] - 1,
                )
                if in_token:
                    end += 1
                for j in range(start, end):
                    if "<mark>" not in processed_chunk[f"{text_type}_token_list"][j]:
                        processed_chunk[f"{text_type}_token_list"][
                            j
                        ] = f"<mark>{processed_chunk[f"{text_type}_token_list"][j]}</mark>"

    for j in range(len(processed_chunk[f"{text_type}_token_list"]) - 1):
        if processed_chunk[f"{text_type}_token_list"][j].endswith(
            "</mark>"
        ) and processed_chunk[f"{text_type}_token_list"][j + 1].startswith("<mark>"):
            processed_chunk[f"{text_type}_token_list"][j] = processed_chunk[
                f"{text_type}_token_list"
            ][j][:-7]
            processed_chunk[f"{text_type}_token_list"][j + 1] = processed_chunk[
                f"{text_type}_token_list"
            ][j + 1][6:]


def simplify_search_result(
    processed_chunk: Dict[str, Any],
    text_type: Literal["original", "translation"],
    context_size: int = 3,
) -> None:
    matched_index_tuple_list: List[Tuple[int, int]] = []
    for j, word in enumerate(processed_chunk[f"{text_type}_token_list"]):
        if word.startswith("<mark>") and word.endswith("</mark>"):
            matched_index_tuple_list.append((j, j + 1))
        elif word.startswith("<mark>"):
            matched_index_tuple_list.append(
                (j, len(processed_chunk[f"{text_type}_token_list"]))
            )
        elif word.endswith("</mark>"):
            matched_index_tuple_list[-1] = (matched_index_tuple_list[-1][0], j + 1)

    output: str = ""
    last_match_end: int = -1
    last_end: int = -1
    for matched_index_tuple in matched_index_tuple_list:
        match_start: int = processed_chunk[f"{text_type}_token_start_index_list"][
            matched_index_tuple[0]
        ]
        match_end: int = processed_chunk[f"{text_type}_token_end_index_list"][
            matched_index_tuple[1] - 1
        ]
        start: int = processed_chunk[f"{text_type}_token_start_index_list"][
            max(0, matched_index_tuple[0] - context_size)
        ]
        end: int = processed_chunk[f"{text_type}_token_end_index_list"][
            min(
                len(processed_chunk[f"{text_type}_token_list"]),
                matched_index_tuple[1] + context_size,
            )
            - 1
        ]
        if (start != 0) and (start > last_end):
            output += "..."
        elif start < last_match_end:
            output = output[: last_match_end - last_end]
            start = last_match_end
        elif start < last_end:
            output = output[: start - last_end]
        output += (
            processed_chunk[f"{text_type}_normalized"][start:match_start]
            + f"<mark>{processed_chunk[f"{text_type}_normalized"][match_start:match_end]}</mark>"
            + processed_chunk[f"{text_type}_normalized"][match_end:end]
        )
        last_match_end = match_end
        last_end = end
    if len(output) > 0:
        if last_end < len(processed_chunk[f"{text_type}_normalized"]):
            output += "..."
        processed_chunk[f"{text_type}_search_result"] = (output, 1)
    else:
        processed_chunk[f"{text_type}_search_result"] = None


def build_search_result(
    processed_chunk_list: List[Dict[str, Any]],
    matched_keyword_index_list_dict_batch: List[
        Dict[str, Optional[List[Optional[int]]]]
    ],
    matched_sentence_index_list_dict_batch: List[
        Dict[str, Optional[List[Optional[Tuple[int, int]]]]]
    ],
    embedding_similar_chunk_info_dict: Dict[int, float],
    context_size: int = 3,
) -> None:
    for i, processed_chunk in enumerate(processed_chunk_list):
        bold_matched_text(
            processed_chunk,
            matched_keyword_index_list_dict_batch[i],
            matched_sentence_index_list_dict_batch[i],
            "original",
        )
        bold_matched_text(
            processed_chunk,
            matched_keyword_index_list_dict_batch[i],
            matched_sentence_index_list_dict_batch[i],
            "translation",
        )

        simplify_search_result(processed_chunk, "original", context_size)
        simplify_search_result(processed_chunk, "translation", context_size)

        if (processed_chunk["original_search_result"] is None) and (
            processed_chunk["translation_search_result"] is None
        ):
            if i in embedding_similar_chunk_info_dict:
                processed_chunk["embedding_search_result"] = (
                    processed_chunk["original_normalized"],
                    embedding_similar_chunk_info_dict[i],
                )
