import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from nltk.corpus import wordnet
from rapidfuzz import fuzz
from rapidfuzz.distance import ScoreAlignment
from sklearn.metrics.pairwise import cosine_similarity

from hirag_prod.resources.functions import get_embedding_service


async def validate_similarity(
    str_list_to_embed: List[str],
    search_list: List[str],
    matched_list_batch: List[Optional[List[Optional[Any]]]],
    threshold: float = 0.9,
) -> None:
    embedding_np_array: np.ndarray = await get_embedding_service().create_embeddings(
        str_list_to_embed + search_list
    )
    cosine_similarity_list: List[float] = (
        cosine_similarity(
            embedding_np_array[: -len(search_list)],
            embedding_np_array[-len(search_list) :],
        )
        .max(axis=1)
        .tolist()
    )

    current_index: int = 0
    for matched_index_list in matched_list_batch:
        if matched_index_list is not None:
            for i in range(len(matched_index_list)):
                if cosine_similarity_list[current_index] <= threshold:
                    matched_index_list[i] = None
                current_index += 1


async def get_synonyms_and_validate(
    word_list: List[str], original_word: str, translated: bool = False
) -> List[str]:
    synonym_set: Set[str] = set()

    for word in word_list:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym_set.add(lemma.name())

    synonym_set.difference_update(word_list)
    if len(synonym_set) == 0:
        return word_list
    synonym_list: List[str] = list(synonym_set)
    embedding_np_array: np.ndarray = await get_embedding_service().create_embeddings(
        synonym_list + word_list + [original_word]
    )
    cosine_similarity_list: List[float] = (
        cosine_similarity(
            embedding_np_array[: -len(word_list) - 1],
            embedding_np_array[-len(word_list) - 1 : -1],
        )
        .max(axis=1)
        .tolist()
    )
    if not translated:
        return [
            synonym_list[i]
            for i, similarity in enumerate(cosine_similarity_list)
            if similarity > 0.8
        ] + word_list
    else:
        original_word_cosine_similarity_list: List[float] = (
            cosine_similarity(embedding_np_array[:-1], embedding_np_array[-1:])
            .flatten()
            .tolist()
        )
        synonym_tuple_list: List[Tuple[str, float]] = [
            (synonym_list[i], original_word_cosine_similarity_list[i])
            for i, similarity in enumerate(cosine_similarity_list)
            if similarity > 0.75
        ]
        synonym_tuple_list.extend(
            [
                (word, original_word_cosine_similarity_list[i - len(word_list)])
                for i, word in enumerate(word_list)
            ]
        )
        synonym_tuple_list.sort(key=lambda x: x[1], reverse=True)

        result_synonym_list: List[str] = []
        if len(synonym_tuple_list) > 0:
            max_original_word_cosine_similarity: float = synonym_tuple_list[0][1]
            for synonym_tuple in synonym_tuple_list:
                if synonym_tuple[1] - max_original_word_cosine_similarity < -0.1:
                    break
                result_synonym_list.append(synonym_tuple[0])
        return result_synonym_list


async def search_by_search_keyword_list(
    chunk_list: List[str], search_list: List[str], context_size: int = 3
) -> Tuple[List[Optional[str]], List[Optional[List[Optional[int]]]]]:
    matched_index_list_batch: List[Optional[List[Optional[int]]]] = [
        None for _ in chunk_list
    ]
    split_chunk_batch: List[List[str]] = [
        re.sub(r"[^a-zA-Z0-9\s]", "", chunk).strip().split() for chunk in chunk_list
    ]
    output_list_batch: List[Optional[str]] = [None for _ in chunk_list]

    for i, split_chunk in enumerate(split_chunk_batch):
        matched_index_set: Set[int] = set()

        for j, word in enumerate(split_chunk):
            for search in search_list:
                if (fuzz.ratio(word.lower(), search.lower()) > 90) or (
                    (len(word) >= len(search))
                    and (fuzz.partial_ratio(word.lower(), search.lower()) > 90)
                ):
                    matched_index_set.add(j)
                    break

        if len(matched_index_set) == 0:
            continue

        # Sort by original position
        matched_index_list_batch[i] = sorted(matched_index_set)

    word_list_to_embed: List[str] = [
        split_chunk[matched_index]
        for split_chunk, matched_index_list in zip(
            split_chunk_batch, matched_index_list_batch
        )
        if (matched_index_list is not None)
        for matched_index in matched_index_list
    ]
    if len(word_list_to_embed) == 0:
        return output_list_batch, matched_index_list_batch
    await validate_similarity(word_list_to_embed, search_list, matched_index_list_batch)

    # Build final output
    for i, matched_index_list in enumerate(matched_index_list_batch):
        if (matched_index_list is None) or (len(matched_index_list) == 0):
            continue
        else:
            output_list: List[str] = []
            last_end: int = -1
            for index in matched_index_list:
                if index is not None:
                    start: int = max(0, index - context_size)
                    end: int = min(len(split_chunk_batch[i]), index + context_size + 1)
                    if (start != 0) and (start > last_end):
                        output_list.append("...")
                    elif start < last_end:
                        output_list = output_list[: start - last_end]
                    split_chunk_batch[i][index] = f"**{split_chunk_batch[i][index]}**"
                    output_list.extend(split_chunk_batch[i][start:end])
                    last_end = end
            if len(output_list) > 0:
                if last_end < len(split_chunk_batch[i]):
                    output_list.append("...")
                output_list_batch[i] = " ".join(output_list)

    return output_list_batch, matched_index_list_batch


async def search_by_search_sentence_list(
    prev_matched_index_list_batch: List[Optional[List[Optional[int]]]],
    chunk_list: List[str],
    search_list: List[str],
    chunk_embedding_list: List[List[float]],
    context_size: int = 3,
) -> List[Optional[Tuple[str, float]]]:
    chunk_batch: List[str] = [
        re.sub(r"[^a-zA-Z0-9\s]", "", chunk).strip() for chunk in chunk_list
    ]
    fuzzy_match_list_batch: List[Optional[List[Optional[Tuple[int, int]]]]] = [
        None for _ in chunk_list
    ]
    embedding_similar_chunk_info_dict: Dict[int, float] = {}
    output_list_batch: List[Optional[Tuple[str, float]]] = [None for _ in chunk_list]

    for i, chunk in enumerate(chunk_batch):
        for search in search_list:
            match_result: ScoreAlignment = fuzz.partial_ratio_alignment(
                chunk.lower(), search.lower()
            )
            if match_result.score > 80:
                if fuzzy_match_list_batch[i] is None:
                    fuzzy_match_list_batch[i] = []
                fuzzy_match_list_batch[i].append(
                    (match_result.src_start, match_result.src_end)
                )

    sentence_list_to_embed: List[str] = [
        chunk_batch[i][fuzzy_match[0] : fuzzy_match[1]]
        for i, fuzzy_match_list in enumerate(fuzzy_match_list_batch)
        if (fuzzy_match_list is not None)
        for fuzzy_match in fuzzy_match_list
    ]
    if len(sentence_list_to_embed) > 0:
        await validate_similarity(
            sentence_list_to_embed, search_list, fuzzy_match_list_batch
        )

    if len(search_list) > 0:
        search_list_embedding_np_array: np.ndarray = (
            await get_embedding_service().create_embeddings(search_list)
        )
        cosine_similarity_list: List[float] = (
            cosine_similarity(chunk_embedding_list, search_list_embedding_np_array)
            .max(axis=1)
            .tolist()
        )
        for i, similarity in enumerate(cosine_similarity_list):
            if similarity > 0.6:
                embedding_similar_chunk_info_dict[i] = similarity

    for i, chunk in enumerate(chunk_batch):
        split_chunk: List[str] = chunk.split()

        if prev_matched_index_list_batch[i] is not None:
            for prev_matched_index in prev_matched_index_list_batch[i]:
                if prev_matched_index is not None:
                    split_chunk[prev_matched_index] = (
                        f"**{split_chunk[prev_matched_index]}**"
                    )

        if fuzzy_match_list_batch[i] is not None:
            for fuzzy_match in fuzzy_match_list_batch[i]:
                if fuzzy_match is not None:
                    start: int = len(chunk[: fuzzy_match[0]].strip().split())
                    end: int = len(chunk[: fuzzy_match[1]].strip().split())
                    for j in range(start, end):
                        if (not split_chunk[j].startswith("**")) and (
                            not split_chunk[j].endswith("**")
                        ):
                            split_chunk[j] = f"**{split_chunk[j]}**"

        for j in range(len(split_chunk) - 1):
            if split_chunk[j].endswith("**") and split_chunk[j + 1].startswith("**"):
                split_chunk[j] = split_chunk[j][:-2]
                split_chunk[j + 1] = split_chunk[j + 1][2:]

        matched_index_tuple_list: List[Tuple[int, int]] = []
        for j, word in enumerate(split_chunk):
            if word.startswith("**") and word.endswith("**"):
                matched_index_tuple_list.append((j, j))
            elif word.startswith("**"):
                matched_index_tuple_list.append((j, -1))
            elif word.endswith("**"):
                matched_index_tuple_list[-1] = (matched_index_tuple_list[-1][0], j)

        output_list: List[str] = []
        last_end: int = -1
        for matched_index_tuple in matched_index_tuple_list:
            start: int = max(0, matched_index_tuple[0] - context_size)
            end: int = min(len(split_chunk), matched_index_tuple[1] + context_size + 1)
            if (start != 0) and (start > last_end):
                output_list.append("...")
            elif start < last_end:
                output_list = output_list[: start - last_end]
            output_list.extend(split_chunk[start:end])
            last_end = end
        if len(output_list) > 0:
            if last_end < len(split_chunk):
                output_list.append("...")
            output_list_batch[i] = (" ".join(output_list), 1)

        if output_list_batch[i] is None:
            if i in embedding_similar_chunk_info_dict:
                output_list_batch[i] = (
                    chunk_batch[i],
                    embedding_similar_chunk_info_dict[i],
                )

    return output_list_batch
