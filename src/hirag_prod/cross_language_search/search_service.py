import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy import ARRAY, Integer, and_, case, func, or_, text, tuple_
from sqlalchemy.sql.functions import coalesce

from hirag_prod.configs.functions import get_envs
from hirag_prod.cross_language_search.functions import (
    build_search_result,
    classify_search,
    create_embeddings_batch,
    get_synonyms_and_validate_and_translate,
    has_traditional_chinese,
    normalize_text,
    prepare_text_to_embed,
    validate_similarity,
)
from hirag_prod.resources.functions import (
    get_chinese_convertor,
)
from hirag_prod.schema import Item
from hirag_prod.storage.vdb_utils import get_item_info_by_scope


async def cross_language_search(
    knowledge_base_id: str,
    workspace_id: str,
    search_content: str,
    ai_search: bool = True,
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    search_embedding_np_array_dict: Optional[Dict[str, np.ndarray]] = None
    search_keyword_list_original: Optional[List[str]] = None
    search_keyword_list: Optional[List[str]] = None
    search_sentence_list_original: Optional[List[str]] = None
    search_sentence_list: Optional[List[str]] = None
    if ai_search:
        (
            synonym_list,
            synonym_embedding_np_array,
            is_english,
            translation_list,
            translation_embedding_np_array,
        ) = await get_synonyms_and_validate_and_translate(search_content)
        if is_english:
            search_list_original_language: List[str] = []
            search_list: List[str] = synonym_list
        else:
            search_list_original_language: List[str] = synonym_list
            search_list: List[str] = translation_list

        (
            search_keyword_list_original,
            keyword_embedding_np_array_original,
            search_sentence_list_original,
            sentence_embedding_np_array_original,
        ) = classify_search(search_list_original_language, synonym_embedding_np_array)
        (
            search_keyword_list,
            keyword_embedding_np_array,
            search_sentence_list,
            sentence_embedding_np_array,
        ) = classify_search(
            search_list,
            (
                synonym_embedding_np_array
                if is_english
                else translation_embedding_np_array
            ),
        )

        search_embedding_np_array_dict = {
            "search_keyword": np.concatenate(
                [keyword_embedding_np_array_original, keyword_embedding_np_array]
            ),
            "search_sentence": np.concatenate(
                [sentence_embedding_np_array_original, sentence_embedding_np_array],
                axis=0,
            ),
        }
        del keyword_embedding_np_array_original
        del sentence_embedding_np_array_original
        del keyword_embedding_np_array
        del sentence_embedding_np_array

    last_cursor: Optional[Any] = None
    batch_size: int = get_envs().KNOWLEDGE_BASE_SEARCH_BATCH_SIZE
    additional_data_to_select: Optional[Dict[Union[str, Tuple[str, ...]], Any]] = None
    additional_where_clause_ai_search_base: Optional[Any] = None
    if ai_search:
        search_by_search_keyword_list_postgres_function: (
            Any
        ) = func.search_by_search_keyword_list(
            Item.token_list,
            Item.translation_token_list,
            search_keyword_list_original,
            search_keyword_list,
        ).table_valued(
            "matched_index_list_original",
            "matched_index_list_translation",
            type_=ARRAY(Integer),
        )
        precise_search_by_search_sentence_list_postgres_function: (
            Any
        ) = func.precise_search_by_search_sentence_list(
            Item.text_normalized,
            Item.translation_normalized,
            search_sentence_list_original,
            search_sentence_list,
        ).table_valued(
            "fuzzy_match_start_index_list_original",
            "fuzzy_match_end_index_list_original",
            "fuzzy_match_start_index_list_translation",
            "fuzzy_match_end_index_list_translation",
            type_=ARRAY(Integer),
        )
        additional_data_to_select = {
            (
                "matched_index_list_original",
                "matched_index_list_translation",
            ): search_by_search_keyword_list_postgres_function,
            (
                "fuzzy_match_start_index_list_original",
                "fuzzy_match_end_index_list_original",
                "fuzzy_match_start_index_list_translation",
                "fuzzy_match_end_index_list_translation",
            ): precise_search_by_search_sentence_list_postgres_function,
        }
        additional_where_clause_ai_search_base = or_(
            search_by_search_keyword_list_postgres_function.c.matched_index_list_original.is_not(
                None
            ),
            search_by_search_keyword_list_postgres_function.c.matched_index_list_translation.is_not(
                None
            ),
            precise_search_by_search_sentence_list_postgres_function.c.fuzzy_match_start_index_list_original.is_not(
                None
            ),
            precise_search_by_search_sentence_list_postgres_function.c.fuzzy_match_start_index_list_translation.is_not(
                None
            ),
        )
        if len(search_embedding_np_array_dict["search_sentence"]) > 0:
            get_search_sentence_cosine_distance_postgres_function: Any = func.least(
                *[
                    Item.vector.cosine_distance(sentence_embedding)
                    for sentence_embedding in search_embedding_np_array_dict[
                        "search_sentence"
                    ]
                ]
            )
            additional_data_to_select["search_sentence_cosine_distance"] = (
                get_search_sentence_cosine_distance_postgres_function
            )
            additional_where_clause_ai_search_base = or_(
                additional_where_clause_ai_search_base,
                text("search_sentence_cosine_distance < 0.4"),
            )
    while True:
        if ai_search:
            additional_where_clause: Optional[Any] = (
                additional_where_clause_ai_search_base
            )
            if last_cursor is not None:
                additional_where_clause = and_(
                    additional_where_clause,
                    last_cursor,
                )
        else:
            additional_where_clause: Optional[Any] = func.lower(Item.text).like(
                f"%{search_content.lower()}%", escape="\\"
            )
            if last_cursor is not None:
                additional_where_clause = and_(
                    additional_where_clause,
                    last_cursor,
                )
        chunk_list = await get_item_info_by_scope(
            knowledge_base_id=knowledge_base_id,
            workspace_id=workspace_id,
            columns_to_select=[
                "documentKey",
                "text_normalized",
                "fileName",
                "uri",
                "type",
                "pageNumber",
                "chunkIdx",
                "chunkType",
                "pageWidth",
                "pageHeight",
                "bbox",
                "token_list",
                "token_start_index_list",
                "token_end_index_list",
                "translation_normalized",
                "translation_token_list",
                "translation_token_start_index_list",
                "translation_token_end_index_list",
            ],
            additional_data_to_select=additional_data_to_select,
            additional_where_clause=additional_where_clause,
            order_by=[
                Item.type,
                Item.fileName,
                coalesce(Item.pageNumber, -1),
                case(
                    (Item.type.in_(["pdf", "image"]), -Item.bbox[2]),
                    else_=coalesce(Item.bbox[1], -1.0),
                ),
                case(
                    (Item.type.in_(["pdf", "image"]), Item.bbox[1]),
                    else_=coalesce(Item.bbox[2], -1.0),
                ),
                -coalesce(Item.bbox[4], -1.0),
                coalesce(Item.bbox[3], -1.0),
                Item.chunkIdx,
            ],
            limit=batch_size,
        )

        if len(chunk_list) == 0:
            break

        if ai_search:
            processed_chunk_list: List[Dict[str, Any]] = [
                {
                    "original_normalized": normalize_text(chunk["text_normalized"]),
                    "translation_normalized": normalize_text(
                        chunk["translation_normalized"]
                    ),
                    "original_token_list": chunk["token_list"],
                    "translation_token_list": chunk["translation_token_list"],
                    "original_token_start_index_list": chunk["token_start_index_list"],
                    "original_token_end_index_list": chunk["token_end_index_list"],
                    "translation_token_start_index_list": chunk[
                        "translation_token_start_index_list"
                    ],
                    "translation_token_end_index_list": chunk[
                        "translation_token_end_index_list"
                    ],
                    "has_traditional_chinese": has_traditional_chinese(
                        chunk["text_normalized"]
                    ),
                    "matched_index_list_original": chunk["matched_index_list_original"],
                    "matched_index_list_translation": chunk[
                        "matched_index_list_translation"
                    ],
                    "fuzzy_match_start_index_list_original": chunk[
                        "fuzzy_match_start_index_list_original"
                    ],
                    "fuzzy_match_end_index_list_original": chunk[
                        "fuzzy_match_end_index_list_original"
                    ],
                    "fuzzy_match_start_index_list_translation": chunk[
                        "fuzzy_match_start_index_list_translation"
                    ],
                    "fuzzy_match_end_index_list_translation": chunk[
                        "fuzzy_match_end_index_list_translation"
                    ],
                    "search_sentence_cosine_distance": chunk.get(
                        "search_sentence_cosine_distance", None
                    ),
                }
                for chunk in chunk_list
            ]

            str_list_dict_to_embed: Dict[str, List[str]] = await prepare_text_to_embed(
                processed_chunk_list
            )

            str_embedding_np_array_dict: Dict[str, List[np.ndarray]] = (
                await create_embeddings_batch(str_list_dict_to_embed)
            )

            await validate_similarity(
                str_embedding_np_array_dict,
                search_embedding_np_array_dict,
                processed_chunk_list,
                "keyword",
            )
            await validate_similarity(
                str_embedding_np_array_dict,
                search_embedding_np_array_dict,
                processed_chunk_list,
                "sentence",
            )
            del str_list_dict_to_embed

            build_search_result(
                processed_chunk_list,
            )

            matched_blocks: List[Dict[str, Any]] = []
            similar_block_tuple_list: List[Tuple[Dict[str, Any], float]] = []
            for chunk, processed_chunk in zip(chunk_list, processed_chunk_list):
                result_tuple: Optional[Tuple[str, float]] = processed_chunk[
                    "original_search_result"
                ]
                is_embedding_result: bool = False
                if result_tuple is None:
                    result_tuple = processed_chunk["translation_search_result"]
                if (
                    (result_tuple is None)
                    and ("embedding_search_result" in processed_chunk)
                    and (chunk["chunkType"] in ["text", "list", "table"])
                    and (len(processed_chunk["original_token_list"]) > 1)
                    and (len(processed_chunk["original_normalized"]) > 6)
                    and (
                        not re.sub(
                            r"\s", "", processed_chunk["original_normalized"]
                        ).isnumeric()
                    )
                ):
                    result_tuple = processed_chunk["embedding_search_result"]
                    is_embedding_result = True
                if result_tuple is not None:
                    block = {
                        "markdown": (
                            result_tuple[0]
                            if not processed_chunk["has_traditional_chinese"]
                            else get_chinese_convertor("s2hk").convert(result_tuple[0])
                        ),
                        "chunk": chunk,
                    }
                    if not is_embedding_result:
                        matched_blocks.append(block)
                    else:
                        similar_block_tuple_list.append((block, result_tuple[1]))
            similar_block_tuple_list.sort(key=lambda x: x[1])

            if (len(matched_blocks) > 0) or (len(similar_block_tuple_list) > 0):
                yield matched_blocks + [
                    block_tuple[0] for block_tuple in similar_block_tuple_list
                ]
        else:
            matched_blocks: List[Dict[str, Any]] = []
            for chunk in chunk_list:
                markdown = re.sub(
                    re.escape(search_content),
                    r"<mark>\g<0></mark>",
                    chunk["text"],
                    flags=re.IGNORECASE,
                )
                block = {
                    "markdown": markdown,
                    "chunk": chunk,
                }
                matched_blocks.append(block)
            if len(matched_blocks) > 0:
                yield matched_blocks

        last_cursor = tuple_(
            Item.type,
            Item.fileName,
            coalesce(Item.pageNumber, -1),
            (
                -Item.bbox[2]
                if chunk_list[-1]["type"] in ["pdf", "image"]
                else coalesce(Item.bbox[1], -1.0)
            ),
            (
                Item.bbox[1]
                if chunk_list[-1]["type"] in ["pdf", "image"]
                else coalesce(Item.bbox[2], -1.0)
            ),
            -coalesce(Item.bbox[4], -1.0),
            coalesce(Item.bbox[3], -1.0),
            Item.chunkIdx,
        ) > (
            chunk_list[-1]["type"],
            chunk_list[-1]["fileName"],
            (
                chunk_list[-1]["pageNumber"]
                if chunk_list[-1]["pageNumber"] is not None
                else -1
            ),
            (
                -chunk_list[-1]["bbox"][1]
                if chunk_list[-1]["type"] in ["pdf", "image"]
                else (
                    chunk_list[-1]["bbox"][0]
                    if (
                        (chunk_list[-1]["bbox"] is not None)
                        and (len(chunk_list[-1]["bbox"]) > 0)
                    )
                    else -1.0
                )
            ),
            (
                chunk_list[-1]["bbox"][0]
                if chunk_list[-1]["type"] in ["pdf", "image"]
                else (
                    chunk_list[-1]["bbox"][1]
                    if (
                        (chunk_list[-1]["bbox"] is not None)
                        and (len(chunk_list[-1]["bbox"]) > 1)
                    )
                    else -1.0
                )
            ),
            -(
                chunk_list[-1]["bbox"][3]
                if (chunk_list[-1]["bbox"] is not None)
                and (len(chunk_list[-1]["bbox"]) > 3)
                else -1.0
            ),
            (
                chunk_list[-1]["bbox"][2]
                if (chunk_list[-1]["bbox"] is not None)
                and (len(chunk_list[-1]["bbox"]) > 2)
                else -1.0
            ),
            chunk_list[-1]["chunkIdx"],
        )
    del search_embedding_np_array_dict
