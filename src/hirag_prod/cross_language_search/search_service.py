import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy import String, and_, case, func, or_, text, tuple_
from sqlalchemy.sql.functions import coalesce

from hirag_prod.configs.functions import get_envs
from hirag_prod.cross_language_search.functions import (
    get_synonyms_and_validate_and_translate,
    has_traditional_chinese,
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
    search_embedding_np_array: Optional[np.ndarray] = None
    if ai_search:
        (
            synonym_list,
            synonym_embedding_np_array,
            is_english,
            translation_list,
            translation_embedding_np_array,
        ) = await get_synonyms_and_validate_and_translate(search_content)
        if is_english:
            search_list_original: List[str] = []
            search_list: List[str] = synonym_list
        else:
            search_list_original: List[str] = synonym_list
            search_list: List[str] = translation_list

        search_embedding_np_array = np.concatenate(
            [synonym_embedding_np_array, translation_embedding_np_array],
            axis=0,
        )

    last_cursor: Optional[Any] = None
    batch_size: int = get_envs().KNOWLEDGE_BASE_SEARCH_BATCH_SIZE

    def additional_sql_generator(
        subquery_column: Optional[Any],
    ) -> Tuple[Optional[Dict[Union[str, Tuple[str, ...]], Any]], Optional[Any]]:
        if not ai_search:
            return None, None
        elif subquery_column is None:
            return None, None
        search_by_search_list_postgres_function: Any = func.search_by_search_list(
            subquery_column.text_normalized,
            subquery_column.translation_normalized,
            subquery_column.token_list,
            subquery_column.token_start_index_list,
            subquery_column.token_end_index_list,
            subquery_column.translation_token_list,
            subquery_column.translation_token_start_index_list,
            subquery_column.translation_token_end_index_list,
            search_list_original,
            search_list,
        ).table_valued(
            "original_text_search_result",
            "translation_search_result",
            type_=String,
        )
        additional_data_to_select: Dict[Union[str, Tuple[str, ...]], Any] = {
            (
                "original_text_search_result",
                "translation_search_result",
            ): search_by_search_list_postgres_function,
        }
        additional_where_clause: Any = or_(
            search_by_search_list_postgres_function.c.original_text_search_result.is_not(
                None
            ),
            search_by_search_list_postgres_function.c.translation_search_result.is_not(
                None
            ),
        )
        if (search_embedding_np_array is not None) and (
            len(search_embedding_np_array) > 0
        ):
            get_search_sentence_cosine_distance_postgres_function: Any = func.least(
                *[
                    subquery_column.vector.cosine_distance(sentence_embedding)
                    for sentence_embedding in search_embedding_np_array
                ]
            )
            additional_data_to_select["search_sentence_cosine_distance"] = (
                get_search_sentence_cosine_distance_postgres_function
            )
            additional_where_clause = or_(
                additional_where_clause,
                text("search_sentence_cosine_distance < 0.4"),
            )
        return additional_data_to_select, additional_where_clause

    while True:
        if ai_search:
            where_clause: Optional[Any] = last_cursor
        else:
            where_clause: Optional[Any] = Item.text_normalized.like(
                f"%{search_content.lower()}%", escape="\\"
            )
            if last_cursor is not None:
                where_clause = and_(
                    where_clause,
                    last_cursor,
                )

        def order_by_generator(object_to_select: Optional[Any]) -> Optional[List[Any]]:
            if not ai_search:
                object_to_select = Item
            elif object_to_select is None:
                return None
            return [
                object_to_select.type,
                object_to_select.fileName,
                coalesce(object_to_select.pageNumber, -1),
                case(
                    (
                        object_to_select.type.in_(["pdf", "image"]),
                        -object_to_select.bbox[2],
                    ),
                    else_=coalesce(object_to_select.bbox[1], -1.0),
                ),
                case(
                    (
                        object_to_select.type.in_(["pdf", "image"]),
                        object_to_select.bbox[1],
                    ),
                    else_=coalesce(object_to_select.bbox[2], -1.0),
                ),
                -coalesce(object_to_select.bbox[4], -1.0),
                coalesce(object_to_select.bbox[3], -1.0),
                object_to_select.chunkIdx,
            ]

        additional_columns_to_select_in_subquery: List[str] = [
            "token_list",
            "token_start_index_list",
            "token_end_index_list",
            "translation_token_list",
            "translation_token_start_index_list",
            "translation_token_end_index_list",
        ]
        if (search_embedding_np_array is not None) and (
            len(search_embedding_np_array) > 0
        ):
            additional_columns_to_select_in_subquery.append("vector")
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
                "translation_normalized",
            ],
            additional_columns_to_select_in_subquery=additional_columns_to_select_in_subquery,
            where_clause=where_clause,
            additional_sql_generator=additional_sql_generator,
            order_by_generator=order_by_generator,
            limit=batch_size,
            use_subquery=ai_search,
        )

        if len(chunk_list) == 0:
            break

        if ai_search:
            matched_blocks: List[Dict[str, Any]] = []
            similar_block_tuple_list: List[Tuple[Dict[str, Any], float]] = []
            for chunk in chunk_list:
                result: Optional[Union[str, Tuple[str, float]]] = chunk[
                    "original_text_search_result"
                ]
                if result is None:
                    result = chunk["translation_search_result"]
                if (
                    (result is None)
                    and ("search_sentence_cosine_distance" in chunk)
                    and (chunk["chunkType"] in ["text", "list", "table"])
                    and (len(chunk["text_normalized"]) > 12)
                    and (not re.sub(r"\s", "", chunk["text_normalized"]).isnumeric())
                ):
                    result = (
                        chunk["text_normalized"],
                        chunk["search_sentence_cosine_distance"],
                    )
                if result is not None:
                    markdown: str = result if isinstance(result, str) else result[0]
                    if has_traditional_chinese(chunk["text_normalized"]):
                        markdown = get_chinese_convertor("hk2s").convert(markdown)
                    block = {
                        "markdown": markdown,
                        "chunk": chunk,
                    }
                    if isinstance(result, str):
                        matched_blocks.append(block)
                    else:
                        similar_block_tuple_list.append((block, result[1]))
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
                    chunk["text_normalized"],
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
    del search_embedding_np_array
