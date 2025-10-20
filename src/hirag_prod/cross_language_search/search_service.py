import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union

from api.schema.chats.request import (
    ExcelHighlight,
    ImageHighlight,
    MarkdownHighlight,
    PDFHighlight,
)
from pgvector import HalfVector
from sqlalchemy import CursorResult, Row, text

from hirag_prod.cross_language_search.functions import (
    get_synonyms_and_validate_and_translate,
    normalize_text,
)
from hirag_prod.resources.functions import (
    get_chinese_convertor,
    get_db_session_maker,
    get_embedding_service,
)


async def cross_language_search(
    knowledge_base_id: str,
    workspace_id: str,
    search_content: Optional[str] = None,
    search_list_original: Optional[List[str]] = None,
    search_list: Optional[List[str]] = None,
    last_row_info: Optional[Tuple[str, int]] = None,
    page_size: int = 100,
    page_delta_number: int = 1,
    ai_search: bool = True,
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    search_embedding_str_list: Optional[List[str]] = None
    if ai_search:
        if (search_list_original is None) or (search_list is None):
            search_content = normalize_text(search_content)
            (
                synonym_list,
                synonym_embedding_np_array,
                is_english,
                translation_list,
                translation_embedding_np_array,
            ) = await get_synonyms_and_validate_and_translate(search_content)
            if is_english:
                search_list_original = []
                search_list = synonym_list
            else:
                search_list_original = synonym_list
                search_list = translation_list

            search_embedding_str_list = []
            for embedding in synonym_embedding_np_array:
                search_embedding_str_list.append(HalfVector(embedding).to_text())
            for embedding in translation_embedding_np_array:
                search_embedding_str_list.append(HalfVector(embedding).to_text())
            del synonym_embedding_np_array
            del translation_embedding_np_array
        else:
            search_embedding_str_list = [
                HalfVector(embedding).to_text()
                for embedding in await get_embedding_service().create_embeddings(
                    search_list_original + search_list
                )
            ]

    sql_str: str = f"""SELECT *
FROM (
    SELECT "Items"."documentKey", "Items"."chunkIdx", "Items".text_normalized, "Items".has_traditional_chinese, "Items"."fileName", "Items".uri, "Items".type, "Items"."pageNumber", "Items"."chunkType", "Items"."pageWidth", "Items"."pageHeight", "Items".bbox{", search_by_search_list(:search_list_original, \"Items\".text_normalized, \"Items\".token_list, \"Items\".token_start_index_list, \"Items\".token_end_index_list, :search_list, \"Items\".translation_normalized, \"Items\".translation_token_list, \"Items\".translation_token_start_index_list, \"Items\".translation_token_end_index_list) AS search_result" if ai_search else ""}{f", least({', '.join([f'\"Items\".vector <=> :search_vector_{i}' for i in range(len(search_embedding_str_list))])}) AS cosine_distance" if ai_search and (search_embedding_str_list is not None) and (len(search_embedding_str_list) > 0) else ""}
    FROM "Items" 
    WHERE "Items"."workspaceId" = :workspace_id AND "Items"."knowledgeBaseId" = :knowledge_base_id{" AND text_normalized LIKE :search_content ESCAPE '\\'" if not ai_search else ""}{{start_point_where_clause_str}}
    ORDER BY "Items".type, "Items"."fileName", coalesce("Items"."pageNumber", -1), CASE WHEN ("Items".type IN ('pdf', 'image')) THEN -"Items".bbox[2] ELSE coalesce("Items".bbox[1], -1.0) END, CASE WHEN ("Items".type IN ('pdf', 'image')) THEN "Items".bbox[1] ELSE coalesce("Items".bbox[2], -1.0) END, -coalesce("Items".bbox[4], -1.0), coalesce("Items".bbox[3], -1.0), "Items"."chunkIdx"
) AS sub_query
{"WHERE sub_query.search_result IS NOT NULL" if ai_search else ""}{" OR (sub_query.cosine_distance < 0.4 AND sub_query.\"chunkType\" IN ('text', 'list', 'table') AND LENGTH(sub_query.text_normalized) > 12 AND NOT (REGEXP_REPLACE(sub_query.text_normalized, '\\s', '', 'g') ~ '^[0-9]+$'))" if ai_search and (search_embedding_str_list is not None) and (len(search_embedding_str_list) > 0) else ""}
LIMIT :batch_size
"""

    sql_parameter_dict: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "knowledge_base_id": knowledge_base_id,
        "batch_size": page_size,
    }
    if ai_search:
        sql_parameter_dict["search_list_original"] = search_list_original
        sql_parameter_dict["search_list"] = search_list
        if search_embedding_str_list is not None:
            for i, embedding in enumerate(search_embedding_str_list):
                sql_parameter_dict[f"search_vector_{i}"] = embedding
    else:
        sql_parameter_dict["search_content"] = f"%{normalize_text(search_content)}%"

    start_point_where_clause_str: Optional[str] = None
    if last_row_info is not None:
        async with get_db_session_maker()() as session:
            query_result: CursorResult = await session.execute(
                text(
                    'SELECT "Items".type, "Items"."fileName", "Items"."pageNumber", "Items".bbox FROM "Items" WHERE "Items"."documentKey" = :document_key AND "Items"."chunkIdx" = :chunk_idx'
                ),
                {"document_key": last_row_info[0], "chunk_idx": last_row_info[1]},
            )
            row: Row[Any] = query_result.first()
            if row is not None:
                start_point_where_clause_str = f' AND ("Items".type, "Items"."fileName", coalesce("Items"."pageNumber", -1), {"-\"Items\".bbox[2]" if row[0] in ["pdf", "image"] else "coalesce(\"Items\".bbox[1], -1.0)"}, {"\"Items\".bbox[1]" if row[0] in ["pdf", "image"] else "coalesce(\"Items\".bbox[2], -1.0)"}, -coalesce("Items".bbox[4], -1.0), coalesce("Items".bbox[3], -1.0), "Items"."chunkIdx") > (:type, :file_name, :page_number, :bbox_item_1, :bbox_item_2, :bbox_item_3, :bbox_item_4, :chunk_idx)'
                sql_parameter_dict["type"] = row[0]
                sql_parameter_dict["file_name"] = row[1]
                sql_parameter_dict["page_number"] = row[2] if row[2] is not None else -1
                sql_parameter_dict["bbox_item_1"] = (
                    -row[3][1]
                    if row[0] in ["pdf", "image"]
                    else (
                        row[3][0]
                        if ((row[3] is not None) and (len(row[3]) > 0))
                        else -1.0
                    )
                )
                sql_parameter_dict["bbox_item_2"] = (
                    row[3][0]
                    if row[0] in ["pdf", "image"]
                    else (
                        row[3][1]
                        if ((row[3] is not None) and (len(row[3]) > 1))
                        else -1.0
                    )
                )
                sql_parameter_dict["bbox_item_3"] = -(
                    row[3][3] if ((row[3] is not None) and (len(row[3]) > 3)) else -1.0
                )
                sql_parameter_dict["bbox_item_4"] = (
                    row[3][2] if ((row[3] is not None) and (len(row[3]) > 2)) else -1.0
                )
                sql_parameter_dict["chunk_idx"] = last_row_info[1]

    for i in range(page_delta_number):
        has_more: bool = False
        if i == page_delta_number - 1:
            sql_parameter_dict["batch_size"] = page_size + 1
        async with get_db_session_maker()() as session:
            query_result: CursorResult = await session.execute(
                text(
                    sql_str.format(
                        start_point_where_clause_str=(
                            start_point_where_clause_str
                            if start_point_where_clause_str is not None
                            else ""
                        )
                    )
                ),
                sql_parameter_dict,
            )
            row_list: Sequence[Row[Any]] = query_result.all()
        if (i == page_delta_number - 1) and (len(row_list) > page_size):
            has_more = True
            row_list = row_list[:-1]
        elif i < page_delta_number - 1:
            start_point_where_clause_str = f' AND ("Items".type, "Items"."fileName", coalesce("Items"."pageNumber", -1), {"-\"Items\".bbox[2]" if row_list[-1][6] in ["pdf", "image"] else "coalesce(\"Items\".bbox[1], -1.0)"}, {"\"Items\".bbox[1]" if row_list[-1][6] in ["pdf", "image"] else "coalesce(\"Items\".bbox[2], -1.0)"}, -coalesce("Items".bbox[4], -1.0), coalesce("Items".bbox[3], -1.0), "Items"."chunkIdx") > (:type, :fileName, :pageNumber, :bbox_item_1, :bbox_item_2, :bbox_item_3, :bbox_item_4, :chunkIdx)'
            sql_parameter_dict["type"] = row_list[-1][6]
            sql_parameter_dict["fileName"] = row_list[-1][4]
            sql_parameter_dict["pageNumber"] = (
                row_list[-1][7] if row_list[-1][7] is not None else -1
            )
            sql_parameter_dict["bbox_item_1"] = (
                -row_list[-1][11][1]
                if row_list[-1][6] in ["pdf", "image"]
                else (
                    row_list[-1][11][0]
                    if ((row_list[-1][11] is not None) and (len(row_list[-1][11]) > 0))
                    else -1.0
                )
            )
            sql_parameter_dict["bbox_item_2"] = (
                row_list[-1][11][0]
                if row_list[-1][6] in ["pdf", "image"]
                else (
                    row_list[-1][11][1]
                    if ((row_list[-1][11] is not None) and (len(row_list[-1][11]) > 1))
                    else -1.0
                )
            )
            sql_parameter_dict["bbox_item_3"] = -(
                row_list[-1][11][3]
                if ((row_list[-1][11] is not None) and (len(row_list[-1][11]) > 3))
                else -1.0
            )
            sql_parameter_dict["bbox_item_4"] = (
                row_list[-1][11][2]
                if ((row_list[-1][11] is not None) and (len(row_list[-1][11]) > 2))
                else -1.0
            )
            sql_parameter_dict["chunkIdx"] = row_list[-1][1]

        if len(row_list) == 0:
            break

        search_result_list: List[Dict[str, Any]] = []
        embedding_search_result_list: List[Tuple[Dict[str, Any], float]] = []
        for row in row_list:
            result: Optional[Union[str, Tuple[str, float]]]
            if len(row) > 12:
                if row[12] is not None:
                    result = row[12]
                else:
                    result = (
                        row[2],
                        row[13],
                    )
            else:
                result = re.sub(
                    re.escape(search_content),
                    r"<mark>\g<0></mark>",
                    row[2],
                    flags=re.IGNORECASE,
                )
            if result is not None:
                if row[6] == "pdf":
                    highlight = PDFHighlight(
                        x1=row[11][0],
                        y1=row[11][1],
                        x2=row[11][2],
                        y2=row[11][3],
                        page_number=row[7],
                        width=row[9],
                        height=row[10],
                    ).to_dict()
                elif row[6] in ["md", "txt"]:
                    highlight = MarkdownHighlight(
                        from_idx=row[11][0],
                        to_idx=row[11][1],
                    ).to_dict()
                elif row[6] == "xlsx":
                    if row[11]:
                        highlight = ExcelHighlight(
                            col=row[11][0],
                            row=row[11][1],
                        ).to_dict()
                    else:
                        highlight = ExcelHighlight(
                            col=None,
                            row=None,
                        ).to_dict()
                else:
                    highlight = ImageHighlight(
                        x1=row[11][0],
                        y1=row[11][1],
                        x2=row[11][2],
                        y2=row[11][3],
                        width=row[9],
                        height=row[10],
                    ).to_dict()

                ext = row[5].split(".")[-1]
                if isinstance(result, str):
                    search_result_list.append(
                        {
                            "markdown": (
                                result
                                if not row[3]
                                else get_chinese_convertor("hk2s").convert(result)
                            ),
                            "id": row[0],
                            "chunkIdx": row[1],
                            "fileUrl": (row[5]),
                            "type": ext,
                            "highlight": highlight,
                            "fileName": row[4],
                        }
                    )
                else:
                    embedding_search_result_list.append(
                        (
                            {
                                "markdown": (
                                    result[0]
                                    if not row[3]
                                    else get_chinese_convertor("hk2s").convert(
                                        result[0]
                                    )
                                ),
                                "id": row[0],
                                "chunkIdx": row[1],
                                "fileUrl": (row[5]),
                                "type": ext,
                                "highlight": highlight,
                                "fileName": row[4],
                            },
                            result[1],
                        )
                    )
        embedding_search_result_list.sort(key=lambda x: x[1])
        search_result_list.extend(
            [
                embedding_search_result[0]
                for embedding_search_result in embedding_search_result_list
            ]
        )
        if has_more:
            search_result_list[-1]["hasMore"] = True
            search_result_list[-1]["search_list_original"] = search_list_original
            search_result_list[-1]["search_list"] = search_list
        yield search_result_list
    del search_embedding_str_list
