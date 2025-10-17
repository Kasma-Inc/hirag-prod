import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union

from api.schema.chats.request import (
    ExcelHighlight,
    ImageHighlight,
    MarkdownHighlight,
    PDFHighlight,
)
from pgvector import HalfVector
from sqlalchemy import CursorResult, Row, TextClause, text

from hirag_prod.configs.functions import get_envs
from hirag_prod.cross_language_search.functions import (
    get_synonyms_and_validate_and_translate,
    normalize_text,
)
from hirag_prod.resources.functions import (
    get_chinese_convertor,
    get_db_session_maker,
)


async def cross_language_search(
    knowledge_base_id: str,
    workspace_id: str,
    search_content: str,
    ai_search: bool = True,
    start_index: int = 0,
    batch_number: int = 1,
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    search_content = normalize_text(search_content)
    search_list_original: Optional[List[str]] = None
    search_list: Optional[List[str]] = None
    search_embedding_str_list: Optional[List[str]] = None
    if ai_search:
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
        del synonym_embedding_np_array
        del translation_embedding_np_array

    sql: TextClause = text(
        f"""SELECT *
FROM (
    SELECT "Items"."documentKey", "Items".text_normalized, "Items".has_traditional_chinese, "Items"."fileName", "Items".uri, "Items".type, "Items"."pageNumber", "Items"."chunkType", "Items"."pageWidth", "Items"."pageHeight", "Items".bbox{", search_by_search_list(:search_list_original, \"Items\".text_normalized, \"Items\".token_list, \"Items\".token_start_index_list, \"Items\".token_end_index_list, :search_list, \"Items\".translation_normalized, \"Items\".translation_token_list, \"Items\".translation_token_start_index_list, \"Items\".translation_token_end_index_list) AS search_result" if ai_search else ""}{f", least({', '.join([f'\"Items\".vector <=> :search_vector_{i}' for i in range(len(search_embedding_str_list))])}) AS cosine_distance" if ai_search and (search_embedding_str_list is not None) and (len(search_embedding_str_list) > 0) else ""}
    FROM "Items" 
    WHERE "Items"."workspaceId" = :workspace_id AND "Items"."knowledgeBaseId" = :knowledge_base_id{" AND text_normalized LIKE :search_content ESCAPE '\\'" if not ai_search else ""}
    ORDER BY "Items".type, "Items"."fileName", coalesce("Items"."pageNumber", -1), CASE WHEN ("Items".type IN ('pdf', 'image')) THEN -"Items".bbox[2] ELSE coalesce("Items".bbox[1], -1.0) END, CASE WHEN ("Items".type IN ('pdf', 'image')) THEN "Items".bbox[1] ELSE coalesce("Items".bbox[2], -1.0) END, -coalesce("Items".bbox[4], -1.0), coalesce("Items".bbox[3], -1.0), "Items"."chunkIdx"
) AS sub_query
{"WHERE sub_query.search_result IS NOT NULL" if ai_search else ""}{" OR sub_query.cosine_distance < 0.4" if ai_search and (search_embedding_str_list is not None) and (len(search_embedding_str_list) > 0) else ""}
LIMIT :batch_size
OFFSET :start_index
"""
    )

    batch_size: int = get_envs().KNOWLEDGE_BASE_SEARCH_BATCH_SIZE
    sql_parameter_dict: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "knowledge_base_id": knowledge_base_id,
        "batch_size": batch_size,
    }
    if ai_search:
        sql_parameter_dict["search_list_original"] = search_list_original
        sql_parameter_dict["search_list"] = search_list
        if search_embedding_str_list is not None:
            for i, embedding in enumerate(search_embedding_str_list):
                sql_parameter_dict[f"search_vector_{i}"] = embedding
    else:
        sql_parameter_dict["search_content"] = f"%{search_content}%"

    current_start_index: int = start_index
    for _ in range(batch_number):
        sql_parameter_dict["start_index"] = current_start_index
        async with get_db_session_maker()() as session:
            query_result: CursorResult = await session.execute(
                sql,
                sql_parameter_dict,
            )
            row_list: Sequence[Row[Any]] = query_result.all()

        if len(row_list) == 0:
            break

        search_result_list: List[Dict[str, Any]] = []
        embedding_search_result_list: List[Tuple[Dict[str, Any], float]] = []
        for row in row_list:
            result: Optional[Union[str, Tuple[str, float]]] = None
            if len(row) > 11:
                if row[11] is not None:
                    result = row[11]
                else:
                    if (
                        (len(row) == 13)
                        and (row[7] in ["text", "list", "table"])
                        and (len(row[1]) > 12)
                        and (not re.sub(r"\s", "", row[1]).isnumeric())
                    ):
                        result = (
                            row[1],
                            row[12],
                        )
            else:
                result = re.sub(
                    re.escape(search_content),
                    r"<mark>\g<0></mark>",
                    row[1],
                    flags=re.IGNORECASE,
                )
            if result is not None:
                if row[5] == "pdf":
                    highlight = PDFHighlight(
                        x1=row[10][0],
                        y1=row[10][1],
                        x2=row[10][2],
                        y2=row[10][3],
                        page_number=row[6],
                        width=row[8],
                        height=row[9],
                    ).to_dict()
                elif row[5] in ["md", "txt"]:
                    highlight = MarkdownHighlight(
                        from_idx=row[10][0],
                        to_idx=row[10][1],
                    ).to_dict()
                elif row[5] == "xlsx":
                    if row[10]:
                        highlight = ExcelHighlight(
                            col=row[10][0],
                            row=row[10][1],
                        ).to_dict()
                    else:
                        highlight = ExcelHighlight(
                            col=None,
                            row=None,
                        ).to_dict()
                else:
                    highlight = ImageHighlight(
                        x1=row[10][0],
                        y1=row[10][1],
                        x2=row[10][2],
                        y2=row[10][3],
                        width=row[8],
                        height=row[9],
                    ).to_dict()

                ext = row[4].split(".")[-1]
                if isinstance(result, str):
                    search_result_list.append(
                        {
                            "markdown": (
                                result
                                if not row[2]
                                else get_chinese_convertor("hk2s").convert(result)
                            ),
                            "id": row[0],
                            "fileUrl": (row[4]),
                            "type": ext,
                            "highlight": highlight,
                            "fileName": row[3],
                        }
                    )
                else:
                    embedding_search_result_list.append(
                        (
                            {
                                "markdown": (
                                    result[0]
                                    if not row[2]
                                    else get_chinese_convertor("hk2s").convert(
                                        result[0]
                                    )
                                ),
                                "id": row[0],
                                "fileUrl": (row[4]),
                                "type": ext,
                                "highlight": highlight,
                                "fileName": row[3],
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
        yield search_result_list
        current_start_index += batch_size
    del search_embedding_str_list
