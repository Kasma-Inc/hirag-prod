from typing import Any, Dict, List, Optional, Tuple

from hirag_prod.configs.functions import get_llm_config
from hirag_prod.cross_language_search.functions import (
    build_search_result,
    classify_search,
    get_synonyms_and_validate,
    has_traditional_chinese,
    normalize_text,
    search_by_search_keyword_list,
    search_by_search_sentence_list,
)
from hirag_prod.cross_language_search.types import TranslationResponse
from hirag_prod.resources.functions import (
    get_chat_service,
    get_chinese_convertor,
    get_translator,
)
from hirag_prod.storage.vdb_utils import get_item_info_by_scope


async def cross_language_search(
    knowledge_base_id: str, workspace_id: str, search_content: str
) -> List[Dict[str, Any]]:
    language: str = (await get_translator().detect(search_content)).lang
    is_english: bool = language == "en"

    search_list_original_language: List[str] = []
    search_list: List[str] = []
    if is_english:
        search_list = await get_synonyms_and_validate(search_content)
    else:
        search_list_original_language = await get_synonyms_and_validate(search_content)
        translation_response: TranslationResponse = await get_chat_service().complete(
            prompt=f"Please translate the following search keyword or sentence into English. Please translate as briefly as possible. Please give at least 6 different possible translations and output them as a JSON list. The search to translate is {search_content}",
            model=get_llm_config().model_name,
            max_tokens=get_llm_config().max_tokens,
            response_format=TranslationResponse,
            timeout=get_llm_config().timeout,
        )
        search_list.extend(translation_response.translation_list)

    search_keyword_list_original, search_sentence_list_original = classify_search(
        search_list_original_language
    )
    search_keyword_list, search_sentence_list = classify_search(search_list)

    chunk_list = await get_item_info_by_scope(
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
    )
    if len(chunk_list) == 0:
        return []
    processed_chunk_list: List[Dict[str, Any]] = [
        {
            "original_normalized": normalize_text(chunk["text"]),
            "translation_normalized": normalize_text(chunk["translation"]),
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
            "original_embedding": chunk["vector"],
            "has_traditional_chinese": has_traditional_chinese(chunk["text"]),
        }
        for chunk in chunk_list
    ]
    matched_keyword_index_list_dict_batch: List[
        Dict[str, Optional[List[Optional[int]]]]
    ] = await search_by_search_keyword_list(
        processed_chunk_list, search_keyword_list_original, search_keyword_list
    )
    matched_sentence_index_list_dict_batch: List[
        Dict[str, Optional[List[Optional[Tuple[int, int]]]]]
    ]
    embedding_similar_chunk_info_dict: Dict[int, float]
    matched_sentence_index_list_dict_batch, embedding_similar_chunk_info_dict = (
        await search_by_search_sentence_list(
            processed_chunk_list, search_sentence_list_original, search_sentence_list
        )
    )

    build_search_result(
        processed_chunk_list,
        matched_keyword_index_list_dict_batch,
        matched_sentence_index_list_dict_batch,
        embedding_similar_chunk_info_dict,
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
        if (result_tuple is None) and ("embedding_Search_result" in processed_chunk):
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
    similar_block_tuple_list.sort(key=lambda x: x[1], reverse=True)

    return matched_blocks + [block_tuple[0] for block_tuple in similar_block_tuple_list]
