import re
from typing import Any, Dict, List, Optional, Tuple

from googletrans.models import Translated
from pydantic import BaseModel

from hirag_prod.configs.functions import get_envs, get_llm_config
from hirag_prod.cross_language_search.functions import (
    get_synonyms_and_validate,
    search_by_search_keyword_list,
    search_by_search_sentence_list,
)
from hirag_prod.resources.functions import get_chat_service, get_translator
from hirag_prod.storage.vdb_utils import get_item_info_by_scope


class TranslationResponse(BaseModel):
    translation_list: List[str]


async def cross_language_search(
    knowledgebase_id: str, workspace_id: str, search_content: str
) -> List[Dict[str, Any]]:
    is_english: bool = (await get_translator().detect(search_content)).lang == "en"

    search_list: List[str] = []
    if is_english:
        search_list.append(search_content)
    else:
        if get_envs().SEARCH_TRANSLATOR_TYPE == "google":
            # Translate search content to English
            search_translation_result: Translated = await get_translator().translate(
                search_content, dest="en"
            )
            search_all_translations: Optional[List[List[Any]]] = (
                search_translation_result.extra_data["all-translations"]
            )
            if search_all_translations is not None:
                for translation_type in search_all_translations:
                    search_list.extend(translation_type[1])
            else:
                search_list.append(search_translation_result.text)
        else:
            translation_response: (
                TranslationResponse
            ) = await get_chat_service().complete(
                prompt=f"Please translate the following search keyword or sentence into English. Please translate as briefly as possible. Please give at least 5 different possible translations and output them as a JSON list. The search to translate is {search_content}",
                model=get_llm_config().model_name,
                max_tokens=get_llm_config().max_tokens,
                response_format=TranslationResponse,
                timeout=get_llm_config().timeout,
            )
            search_list.extend(translation_response.translation_list)

    search_keyword_list: List[str] = []
    search_sentence_list: List[str] = []
    for search in search_list:
        if " " not in search:
            search_keyword_list.append(re.sub(r"[^a-zA-Z0-9\s]", "", search))
        else:
            search_sentence_list.append(search)

    search_keyword_list = await get_synonyms_and_validate(
        search_keyword_list, search_content, not is_english
    )

    matched_blocks: List[Dict[str, Any]] = []
    similar_block_tuple_list: List[Tuple[Dict[str, Any], float]] = []
    chunks = await get_item_info_by_scope(
        knowledge_base_id=knowledgebase_id,
        workspace_id=workspace_id,
    )
    keyword_search_result_list: List[Optional[str]]
    matched_index_list_batch: List[Optional[List[Optional[int]]]]
    keyword_search_result_list, matched_index_list_batch = (
        await search_by_search_keyword_list(
            [chunk["translation"] for chunk in chunks], search_keyword_list
        )
    )
    search_result_list: List[Optional[Tuple[str, float]]] = (
        await search_by_search_sentence_list(
            matched_index_list_batch,
            [chunk["translation"] for chunk in chunks],
            search_sentence_list,
            [chunk["vector"] for chunk in chunks],
        )
    )

    for chunk, result_tuple in zip(chunks, search_result_list):
        if result_tuple is not None:
            block = {
                "markdown": result_tuple[0],
                "chunk": chunk,
            }
            if "**" in result_tuple[0]:
                matched_blocks.append(block)
            else:
                similar_block_tuple_list.append((block, result_tuple[1]))
    similar_block_tuple_list.sort(key=lambda x: x[1], reverse=True)

    return matched_blocks + [block_tuple[0] for block_tuple in similar_block_tuple_list]
