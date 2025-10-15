import re
from typing import Dict, List, Set, Union

from hirag_prod.resources.functions import get_reranker


def detect_language(text: str) -> Set[str]:
    detected_languages = set()

    simplified_chinese_pattern = r"[\u4e00-\u9fff]"
    traditional_chinese_specific_pattern = r"[\u3400-\u4dbf\uf900-\ufaff]"
    english_pattern = r"[a-zA-Z]"

    if re.search(english_pattern, text):
        detected_languages.add("ENGLISH")
    if re.search(traditional_chinese_specific_pattern, text):
        detected_languages.add("CHINESE")
    if re.search(simplified_chinese_pattern, text):
        detected_languages.add("CHINESE")

    return detected_languages


async def apply_reranking(
    query: Union[str, List[str]],
    results: List[Dict],
    topk: int,
    topn: int,
    rerank_with_time=False,
    key: str = "text",
) -> List[Dict]:
    if not results:
        return results
    query = query.copy() if isinstance(query, list) else query
    # Top k is the number of items to rerank, and top n is the final number of items to return
    topn = min(topn, len(results))
    topk = min(topk, len(results))
    if topn > topk:
        raise ValueError(f"topn ({topn}) must be <= topk ({topk})")
    reranker = get_reranker()
    items_to_rerank = results[:topk]
    reranked_items = await reranker.rerank(
        query=query, items=items_to_rerank, key=key, rerank_with_time=rerank_with_time
    )
    return reranked_items[:topn]
