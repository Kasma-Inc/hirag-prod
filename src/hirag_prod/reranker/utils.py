from typing import Dict, List, Union

from hirag_prod.resources.functions import get_reranker


async def apply_reranking(
    query: Union[str, List[str]],
    results: List[Dict],
    topk: int,
    topn: int,
    key: str = "text",
) -> List[Dict]:
    if not results:
        return results
    # Top k is the number of items to rerank, and top n is the final number of items to return
    topn = min(topn, len(results))
    topk = min(topk, len(results))
    if topn > topk:
        raise ValueError(f"topn ({topn}) must be <= topk ({topk})")
    reranker = get_reranker()
    items_to_rerank = results[:topk]
    reranked_items = await reranker.rerank(query=query, items=items_to_rerank, key=key)
    return reranked_items[:topn]
