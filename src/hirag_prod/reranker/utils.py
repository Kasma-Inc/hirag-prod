from typing import List, Dict
from .factory import create_reranker

async def apply_reranking(query: str, results: List[Dict], topn: int, topk: int) -> List[Dict]:
    if not results:
        return results
    
    reranker = create_reranker()
    items_to_rerank = results[:topn]
    reranked_items = await reranker.rerank(query, items_to_rerank, topn)
    remaining_items = results[topn:]
    final_results = reranked_items + remaining_items
    return final_results[:topk]