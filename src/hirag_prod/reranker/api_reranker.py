import os
import httpx
from typing import List, Dict
from .base import Reranker

class ApiReranker(Reranker):
    def __init__(self, api_key: str, endpoint: str, model: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint  
        self.model = model
    
    async def rerank(self, query: str, items: List[Dict], topn: int) -> List[Dict]:
        if not items or topn <= 0:
            return []
        
        topn = min(topn, len(items))
        documents = [item.get("text", "") for item in items]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "query": query,
                    "documents": documents, 
                    "model": self.model,
                    "top_k": topn
                }
            )
            response.raise_for_status()
            
            results = response.json().get("data", [])
            reranked = []
            for r in results:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(items):
                    item = items[idx].copy()
                    item["score"] = r.get("relevance_score", 0.0)
                    reranked.append(item)
            return reranked