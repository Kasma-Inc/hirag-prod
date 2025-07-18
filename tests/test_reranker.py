import os

import pytest
from dotenv import load_dotenv

from hirag_prod._llm import EmbeddingService
from hirag_prod.storage.lancedb import LanceDB
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_reranker():
    strategy_provider = RetrievalStrategyProvider()
    lance_db = await LanceDB.create(
        embedding_func=EmbeddingService().create_embeddings,
        db_url="kb/test.db",
        strategy_provider=strategy_provider,
    )

    # Load a document first using the same approach as in test_loader.py
    with open(os.path.join(os.path.dirname(__file__), "test_files/test.txt"), "r") as f:
        test_to_embed = f.read()

    await lance_db.upsert_text(
        text_to_embed=test_to_embed,
        properties={
            "text": test_to_embed,
            "document_key": "test",
            "filename": "test.txt",
            "private": True,
        },
        table_name="test",
        mode="overwrite",
    )

    async_table = await lance_db.upsert_text(
        text_to_embed="Repeat, Hello, world!",
        properties={
            "text": "Repeat, Hello, world!",
            "document_key": "test_append",
            "filename": "in_memory_test",
            "private": True,
        },
        table_name="test",
        mode="append",
    )
    topk = 2
    topn = 1
    recall_query = await lance_db.query(
        query="tell me about bitcoin",
        table=async_table,
        topk=topk,
        topn=topn,
    )
    assert recall_query is not None
    assert len(recall_query) == topn

    for result in recall_query:
        assert "text" in result.keys()
        assert result["text"] is not None
        assert "_relevance_score" in result.keys()
        assert result["_relevance_score"] is not None
