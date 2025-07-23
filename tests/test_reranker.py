import os

import pytest
from dotenv import load_dotenv
from lancedb.rerankers import VoyageAIReranker

from hirag_prod._llm import EmbeddingService
from hirag_prod.reranker import LocalReranker
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

    # Verify results
    assert recall_query is not None
    assert len(recall_query) == topn

    for result in recall_query:
        assert "text" in result.keys()
        assert result["text"] is not None
        assert "_relevance_score" in result.keys()
        assert result["_relevance_score"] is not None


@pytest.mark.asyncio
async def test_reranker_initialization():
    """Test that reranker initializes correctly based on environment"""
    reranker_type = os.getenv("RERANKER_TYPE", "api")

    if reranker_type == "local":

        reranker = LocalReranker()
        assert reranker.base_url
        assert reranker.auth_token
        assert reranker.model_name

    else:
        reranker = VoyageAIReranker(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model_name=os.getenv("API_RERANKER_MODEL", "rerank-2"),
            return_score="relevance",
        )
        assert reranker.api_key
        assert reranker.model_name
