import os

import pytest
from dotenv import load_dotenv

from hirag_prod._llm import EmbeddingService
from hirag_prod.chunk import FixTokenChunk
from hirag_prod.loader import load_document
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
    with open(os.path.join(os.path.dirname(__file__), "test.txt"), "r") as f:
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


@pytest.mark.asyncio
async def test_reranker_with_chunked_documents():
    strategy_provider = RetrievalStrategyProvider()
    lance_db = await LanceDB.create(
        embedding_func=EmbeddingService().create_embeddings,
        db_url="kb/test.db",
        strategy_provider=strategy_provider,
    )

    # Load a pdf ducument, no relevant contents with query
    document_path = f"{os.path.dirname(__file__)}/Guide-to-U.S.-Healthcare-System.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    documents = load_document(
        document_path, content_type, document_meta, loader_type="langchain"
    )
    # For saving time, reduce the number of documents to 5
    documents = documents[:5]

    # Load a text document, containing contents highly relevant to the query
    document_path2 = f"{os.path.dirname(__file__)}/test.txt"
    content_type2 = "text/csv"
    document_meta2 = {
        "type": "text",
        "filename": "test.txt",
        "uri": document_path2,
        "private": False,
    }
    documents2 = load_document(
        document_path2, content_type2, document_meta2, loader_type="langchain"
    )

    # For saving time, reduce the number of documents to 5
    documents2 = documents2[:5]
    documents.extend(documents2)

    # Test chunking the loaded documents
    chunker = FixTokenChunk(chunk_size=500, chunk_overlap=50)
    chunked_docs = []

    for document in documents:
        chunks = chunker.chunk(document)
        chunked_docs.extend(chunks)

    # Verify the chunking results
    assert chunked_docs is not None
    assert len(chunked_docs) > 0

    # Check that each chunk has the expected metadata
    for chunk_id, chunk in enumerate(chunked_docs):
        assert "chunk_idx" in chunk.metadata.__dict__
        assert chunk.page_content is not None

    document_list = set()
    for chunk_id, chunk in enumerate(chunked_docs):
        mode = "overwrite" if chunk_id == 0 else "append"
        document_list.add(chunk.metadata.uri)
        await lance_db.upsert_text(
            text_to_embed=chunk.page_content,
            properties={
                "text": chunk.page_content,
                "document_key": chunk.metadata.uri,
                "filename": chunk.metadata.filename,
                "private": True,
            },
            table_name="test",
            mode=mode,
        )
    document_list = list(document_list)

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
    document_list.append("test_append")
    topk = 10
    topn = 5
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
