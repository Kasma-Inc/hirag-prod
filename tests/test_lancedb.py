import os

import pytest
from dotenv import load_dotenv

from hirag_prod._llm import EmbeddingService
from hirag_prod.schema import Entity
from hirag_prod.storage.lancedb import LanceDB
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

load_dotenv(override=True)

try:
    EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION"))
except ValueError as e:
    raise ValueError(f"EMBEDDING_DIMENSION must be an integer: {e}")


@pytest.mark.asyncio
async def test_lancedb():
    strategy_provider = RetrievalStrategyProvider()
    lance_db = await LanceDB.create(
        embedding_func=EmbeddingService().create_embeddings,
        db_url="kb/test.db",
        strategy_provider=strategy_provider,
    )

    # Load text from test.txt
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
    table = await lance_db.get_table("test")
    assert table.to_pandas()["text"].iloc[0] == test_to_embed
    assert table.to_pandas()["document_key"].iloc[0] == "test"
    assert table.to_pandas()["filename"].iloc[0] == "test.txt"
    assert table.to_pandas()["private"].iloc[0] == True
    assert table.to_pandas()["vector"].iloc[0].shape == (EMBEDDING_DIMENSION)

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
    table = await lance_db.get_table("test")
    assert table.to_pandas()["text"].iloc[1] == "Repeat, Hello, world!"
    assert table.to_pandas()["document_key"].iloc[1] == "test_append"
    assert table.to_pandas()["filename"].iloc[1] == "in_memory_test"
    assert table.to_pandas()["private"].iloc[1] == True
    assert table.to_pandas()["vector"].iloc[0].shape == (EMBEDDING_DIMENSION)
    assert table.to_pandas().columns.tolist() == [
        "text",
        "document_key",
        "filename",
        "private",
        "vector",
    ]

    recall = await lance_db.query(
        query="tell me about bitcoin",
        table=async_table,
        topk=3,
        columns_to_select=["text", "document_key", "filename", "private"],
        distance_threshold=100,  # a very high threshold to ensure all results are returned
        topn=2,
    )
    assert len(recall) == 2
    assert recall[0]["text"] == test_to_embed
    assert recall[1]["text"] == "Repeat, Hello, world!"


@pytest.mark.asyncio
async def test_lancedb_with_entity():
    entities = [
        Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            metadata={
                "entity_type": '"GEO"',
                "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
            page_content='"UNITED STATES"',
        ),
        Entity(
            id="ent-5a28a79d61d9ba7001246e3fdebbe108",
            metadata={
                "entity_type": '"EVENT"',
                "description": "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage.",
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
            page_content='"HEALTH CARE SYSTEM"',
        ),
        Entity(
            id="ent-2a422318fc58c5302a5ba9365bcbc0be",
            metadata={
                "entity_type": '"ORGANIZATION"',
                "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
            },
            page_content='"INSURANCE COMPANIES"',
        ),
        Entity(
            id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            metadata={
                "entity_type": '"ORGANIZATION"',
                "description": "Health Care Providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
            },
            page_content='"HEALTH CARE PROVIDERS"',
        ),
    ]
    strategy_provider = RetrievalStrategyProvider()
    lance_db = await LanceDB.create(
        embedding_func=EmbeddingService().create_embeddings,
        db_url="kb/test.db",
        strategy_provider=strategy_provider,
    )
    for entity in entities:
        await lance_db.upsert_text(
            text_to_embed=entity.metadata.description,
            properties={
                "document_key": entity.id,
                "text": entity.page_content,
                **entity.metadata.__dict__,
            },
            table_name="test_entity",
            mode="overwrite",
        )
    table = await lance_db.get_table("test_entity")
    assert set(table.schema.names) == {
        "text",
        "document_key",
        "entity_type",
        "description",
        "chunk_ids",
        "vector",
    }

    @pytest.mark.asyncio
    async def test_lancedb_upsert_texts():
        strategy_provider = RetrievalStrategyProvider()
        lance_db = await LanceDB.create(
            embedding_func=EmbeddingService().create_embeddings,
            db_url="kb/batch.db",
            strategy_provider=strategy_provider,
        )

        texts = ["foo", "bar"]
        props = [{"text": t, "document_key": f"id-{i}"} for i, t in enumerate(texts)]

        table = await lance_db.upsert_texts(
            texts_to_embed=texts,
            properties_list=props,
            table_name="batch",
            mode="overwrite",
        )

        data = await table.to_arrow()
        assert data.to_pandas()["text"].tolist() == texts
