import pytest
from dotenv import load_dotenv

from hirag_prod._llm import ChatCompletion
from hirag_prod.schema import Entity, Relation
from hirag_prod.storage.networkx import NetworkXGDB

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_networkx_gdb():
    relations = [
        Relation(
            source="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            target="ent-5a28a79d61d9ba7001246e3fdebbe108",
            properties={
                "description": "The United States operates a free market health care system, which defines its overall structure and operation.",
                "weight": 9.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-5a28a79d61d9ba7001246e3fdebbe108",
            target="ent-2a422318fc58c5302a5ba9365bcbc0be",
            properties={
                "description": "The health care system in the U.S. is heavily influenced by insurance companies that provide policies to consumers and sign contracts with healthcare providers.",
                "weight": 8.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            target="ent-2a422318fc58c5302a5ba9365bcbc0be",
            properties={
                "description": "Insurance companies operate within the framework of the U.S. health care system, affecting how services are delivered and financed.",
                "weight": 7.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-2a422318fc58c5302a5ba9365bcbc0be",
            target="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            properties={
                "description": "Insurance companies restrict payment to health care providers based on contracts that set fixed fees for services.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
    ]

    gdb = NetworkXGDB.create(
        path="test.gpickle",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
    )
    for relation in relations:
        await gdb.upsert_relation(relation)
    await gdb.dump()


@pytest.mark.asyncio
async def test_merge_node():
    gdb = NetworkXGDB.create(
        path="test.gpickle",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
    )
    description1 = "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
    description2 = "The medical system in the United States is a complex network of hospitals, clinics, and other healthcare providers that provide medical care to the population."
    node1 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description1,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    node2 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description2,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    await gdb.upsert_node(node1)
    await gdb.upsert_node(node2)

    node = await gdb.query_node(node1.id)
    assert node.metadata.description != description1
    assert node.metadata.description != description2
    assert isinstance(node.metadata.description, str)
    assert len(node.metadata.description) > 0


@pytest.mark.asyncio
async def test_query_one_hop():
    gdb = NetworkXGDB.create(path="test.gpickle", llm_func=ChatCompletion().complete)

    relations = [
        Relation(
            source="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            target="ent-5a28a79d61d9ba7001246e3fdebbe108",
            properties={
                "description": "The United States operates a free market health care system, which defines its overall structure and operation.",
                "weight": 9.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-5a28a79d61d9ba7001246e3fdebbe108",
            target="ent-2a422318fc58c5302a5ba9365bcbc0be",
            properties={
                "description": "The health care system in the U.S. is heavily influenced by insurance companies that provide policies to consumers and sign contracts with healthcare providers.",
                "weight": 8.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            target="ent-2a422318fc58c5302a5ba9365bcbc0be",
            properties={
                "description": "Insurance companies operate within the framework of the U.S. health care system, affecting how services are delivered and financed.",
                "weight": 7.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source="ent-2a422318fc58c5302a5ba9365bcbc0be",
            target="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            properties={
                "description": "Insurance companies restrict payment to health care providers based on contracts that set fixed fees for services.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
        Relation(
            source="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            target="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            properties={
                "description": "Health care providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
    ]

    gdb = NetworkXGDB.create(
        path="test.gpickle",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
    )

    for relation in relations:
        await gdb.upsert_relation(relation)
    neighbors, edges = await gdb.query_one_hop("ent-8ac4883b1b6f421ea5f0196eb317b2ba")
    assert len(neighbors) == 2
    assert len(edges) == 2
    assert set([n.id for n in neighbors]) == {
        "ent-3ff39c0f9a2e36a5d47ded059ba14673",
        "ent-2a422318fc58c5302a5ba9365bcbc0be",
    }
    assert set([e.source for e in edges]) == {
        "ent-8ac4883b1b6f421ea5f0196eb317b2ba",
    }
    assert set([e.target for e in edges]) == {
        "ent-3ff39c0f9a2e36a5d47ded059ba14673",
        "ent-2a422318fc58c5302a5ba9365bcbc0be",
    }


@pytest.mark.asyncio
async def test_merge_nodes():
    gdb = NetworkXGDB.create(
        path="test.gpickle",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
    )
    description1 = "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
    description2 = "The medical system in the United States is a complex network of hospitals, clinics, and other healthcare providers that provide medical care to the population."
    node1 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description1,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    node2 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description2,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    await gdb.upsert_nodes([node1, node2])
    node = await gdb.query_node(node1.id)
    breakpoint()
    assert node.metadata.description == description1
