import pytest
import time
from dotenv import load_dotenv

from hirag_prod._llm import ChatCompletion
from hirag_prod.schema import Entity, Relation
from hirag_prod.storage.flavius import FlaviusGDB

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_flavius_gdb():
    """Test basic FlaviusGDB operations with relations"""
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

    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",  # 确保Flavius服务运行
    )
    
    try:
        for relation in relations:
            await gdb.upsert_relation(relation)
        await gdb.dump()
    finally:
        await gdb.clean_up()


@pytest.mark.asyncio
async def test_merge_node():
    """Test node merging functionality in FlaviusGDB"""
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )
    
    try:
        description1 = "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
        description2 = "The medical system in the United States is a complex network of hospitals, clinics, and other healthcare providers that provide medical care to the population."
        
        node1 = Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            page_content="UNITED STATES",
            metadata={
                "entity_type": "GEO",
                "description": [description1],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        )
        node2 = Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            page_content="UNITED STATES",
            metadata={
                "entity_type": "GEO",
                "description": [description2],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        )
        
        await gdb.upsert_node(node1)
        await gdb.upsert_node(node2)

        node = await gdb.query_node(node1.id)
        assert node.metadata.description[0] == description1
        assert node.metadata.description[0] != description2
        assert isinstance(node.metadata.description[0], str)
        assert len(node.metadata.description) > 0
    finally:
        await gdb.clean_up()


@pytest.mark.asyncio
async def test_query_one_hop():
    """Test one-hop neighbor query in FlaviusGDB"""
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )

    # First, create all nodes with proper content
    nodes = [
        Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            page_content="UNITED STATES",
            metadata={
                "entity_type": "GEO",
                "description": ["The United States"],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        ),
        Entity(
            id="ent-5a28a79d61d9ba7001246e3fdebbe108",
            page_content="HEALTH CARE SYSTEM",
            metadata={
                "entity_type": "SYSTEM",
                "description": ["The health care system"],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        ),
        Entity(
            id="ent-2a422318fc58c5302a5ba9365bcbc0be",
            page_content="INSURANCE COMPANIES",
            metadata={
                "entity_type": "ORG",
                "description": ["Insurance companies"],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        ),
        Entity(
            id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            page_content="HEALTH CARE PROVIDERS",
            metadata={
                "entity_type": "ORG",
                "description": ["Health care providers"],
                "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
            },
        ),
    ]

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
        Relation(
            source="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            target="ent-2a422318fc58c5302a5ba9365bcbc0be",
            properties={
                "description": "Health care providers work with insurance companies to provide medical services.",
                "weight": 7.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
    ]

    try:
        # First insert all nodes
        for node in nodes:
            await gdb.upsert_node(node)
        
        # Then insert all relations
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
    finally:
        await gdb.clean_up()


@pytest.mark.asyncio
async def test_merge_nodes():
    """Test batch node merging in FlaviusGDB"""
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )
    
    try:
        description1 = "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
        description2 = "The medical system in the United States is a complex network of hospitals, clinics, and other healthcare providers that provide medical care to the population."
        
        node1 = Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            page_content="UNITED STATES",
            metadata={
                "entity_type": "GEO",
                "description": [description1],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        )
        node2 = Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            page_content="UNITED STATES",
            metadata={
                "entity_type": "GEO",
                "description": [description2],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
        )
        
        await gdb.upsert_nodes([node1, node2])
        node = await gdb.query_node(node1.id)
        assert node.metadata.description[0] == description1
    finally:
        await gdb.clean_up()


@pytest.mark.asyncio
async def test_node_query():
    """Test individual node query in FlaviusGDB"""
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )
    
    try:
        node = Entity(
            id="ent-test-node-123",
            page_content="TEST ENTITY",
            metadata={
                "entity_type": "TEST",
                "description": ["This is a test entity"],
                "chunk_ids": ["chunk-test-123"],
            },
        )
        
        await gdb.upsert_node(node)
        queried_node = await gdb.query_node("ent-test-node-123")
        
        assert queried_node.id == "ent-test-node-123"
        assert queried_node.page_content == "TEST ENTITY"
        assert queried_node.metadata.entity_type == "TEST"
        assert queried_node.metadata.description == ["This is a test entity"]
        assert queried_node.metadata.chunk_ids == ["chunk-test-123"]
    finally:
        await gdb.clean_up()


@pytest.mark.asyncio
async def test_edge_query():
    """Test edge query in FlaviusGDB"""
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=ChatCompletion().complete,
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )
    
    try:
        # Create source and target nodes first
        source_node = Entity(
            id="ent-source-123",
            page_content="SOURCE NODE",
            metadata={
                "entity_type": "SOURCE",
                "description": ["Source entity"],
                "chunk_ids": ["chunk-source-123"],
            },
        )
        target_node = Entity(
            id="ent-target-456",
            page_content="TARGET NODE",
            metadata={
                "entity_type": "TARGET",
                "description": ["Target entity"],
                "chunk_ids": ["chunk-target-456"],
            },
        )
        
        await gdb.upsert_node(source_node)
        await gdb.upsert_node(target_node)
        
        # Create relation
        relation = Relation(
            source="ent-source-123",
            target="ent-target-456",
            properties={
                "relation_type": "TEST_RELATION",
                "weight": 5.0,
                "description": "Test relationship",
            },
        )
        
        await gdb.upsert_relation(relation)
        
        # Query the edge (note: this might need adjustment based on actual edge ID format)
        # For now, we'll test that the relation was created by querying one-hop neighbors
        neighbors, edges = await gdb.query_one_hop("ent-source-123")
        
        assert len(neighbors) == 1
        assert len(edges) == 1
        assert neighbors[0].id == "ent-target-456"
        assert edges[0].source == "ent-source-123"
        assert edges[0].target == "ent-target-456"
        assert edges[0].properties["relation_type"] == "TEST_RELATION"
    finally:
        await gdb.clean_up() 