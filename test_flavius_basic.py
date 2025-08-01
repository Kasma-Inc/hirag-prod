#!/usr/bin/env python3
"""
Simplified Flavius Test Script
"""

import asyncio
import os
from hirag_prod.storage.flavius import FlaviusGDB
from hirag_prod.schema import Entity, Relation

async def test_basic_functionality():
    """Test basic functionality"""
    print("🚀 Starting Flavius basic functionality test...")
    
    # Set environment variable
    os.environ['EMBEDDING_DIMENSION'] = '1536'
    
    # Create FlaviusGDB instance
    gdb = FlaviusGDB.create(
        path="test_flavius.json",
        llm_func=lambda x: "test",
        llm_model_name="gpt-4o-mini",
        flavius_url="http://localhost:30000",
    )
    
    try:
        print(f"✅ FlaviusGDB created successfully, namespace: {gdb.namespace}")
        
        # Test node insertion and query
        print("\n📝 Testing node insertion and query...")
        node = Entity(
            id="test-node-001",
            page_content="Test Node",
            metadata={
                "entity_type": "TEST",
                "description": ["This is a test node"],
                "chunk_ids": ["chunk-001"],
            },
        )
        
        await gdb.upsert_node(node)
        print("✅ Node inserted successfully")
        
        queried_node = await gdb.query_node("test-node-001")
        print(f"✅ Node queried successfully: {queried_node.page_content}")
        assert queried_node.page_content == "Test Node"
        
        # Test relation insertion and query
        print("\n🔗 Testing relation insertion and query...")
        relation = Relation(
            source="test-node-001",
            target="test-node-002",
            properties={
                "relation_type": "TEST_RELATION",
                "weight": 1.0,
                "description": "Test relation",
            },
        )
        
        await gdb.upsert_relation(relation)
        print("✅ Relation inserted successfully")
        
        # Test one-hop query
        neighbors, edges = await gdb.query_one_hop("test-node-001")
        print(f"✅ One-hop query successful: found {len(neighbors)} neighbors, {len(edges)} edges")
        
        # Test dump and load
        print("\n💾 Testing dump and load...")
        await gdb.dump()
        metadata = FlaviusGDB.load("test_flavius.json")
        print(f"✅ Dump/load successful: {metadata}")
        
        print("\n🎉 All basic functionality tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await gdb.clean_up()
        print("🧹 Cleanup complete")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())