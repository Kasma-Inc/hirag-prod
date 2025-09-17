import os

import pytest
from dotenv import load_dotenv

from hirag_prod.reranker import ApiReranker, LocalReranker, create_reranker

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_api_reranker_integration():
    """Test ApiReranker with real VoyageAI API"""
    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    reranker = create_reranker("api")

    test_items = [
        {"text": "Machine learning algorithms for data analysis", "id": 1},
        {"text": "Deep learning neural networks and transformers", "id": 2},
        {"text": "Natural language processing techniques", "id": 3},
        {"text": "Computer vision and image recognition", "id": 4},
        {"text": "Statistical analysis and data mining techniques", "id": 5},
        {"text": "Artificial intelligence and neural network architectures", "id": 6},
        {"text": "Pattern recognition and feature extraction", "id": 7},
        {"text": "Reinforcement learning and decision trees", "id": 8},
    ]

    query = "deep learning neural networks"
    result = await reranker.rerank(query, test_items, topn=5)

    assert len(result) <= 5
    assert len(result) >= 3
    assert all("score" in item for item in result)
    assert all("text" in item for item in result)
    assert all("id" in item for item in result)

    # Results should be sorted by score (highest first)
    scores = [item["relevance_score"] for item in result]
    assert scores == sorted(scores, reverse=True)

    print(f"✅ API Reranker test passed: {len(result)} items reranked")
    print(f"Top result: {result[0]['text']} (score: {result[0]['score']})")


def test_local_reranker_validation():
    old_base_url = os.environ.get("RERANKER_MODEL_BASE_URL")
    old_auth_token = os.environ.get("RERANKER_MODEL_Authorization")

    try:
        if "RERANKER_MODEL_BASE_URL" in os.environ:
            del os.environ["RERANKER_MODEL_BASE_URL"]
        if "RERANKER_MODEL_Authorization" in os.environ:
            del os.environ["RERANKER_MODEL_Authorization"]

        with pytest.raises(ValueError, match="RERANKER_MODEL_BASE_URL"):
            create_reranker("local")

        os.environ["RERANKER_MODEL_BASE_URL"] = "http://localhost:8000"
        with pytest.raises(ValueError, match="RERANKER_MODEL_Authorization"):
            create_reranker("local")

        os.environ["RERANKER_MODEL_Authorization"] = "test_token"
        reranker = create_reranker("local")
        assert reranker.base_url == "http://localhost:8000"
        assert reranker.auth_token == "test_token"

    finally:
        if old_base_url is not None:
            os.environ["RERANKER_MODEL_BASE_URL"] = old_base_url
        if old_auth_token is not None:
            os.environ["RERANKER_MODEL_Authorization"] = old_auth_token


@pytest.mark.asyncio
async def test_reranker_factory():
    if os.getenv("VOYAGE_API_KEY"):
        api_reranker = create_reranker("api")
        assert isinstance(api_reranker, ApiReranker)

    old_vars = {}
    for var in ["RERANKER_MODEL_BASE_URL", "RERANKER_MODEL_Authorization"]:
        old_vars[var] = os.environ.get(var)
        os.environ[var] = "test_value"

    try:
        local_reranker = create_reranker("local")
        assert isinstance(local_reranker, LocalReranker)
    finally:
        for var, value in old_vars.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    with pytest.raises(ValueError, match="Unsupported reranker type"):
        create_reranker("invalid")


@pytest.mark.asyncio
async def test_local_reranker_integration():
    base_url = os.getenv("RERANKER_MODEL_BASE_URL")
    auth_token = os.getenv("RERANKER_MODEL_Authorization")

    if not base_url or not auth_token:
        pytest.skip("Local reranker environment variables not set")

    reranker = create_reranker("local")

    test_items = [
        {"text": "Machine learning algorithms for predictive analytics", "id": 1},
        {
            "text": "Deep learning neural networks with transformer architecture",
            "id": 2,
        },
        {"text": "Natural language processing and text understanding", "id": 3},
        {"text": "Computer vision for image classification tasks", "id": 4},
        {"text": "Statistical analysis for data science applications", "id": 5},
        {"text": "Artificial intelligence and machine learning frameworks", "id": 6},
    ]

    query = "deep learning neural networks"

    result = await reranker.rerank(query, test_items, topn=4)

    assert len(result) <= 4
    assert len(result) >= 2
    assert all("score" in item for item in result)
    assert all("text" in item for item in result)
    assert all("id" in item for item in result)

    scores = [item["relevance_score"] for item in result]
    assert scores == sorted(scores, reverse=True)

    print(f"✅ Local Reranker test passed: {len(result)} items reranked")
    print(f"Top result: {result[0]['text']} (score: {result[0]['score']})")


@pytest.mark.asyncio
async def test_empty_and_edge_cases():
    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    reranker = create_reranker("api")

    result = await reranker.rerank("test query", [], 5)
    assert result == []

    test_items = [{"text": "test", "id": 1}]
    result = await reranker.rerank("test", test_items, 0)
    assert result == []

    test_items_large = [
        {"text": "Python programming language", "id": 1},
        {"text": "JavaScript web development", "id": 2},
        {"text": "Java enterprise applications", "id": 3},
        {"text": "C++ systems programming", "id": 4},
        {"text": "Go microservices development", "id": 5},
    ]
    result = await reranker.rerank("Python programming", test_items_large, 10)
    assert len(result) <= len(test_items_large)
    assert len(result) >= 3

    print("✅ Edge cases test passed")


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        if os.getenv("VOYAGE_API_KEY"):
            print("Running API reranker integration test...")
            await test_api_reranker_integration()

            print("Running edge cases test...")
            await test_empty_and_edge_cases()

        print("Running validation tests...")
        test_local_reranker_validation()
        await test_reranker_factory()

        print("All tests completed!")

    asyncio.run(run_tests())
