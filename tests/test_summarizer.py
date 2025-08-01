import pytest

from hirag_prod._llm import create_chat_service
from hirag_prod.summarization import TrancatedAggregateSummarizer


@pytest.mark.asyncio
async def test_summarizer():
    summarizer = TrancatedAggregateSummarizer(
        extract_func=create_chat_service().complete,
        llm_model_name="gpt-4o-mini",
    )
    summary = await summarizer.summarize_entity(
        entity_name="test",
        descriptions=[
            "This is a unit test for the summarizer",
            "Unit test for the summaring entity descriptions",
        ],
    )
    assert summary is not None
