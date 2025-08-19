import os

import pytest

from hirag_prod import HiRAG


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skip the test for it is time-consuming")
async def test_index():
    index = await HiRAG.create()
    document_path = (
        f"{os.path.dirname(__file__)}/test_files/Guide-to-U.S.-Healthcare-System.pdf"
    )
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
    )
