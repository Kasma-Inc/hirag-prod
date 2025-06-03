import asyncio
import logging

from hirag_prod import HiRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def index():
    index = await HiRAG.create()
    document_path = f"tests/Guide-to-U.S.-Healthcare-System.pdf"
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
    print(await index.query_all("tell me about bitcoin"))


if __name__ == "__main__":
    asyncio.run(index())
