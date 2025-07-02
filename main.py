# This is a quickstart script for the HiRAG system.
import asyncio
import logging

from hirag_prod import HiRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv("/chatbot/.env")


async def index():
    index = await HiRAG.create()
    document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
    content_type = "text/plain"
    document_meta = {
        "type": "txt",
        "filename": "2wiki_subcorpus.txt",
        "uri": document_path,
        "private": False,
    }
    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
    )
    print(await index.query_all("When did Lothair Ii's mother die?"))


if __name__ == "__main__":
    asyncio.run(index())
