# This is a quickstart script for the HiRAG system.
import asyncio
import logging

from hirag_prod import HiRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv("/chatbot/.env", override=True)

# Default Database Configuration
VECTOR_DB_PATH = "kb/hirag.db"
GRAPH_DB_PATH = "kb/hirag.gpickle"


async def index():
    index = await HiRAG.create(
        vector_db_path=VECTOR_DB_PATH, graph_db_path=GRAPH_DB_PATH
    )

    await index.set_language("en")  # en | cn

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
        workspace_id="test_workspace",
        knowledge_base_id="test_kb",
    )

    ret = await index.query(
        "When did Lothair Ii's mother die?",
        summary=True,
        workspace_id="test_workspace",
        knowledge_base_id="test_kb",
    )

    print("———————————————————— Chunks ————————————————————\n")
    print(ret["chunks"])
    print("\n\n———————————————————— Summary ————————————————————\n")
    print(ret["summary"])


if __name__ == "__main__":
    asyncio.run(index())
