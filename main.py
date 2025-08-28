# This is a quickstart script for the HiRAG system.
import asyncio
import logging
import os

from hirag_prod import HiRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv("/chatbot/.env", override=True)

# Default Database Configuration
VECTOR_DB_PATH = os.getenv("POSTGRES_URL_NO_SSL_DEV")
GRAPH_DB_PATH = "kb/hirag.gpickle"


def get_test(id: str):
    if id == "wiki_subcorpus":
        document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
        content_type = "text/plain"
        document_meta = {
            "type": "txt",
            "filename": "2wiki_subcorpus.txt",
            "uri": document_path,
            "private": False,
        }
        return document_path, content_type, document_meta
    elif id == "small_pdf":
        document_path = f"s3://monkeyocr/test/input/test_pdf/small.pdf"
        content_type = "application/pdf"
        document_meta = {
            "type": "pdf",
            "filename": "small.pdf",
            "uri": document_path,
            "private": False,
        }
        return document_path, content_type, document_meta


async def index():
    index = await HiRAG.create(
        vector_db_path=VECTOR_DB_PATH, graph_db_path=GRAPH_DB_PATH, vdb_type="pgvector"
    )

    await index.set_language("en")  # en | cn

    document_path, content_type, document_meta = get_test("small_pdf")

    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        workspace_id="test_workspace",
        knowledge_base_id="test_pg",
        loader_type="dots_ocr",
    )

    ret = await index.query(
        "Machine Learning Detection Methods?",
        summary=True,
        workspace_id="test_workspace",
        knowledge_base_id="test_pg",
    )

    print("———————————————————— Chunks ————————————————————\n")
    print(ret["chunks"])
    print("\n\n———————————————————— Summary ————————————————————\n")
    print(ret["summary"])


if __name__ == "__main__":
    asyncio.run(index())
