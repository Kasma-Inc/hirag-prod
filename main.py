# This is a quickstart script for the HiRAG system.
import asyncio
import logging

from hirag_prod import HiRAG
from hirag_prod.configs.cli_options import CliOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv(".env", override=True)


def get_test(id: str):
    if id == "wiki_subcorpus" or id == "1":
        document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
        content_type = "text/plain"
        document_meta = {
            "type": "txt",
            "fileName": "2wiki_subcorpus.txt",
            "uri": document_path,
            "private": False,
        }
        query = "When did Lothair II's mother die?"
        return document_path, content_type, document_meta, query
    elif id == "s3: small_pdf" or id == "2":
        document_path = f"s3://monkeyocr/test/input/test_pdf/small.pdf"
        content_type = "application/pdf"
        document_meta = {
            "type": "pdf",
            "fileName": "small.pdf",
            "uri": document_path,
            "private": False,
        }
        query = "Machine learning in detection"
        return document_path, content_type, document_meta, query
    elif id == "oss: U.S.Health" or id == "3":
        document_path = (
            f"oss://graxy-dev/ofnil/tmp/test/Guide-to-U.S.-Healthcare-System.pdf"
        )
        content_type = "application/pdf"
        document_meta = {
            "type": "pdf",
            "fileName": "Guide-to-U.S.-Healthcare-System.pdf",
            "uri": document_path,
            "private": False,
        }
        query = "What is the structure of the U.S. healthcare system?"
        return document_path, content_type, document_meta, query
    elif id == "md-itinerary" or id == "4":
        document_path = f"s3://monkeyocr/test/input/test_md/Ideal holiday itinerary.md"
        content_type = "text/markdown"
        document_meta = {
            "type": "md",
            "fileName": "Ideal holiday itinerary.md",
            "uri": document_path,
            "private": False,
        }
        query = "What are the focuses of the holiday plan?"
        return document_path, content_type, document_meta, query
    elif id == "md-wiki" or id == "5":
        document_path = f"tests/test_files/fresh_wiki_article.md"
        content_type = "text/markdown"
        document_meta = {
            "type": "md",
            "fileName": "fresh_wiki_article.md",
            "uri": document_path,
            "private": False,
        }
        query = "What is the cause of Odisha train accident in 2023?"
        return document_path, content_type, document_meta, query
    else:
        # Default to small.pdf if test not found
        document_path = f"s3://monkeyocr/test/input/test_pdf/small.pdf"
        content_type = "application/pdf"
        document_meta = {
            "type": "pdf",
            "fileName": "small.pdf",
            "uri": document_path,
            "private": False,
        }
        query = "Machine learning in detection"
        return document_path, content_type, document_meta, query


async def index(test_id="2"):
    index = await HiRAG.create()

    await index.set_language("en")  # en | cn

    document_path, content_type, document_meta, query = get_test(test_id)

    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        workspace_id="test_workspace",
        knowledge_base_id="test_pg",
        loader_type="dots_ocr",
        overwrite=True,
    )

    ret = await index.query(
        query=query,
        summary=True,
        workspace_id="test_workspace",
        knowledge_base_id="test_pg",
    )

    print("———————————————————— Chunks ————————————————————\n")
    print(ret["chunks"])
    print("\n\n———————————————————— Summary ————————————————————\n")
    print(ret["summary"])


def main():
    # Use the integrated CLI options instead of a separate argument parser
    cli_options = CliOptions()

    # Print available tests for user reference
    print("Available tests:")
    print("  1 / wiki_subcorpus - 2wiki subcorpus text file")
    print("  2 / s3: small_pdf - Small PDF from S3 (default)")
    print("  3 / oss: U.S.Health - U.S. Healthcare guide PDF")
    print("  4 / md-itinerary - Holiday itinerary markdown")
    print("  5 / md-wiki - Wikipedia article markdown")
    print(f"\nRunning test: {cli_options.test}\n")

    asyncio.run(index(cli_options.test))


if __name__ == "__main__":
    main()
