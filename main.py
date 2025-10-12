# This is a quickstart script for the HiRAG system.
import asyncio
import json
import logging
import os
from datetime import datetime

from hirag_prod import HiRAG
from hirag_prod.configs.cli_options import CliOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv(".env", override=True)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and other non-serializable objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle other non-serializable types by converting to string
        try:
            return str(obj)
        except:
            return repr(obj)


def get_test(id: str):
    if id == "wiki_subcorpus" or id == "1":
        document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
        content_type = "text/plain"
        document_meta = {
            "type": "txt",
            "fileName": "2wiki_subcorpus.txt",
            "uri": document_path,
            "private": False,
            "createdBy": "wiki_subcorpus_test",
            "updatedBy": "wiki_subcorpus_test",
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
            "createdBy": "small_pdf_test",
            "updatedBy": "small_pdf_test",
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
            "createdBy": "U.S.Health_test",
            "updatedBy": "U.S.Health_test",
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
            "createdBy": "itinerary_test",
            "updatedBy": "itinerary_test",
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
            "createdBy": "wiki_article_test",
            "updatedBy": "wiki_article_test",
        }
        query = "What is the cause of Odisha train accident in 2023?"
        return document_path, content_type, document_meta, query
    elif id == "oss: interim" or id == "6":
        document_path = f"oss://graxy-dev/ofnil/tmp/test/2023-24_INTERIM_notes_to_the_condensed_consolidated_interim_financial_information.pdf"
        content_type = "application/pdf"
        document_meta = {
            "type": "pdf",
            "fileName": "2023-24_INTERIM_notes_to_the_condensed_consolidated_interim_financial_information.pdf",
            "uri": document_path,
            "private": False,
            "createdBy": "interim_test",
            "updatedBy": "interim_test",
        }
        query = "What are the key financial highlights for 2023-24?"
        return document_path, content_type, document_meta, query

    # Warning this test takes a extremely long time to process
    elif id == "translation_test" or id == "7":
        document_path_base = f"s3://monkeyocr/test/input/test_pdf/fire_dept/"
        filenames = [
            # "Cap 123 Consolidated version for the Whole Chapter (01-09-2023) (English).pdf",
            # "Cap 123 Consolidated version for the Whole Chapter (01-09-2023) (Traditional Chinese).pdf",
            "Cap 502 Consolidated version for the Whole Chapter (13-12-2024) (English).pdf",
            "Cap 502 Consolidated version for the Whole Chapter (13-12-2024) (Traditional Chinese).pdf",
            "Cap 95 Consolidated version for the Whole Chapter (15-10-2021) (English).pdf",
            "Cap 95 Consolidated version for the Whole Chapter (15-10-2021) (Traditional Chinese).pdf",
            "Cap 95B Consolidated version for the Whole Chapter (01-11-2023) (English).pdf",
            "Cap 95B Consolidated version for the Whole Chapter (01-11-2023) (Traditional Chinese).pdf",
        ]
        document_paths = [document_path_base + fn for fn in filenames]
        content_type = "application/pdf"
        document_metas = [
            {
                "type": "pdf",
                "fileName": fn,
                "uri": document_path_base + fn,
                "private": False,
                "createdBy": "translation_test",
                "updatedBy": "translation_test",
            }
            for fn in filenames
        ]
        query = [
            "What is the short title of Cap. 95?",
            "According to Cap. 95, what are the duties of the Fire Services Department under Section 7?",
            "In Cap. 95, what powers does the Director have under Section 9 regarding fire hazards?",
            "What is the punishment for subordinate officers and members of other ranks for offences against discipline in Cap. 95's Third Schedule?",
            "In Cap. 95 Part IV, what does the Fire Services Department Welfare Fund consist of under Section 19B?",
            "What are the ranks in the Fire Services Department as per the Sixth Schedule of Cap. 95?",
            "In Cap. 123, what is required for the commencement of building works under Section 14?",
            "According to Cap. 123 Section 16, on what grounds may approval or consent be refused?",
            "What is a Closure Order under Section 27 of Cap. 123?",
            "In Cap. 123, what are the powers of the Building Authority under Section 22?",
            "What is the citation for Cap. 95B regulations?",
            "Under Cap. 95B Regulation 3, who approves portable equipment and what is the appeal process?",
            "In Cap. 95B, what is the definition of 'portable equipment'?",
            "According to Cap. 95B Regulation 6, who is allowed to install fire service installations or equipment?",
            "What are the duties of owners under Cap. 95B Regulation 8 regarding maintenance and inspection?",
            "In Cap. 95B Regulation 9, what must a certificate issued by a registered contractor include?",
            "What is the penalty under Cap. 95B Regulation 12 for contraventions?",
        ]
        return document_paths, content_type, document_metas, query

    elif id == "PiT_test" or id == "8":
        document_path_base = f"s3://monkeyocr/test/input/test_pdf/fire_dept/"
        filenames = [
            "Cap 95B Consolidated version for the Whole Chapter (01-11-2023) (English).pdf",
            "Cap 95B Consolidated version for the Whole Chapter (01-09-2021) (English).pdf",
            "Cap 95B Consolidated version for the Whole Chapter (19-09-2019) (English).pdf",
            "Cap 95B PDF (01-01-2004) (English).pdf",
        ]
        document_paths = [document_path_base + fn for fn in filenames]
        content_type = "application/pdf"
        document_metas = [
            {
                "type": "pdf",
                "fileName": fn,
                "uri": document_path_base + fn,
                "private": False,
                "createdBy": "PiT_test",
                "updatedBy": "PiT_test",
            }
            for fn in filenames
        ]
        query = [
            "What are the definitions of key terms like 'portable equipment', 'registered contractor', and 'stand-alone fire detector' in Cap 95B?",
            "What are the rules for approval, listing, and publication of portable equipment under regulations 3 and 4 of Cap 95B?",
            "What prohibitions apply to the sale or supply of portable equipment under regulation 5 of Cap 95B?",
            "Who is authorized to install, maintain, inspect, or repair fire service installations under regulations 6 and 7 of Cap 95B, and what exemptions apply?",
            "What duties do owners have for maintaining and inspecting fire service installations under regulation 8 of Cap 95B, including any exceptions?",
        ]
        return document_paths, content_type, document_metas, query

    elif id == "png" or id == "9":
        document_path = f"oss://graxy-dev/ofnil/tmp/user/2e9946d0-eb85-4508-b499-feda899d0314/aliyun-截图.png"
        content_type = "multimodal/image"
        document_meta = {
            "type": "png",
            "fileName": "aliyun-截图.png",
            "uri": document_path,
            "private": False,
            "createdBy": "png_test",
            "updatedBy": "png_test",
        }
        query = "What is depicted in the image?"
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
            "createdBy": "small_pdf_test",
            "updatedBy": "small_pdf_test",
        }
        query = "Machine learning in detection"
        return document_path, content_type, document_meta, query


def save_chunks_to_json(chunks, query, filename="logs/retrieved_chunks.json"):
    """
    Save chunks to a JSON file by appending to existing data.
    """

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    terms_to_save = ["text", "fileName"]

    # Filter chunks to only include specified terms
    chunks = [
        {k: v for k, v in chunk.items() if k in terms_to_save} for chunk in chunks
    ]

    for i in range(len(chunks)):
        chunks[i]["saveId"] = f"{i+1}"

    # Prepare the data entry
    entry = {"query": query, "chunks": chunks}

    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
    else:
        data = []

    # Append new entry
    data.append(entry)

    # Save back to file using custom encoder
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

    print(f"Saved {len(chunks)} chunks to {filename}")


def print_chunks_user_friendly(chunks):
    """
    Print chunks in a more user-friendly format.
    """

    if not chunks:
        print("No chunks found.")
        return

    print(f"Found {len(chunks)} relevant chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"📄 Chunk {i}")
        print(f"   Source: {chunk.get('fileName', 'Unknown')}")
        print(f"   Chunk ID: {chunk.get('documentKey', 'Unknown')}")
        if "relevance_score" in chunk:
            print(f"   Relevance Score: {chunk['relevance_score']:.4f}")
        if "pagerank_score" in chunk:
            print(f"   PageRank Score: {chunk['pagerank_score']:.4f}")

        # Clean up the text content
        text = chunk.get("text", "")

        # For non-table content, clean up and display
        clean_text = text.strip()
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        print(f"   Content: {clean_text}")

        print(f"   Last Updated: {chunk.get('updatedAt', 'Unknown')}")
        print("   " + "─" * 50)
        print()


async def index(test_id="2", summary=True, save_json=False, no_insert=False):
    index = await HiRAG.create()

    await index.set_language("en")  # en | cn

    document_path, content_type, document_meta, query = get_test(test_id)

    if isinstance(document_path, str):
        document_path = [document_path]
        document_meta = [document_meta]

    if not no_insert:
        for dp, dm in zip(document_path, document_meta):
            await index.insert_to_kb(
                file_id="test_id",
                document_path=dp,
                content_type=content_type,
                document_meta=dm,
                workspace_id="test_workspace",
                knowledge_base_id="test_pg",
            )
    else:
        print("Skipping document insertion (--no-insert flag enabled)")

    if isinstance(query, str):
        query = [query]

    if isinstance(query, list):
        for q in query:
            ret = await index.query(
                query=q,
                summary=False,
                workspace_id="test_workspace",
                knowledge_base_id="test_pg",
                threshold=0.0,
                strategy="raw",
                translation=["en", "zh"],
            )

            print("———————————————————— Raw Results ————————————————————\n")
            print_chunks_user_friendly(ret["chunks"])

            filtered_ret = await index.apply_strategy_to_chunks(
                chunks_dict=ret,
                strategy="hybrid",
                filter_by_clustering=True,
                workspace_id="test_workspace",
                knowledge_base_id="test_pg",
            )

            print("———————————————————— Query ————————————————————\n")
            print(f"Query: {q}\n")
            print("———————————————————— Chunks ————————————————————\n")
            print_chunks_user_friendly(filtered_ret["chunks"])

            filtered_ret["summary"] = await index.generate_summary(
                workspace_id="test_workspace",
                knowledge_base_id="test_pg",
                chunks=filtered_ret["chunks"],
                query=q,
            )

            # Save chunks to JSON file if enabled
            if save_json:
                save_chunks_to_json(
                    filtered_ret["chunks"], q, f"logs/{test_id}_retrieved_chunks.json"
                )

            if summary:
                print("———————————————————— Summary ————————————————————\n")
                print(filtered_ret["summary"])
            print("\n\n")


def main():
    # Use the integrated CLI options instead of a separate argument parser
    cli_options = CliOptions()

    # Print available tests for user reference
    print(f"\nRunning test: {cli_options.test}")
    print(f"Summary: {cli_options.summary}")
    print(f"Save JSON: {cli_options.save_json}")
    print(f"No Insert: {cli_options.no_insert}\n")

    asyncio.run(
        index(
            cli_options.test,
            cli_options.summary,
            cli_options.save_json,
            cli_options.no_insert,
        )
    )


if __name__ == "__main__":
    main()
