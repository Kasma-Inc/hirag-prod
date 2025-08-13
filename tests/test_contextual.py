import asyncio
import json
import os
import sys

sys.path.append("src")
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod.contextual.client import ContextualClient


def save_file(output_dir, res):
    # Get the file name from response and construct output paths
    file_name = res.get("file_name", "unknown_file")
    # Remove extension and use as base name
    base_name = os.path.splitext(file_name)[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct output paths using the output directory and retrieved file name
    output_path = os.path.join(output_dir, f"{base_name}.json")

    # Save the res as a pretty JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print(f"Saved JSON to: {output_path}")

    # Extract markdown content if available
    md_content = ""
    hierarchy_content = ""

    # Now res is a dictionary, so we access fields directly
    if "markdown_document" in res:
        md_content = res["markdown_document"]

    # Handle both API response format and database cached format
    if "table_of_content" in res:
        # Database cached format (from our storage_util.py)
        hierarchy_content = res["table_of_content"]
    elif "document_metadata" in res and res["document_metadata"]:
        # API response format (direct from Contextual AI)
        document_metadata = res["document_metadata"]
        if "hierarchy" in document_metadata and document_metadata["hierarchy"]:
            hierarchy = document_metadata["hierarchy"]
            if "table_of_contents" in hierarchy:
                hierarchy_content = hierarchy["table_of_contents"]

    if md_content:
        md_output_path = os.path.join(output_dir, f"{base_name}.md")
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved markdown to: {md_output_path}")

    if hierarchy_content:
        hierarchy_output_path = os.path.join(output_dir, f"{base_name}_hierarchy.md")
        with open(hierarchy_output_path, "w", encoding="utf-8") as f_hierarchy:
            f_hierarchy.write(hierarchy_content)
        print(f"Saved table of contents to: {hierarchy_output_path}")


async def get_doc(doc_id, output_dir):
    """Test the contextual client by parsing a document."""
    client = ContextualClient()

    # Create database engine and session if connection string is available
    db_url = os.getenv("POSTGRES_URL_NO_SSL_DEV")
    if db_url:
        engine = client.create_db_engine(db_url)
        async with AsyncSession(engine) as session:
            res = await client.get_parse_results(doc_id, session=session)
    else:
        res = await client.get_parse_results(doc_id)

    save_file(output_dir, res)


async def parse_doc(file_path: str, output_dir: str):
    """Test the contextual client by parsing a document."""
    client = ContextualClient()

    print(f"Document: {file_path}")

    # Create database engine and session if connection string is available
    db_url = os.getenv("POSTGRES_URL_NO_SSL_DEV")
    if db_url:
        engine = client.create_db_engine(db_url)
        async with AsyncSession(engine) as session:
            res = await client.parse_document_sync(file_path, session=session)
    else:
        res = await client.parse_document_sync(file_path)

    print("✅ Completed successfully!")

    save_file(output_dir, res)


if __name__ == "__main__":
    if not os.getenv("CONTEXTUAL_API_KEY"):
        print("Error: CONTEXTUAL_API_KEY environment variable is not set.")
        print("Please set your API key: export CONTEXTUAL_API_KEY='your_api_key'")
        sys.exit(1)

    # asyncio.run(parse_doc("tests/test_files/COMP4471_Fall_24-25.pdf", "tests/output/"))
    asyncio.run(get_doc("49968c8b-cdb6-48d6-b15f-8b5549ec1452", "tests/output/"))
