"""
Test script for querying by document keys in HiRAG system.
Similar to main.py but focused on testing query_by_keys functionality.
"""

import asyncio
import logging

from hirag_prod import HiRAG
from hirag_prod.configs.cli_options import CliOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv(".env", override=True)


async def test_query_by_keys(save_json=False):
    """
    Test the query_by_keys functionality of HiRAG.
    """

    # Initialize HiRAG
    index = await HiRAG.create()
    await index.set_language("en")  # en | cn

    workspace_id = "test_workspace"
    knowledge_base_id = "test_pg"

    file_list = await index.list_kb_files(
        workspace_id=workspace_id,
        knowledge_base_id=knowledge_base_id,
    )
    print(f"Files in knowledge base: {file_list}")

    file_ids = [file["id"] for file in file_list]

    file_details = await index.query_file_by_ids(
        workspace_id=workspace_id,
        knowledge_base_id=knowledge_base_id,
        file_ids=file_ids,
    )

    print(f"Query by file id results: {file_details}")
    print(f"{'='*60}\n")


def main():
    """
    Main entry point for the test script.
    """
    # Use CLI options for configuration
    cli_options = CliOptions()

    print("\n" + "=" * 60)
    print("Query By Keys Test Configuration")
    print("=" * 60)
    print(f"Save JSON: {cli_options.save_json}")
    print("=" * 60)

    asyncio.run(
        test_query_by_keys(
            save_json=cli_options.save_json,
        )
    )


if __name__ == "__main__":
    main()
