import asyncio
import logging

from hirag_prod import HiRAG
from hirag_prod.configs.cli_options import CliOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv(".env", override=True)


# To avoid errors with not able to read .env, run in root folder chatbot/
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
    print(f"Files in knowledge base: {[fn.get('fileName') for fn in file_list]}")
    print(f"{'='*60}\n")

    file_ids = [file["id"] for file in file_list]

    file_details = await index.query_file_by_ids(
        workspace_id=workspace_id,
        knowledge_base_id=knowledge_base_id,
        file_ids=file_ids,
    )

    print(
        f"Query by file id results: {[{'fileName': fd.get('fileName'), 'ToC': fd.get('tableOfContents')} for fd in file_details]}"
    )
    print(f"{'='*60}\n")

    # Randomly get tableOfContents that contain hierarchy.blocks.id
    header_ids = []
    for file in file_details:
        toc = file.get("tableOfContents", [])
        if toc:
            blocks = toc.get("hierarchy", {}).get("blocks", [])
            for block in blocks:
                header_id = block.get("id")
                if header_id:
                    header_ids.append(header_id)

    print(f"Header IDs extracted: {header_ids}")
    print(f"{'='*60}\n")

    chunks = await index.query_by_headers(
        workspace_id=workspace_id,
        knowledge_base_id=knowledge_base_id,
        headers=header_ids,
    )

    print(f"Query by header IDs results: {chunks}")
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
