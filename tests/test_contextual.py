import asyncio
import json
import os
import sys

sys.path.append("src")
from hirag_prod.contextual.client import ContextualClient


async def parse_doc(file_path: str, output_path: str):
    """Test the contextual client by parsing a document."""
    client = ContextualClient()

    print(f"Parsing document: {file_path}")

    res = await client.parse_document_sync(file_path)
    # res = await client.wait_for_parse_completion("ad38d2ce-9a0e-49d8-9d21-454f6de51519")

    print("✅ Parse completed successfully!")

    # Save the res as a pretty JSON
    with open(output_path, "w", encoding="utf-8") as f:
        if hasattr(res, "model_dump"):
            json.dump(res.model_dump(), f, ensure_ascii=False, indent=4)
        else:
            json.dump(res, f, ensure_ascii=False, indent=4)

    # Extract markdown content if available
    md_content = ""
    hierarchy_content = ""

    if hasattr(res, "markdown_document"):
        md_content = res.markdown_document
        if hasattr(res, "document_metadata") and hasattr(
            res.document_metadata, "hierarchy"
        ):
            hierarchy_content = getattr(
                res.document_metadata.hierarchy, "table_of_contents", ""
            )

    if md_content:
        with open(output_path.replace(".json", ".md"), "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved markdown to: {output_path.replace('.json', '.md')}")

    if hierarchy_content:
        with open(
            output_path.replace(".json", "_hierarchy.md"), "w", encoding="utf-8"
        ) as f_hierarchy:
            f_hierarchy.write(hierarchy_content)
        print(
            f"Saved table of contents to: {output_path.replace('.json', '_hierarchy.md')}"
        )


if __name__ == "__main__":
    if not os.getenv("CONTEXTUAL_API_KEY"):
        print("Error: CONTEXTUAL_API_KEY environment variable is not set.")
        print("Please set your API key: export CONTEXTUAL_API_KEY='your_api_key'")
        sys.exit(1)

    asyncio.run(
        parse_doc(
            "tests/test_files/PGhandbook2025.pdf", "tests/output/PGhandbook2025.json"
        )
    )
