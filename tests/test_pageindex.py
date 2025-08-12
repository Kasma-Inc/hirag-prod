import os
import json
import pytest
from pathlib import Path

from hirag_prod.pageindex import RemotePageIndex, ParseResponse

# Only a simple test, no pytest used

def print_hierarchy(response: ParseResponse):
    """Print the document hierarchy in a pretty tree format."""
    print("\n" + "="*60)
    print(f"📄 DOCUMENT HIERARCHY: {response.file_name}")
    print("="*60)
    
    # Print table of contents first
    if response.hierarchy.get("table_of_contents"):
        print("\n📋 TABLE OF CONTENTS:")
        print("-" * 30)
        print(response.hierarchy["table_of_contents"])
    
    # Print detailed hierarchy
    print("\n🌳 DETAILED HIERARCHY:")
    print("-" * 30)
    
    blocks = response.hierarchy.get("blocks", [])
    
    # Group blocks by hierarchy level for better visualization
    for block in blocks:
        level = block.get("hierarchy_level", 0)
        block_type = block.get("type", "unknown")
        block_id = block.get("id", "no-id")
        page_index = block.get("page_index", 0)
        
        # Create indentation based on hierarchy level
        indent = "  " * level
        
        # Type emoji mapping
        type_emojis = {
            "heading": "📌",
            "text": "📝",
            "table": "📊",
            "figure": "🖼️"
        }
        emoji = type_emojis.get(block_type, "❓")
        
        # Extract title from markdown (first line without #)
        markdown = block.get("markdown", "")
        title = ""
        if markdown:
            first_line = markdown.split('\n')[0]
            title = first_line.replace("#", "").strip()
            if not title and len(markdown.split('\n')) > 1:
                title = markdown.split('\n')[1].strip()[:50] + "..."
        
        if not title:
            title = f"Block {block_id}"
        
        # Print the hierarchy item
        print(f"{indent}{emoji} [{block_type.upper()}] {title}")
        print(f"{indent}   └─ Page: {page_index}, ID: {block_id}")
        
        # Show parent relationships
        parent_ids = block.get("parent_ids", [])
        if parent_ids:
            print(f"{indent}   └─ Parents: {', '.join(parent_ids)}")
    
    print("\n" + "="*60)
    print(f"📊 SUMMARY: {len(blocks)} blocks across {len(response.pages)} pages")
    
    # Count blocks by type
    type_counts = {}
    for block in blocks:
        block_type = block.get("type", "unknown")
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    
    print("📈 Block types:")
    for block_type, count in sorted(type_counts.items()):
        emoji = {"heading": "📌", "text": "📝", "table": "📊", "figure": "🖼️"}.get(block_type, "❓")
        print(f"   {emoji} {block_type}: {count}")
    
    print("="*60)

def test_process_pdf():
    api_key = os.getenv("PAGE_INDEX_API_KEY")
    remote_index = RemotePageIndex(api_key=api_key)

    # Test processing a sample PDF
    response = remote_index.process_pdf("tests/test_files/PosterLlama_2404.00995v3.pdf")
    assert response is not None
    assert response.status == "completed" 
    assert len(response.pages) > 0
    
    # Convert result to dict
    response_dict = response.to_dict()
    
    # Save as JSON
    output_path = Path("tests/pi") / f"{response.file_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(response_dict, f, default=str, indent=2, ensure_ascii=False)
        
    # Also Save a md for parsing results
    md_output_path = Path("tests/pi") / f"{response.file_name}.md"
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_output_path, "w") as f:
        f.write(response.markdown_document)

    # Then show the hierarchy
    print_hierarchy(response)
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 JSON saved to: {output_path}")
    print(f"📄 Markdown saved to: {md_output_path}")

if __name__ == "__main__":
    test_process_pdf()