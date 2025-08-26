# Test script for testing dynamic schema generation from Pydantic models
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
from pydantic import BaseModel


def test_pydantic_to_pyarrow():
    """Test the schema conversion logic"""
    print("Testing enhanced Pydantic to PyArrow schema conversion...\n")

    # Enhanced test class matching FileMetadata
    class EnhancedFileMetadata(BaseModel):
        filename: Optional[str] = None
        page_number: Optional[int] = None
        uri: Optional[str] = None
        private: Optional[bool] = None
        uploaded_at: Optional[datetime] = None
        headers: Optional[List[str]] = None
        bbox: Optional[List[List[float]]] = None
        # New fields
        markdown_content: Optional[str] = None
        table_of_contents: Optional[List[Dict[str, Any]]] = None
        page_count: Optional[int] = None
        processing_status: Optional[str] = None

    # Test manual schema creation
    schema_fields = []
    for field_name, field_info in EnhancedFileMetadata.model_fields.items():
        field_type = field_info.annotation
        print(f"Field: {field_name}")
        print(f"  Type: {field_type}")

        # Enhanced type mapping
        if field_type == Optional[str] or field_type == str:
            pa_type = pa.string()
        elif field_type == Optional[int] or field_type == int:
            pa_type = pa.int32()
        elif field_type == Optional[bool] or field_type == bool:
            pa_type = pa.bool_()
        elif field_type == Optional[datetime] or field_type == datetime:
            pa_type = pa.timestamp("us")
        elif field_type == Optional[List[str]] or field_type == List[str]:
            pa_type = pa.list_(pa.string())
        elif (
            field_type == Optional[List[List[float]]] or field_type == List[List[float]]
        ):
            pa_type = pa.list_(pa.list_(pa.float32()))
        elif "List[Dict" in str(field_type) or "Optional[List[Dict" in str(field_type):
            pa_type = pa.string()  # Store as JSON string
        else:
            pa_type = pa.string()  # fallback

        print(f"  PyArrow: {pa_type}")
        schema_fields.append(pa.field(field_name, pa_type))
        print()

    # Add additional fields
    schema_fields.extend(
        [
            pa.field("document_id", pa.string()),
            pa.field("page_content", pa.string()),
            pa.field("metadata_json", pa.string()),
        ]
    )

    schema = pa.schema(schema_fields)

    print(f"✅ Generated enhanced file schema with {len(schema)} fields:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    print(f"\n📊 Schema Summary:")
    print(f"  - Pydantic fields: {len(EnhancedFileMetadata.model_fields)}")
    print(f"  - Additional fields: 3")
    print(f"  - Total schema fields: {len(schema)}")

    print("\n🎉 Enhanced schema generation test completed successfully!")


if __name__ == "__main__":
    test_pydantic_to_pyarrow()
