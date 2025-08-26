import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel
from pydantic.fields import FieldInfo


def pydantic_to_pyarrow_type(
    field_type: type, field_info: FieldInfo = None
) -> pa.DataType:
    """Convert a Pydantic field type to PyArrow data type"""

    # String, Int, Float, Bool & Datetime
    if field_type is str:
        return pa.string()
    elif field_type is int:
        return pa.int32()
    elif field_type is float:
        return pa.float32()
    elif field_type is bool:
        return pa.bool_()
    elif field_type is datetime:
        return pa.timestamp("us")

    # Get origin and args for generic types
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Union types (including Optional)
    if origin is Union:
        # Check if it's Optional (Union with None)
        if len(args) == 2 and type(None) in args:
            # Get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return pydantic_to_pyarrow_type(non_none_type, field_info)
        # For other unions, use the first type
        return pydantic_to_pyarrow_type(args[0], field_info)

    # Handle List types
    if origin is list or origin is List:
        if args:
            item_type = args[0]
            # Check for nested List[List[float]]
            item_origin = get_origin(item_type)
            if item_origin is list or item_origin is List:
                item_args = get_args(item_type)
                if item_args and item_args[0] is float:
                    return pa.list_(pa.list_(pa.float32()))
                return pa.list_(pa.list_(pa.string()))  # Default nested list
            # Check for List[Dict[str, Any]] - store as JSON strings
            elif (
                item_origin is dict
                or hasattr(item_type, "__origin__")
                and str(item_type).startswith("dict")
            ):
                return pa.list_(pa.string())  # Store as list of JSON strings
            else:
                # Single level list
                item_pa_type = pydantic_to_pyarrow_type(item_type, field_info)
                return pa.list_(item_pa_type)
        return pa.list_(pa.string())  # Default to list of strings

    # Handle Dict types - store as JSON strings
    if origin is dict or (
        hasattr(field_type, "__origin__") and str(field_type).startswith("dict")
    ):
        return pa.string()  # Store complex dicts as JSON strings

    # Handle string representation of types (for older Python or complex types)
    if hasattr(field_type, "__str__"):
        type_str = str(field_type)
        if "Optional[str]" in type_str or "Union[str, None]" in type_str:
            return pa.string()
        elif "Optional[int]" in type_str or "Union[int, None]" in type_str:
            return pa.int32()
        elif "Optional[float]" in type_str or "Union[float, None]" in type_str:
            return pa.float32()
        elif "Optional[bool]" in type_str or "Union[bool, None]" in type_str:
            return pa.bool_()
        elif "Optional[datetime]" in type_str or "Union[datetime, None]" in type_str:
            return pa.timestamp("us")
        elif "List[str]" in type_str:
            return pa.list_(pa.string())
        elif "List[List[float]]" in type_str:
            return pa.list_(pa.list_(pa.float32()))
        elif "List[Dict" in type_str or "Dict[" in type_str:
            return pa.string()  # Store complex structures as JSON

    # Default fallback
    return pa.string()


def pydantic_model_to_pyarrow_schema(
    model_class: type[BaseModel],
    additional_fields: Dict[str, pa.DataType] = None,
    exclude_fields: List[str] = None,
) -> pa.Schema:
    """
    Convert a Pydantic model to PyArrow schema

    Args:
        model_class: The Pydantic model class
        additional_fields: Additional fields to add to the schema
        exclude_fields: Fields to exclude from the schema

    Returns:
        PyArrow schema
    """
    fields = []
    exclude_fields = exclude_fields or []

    # Process Pydantic model fields
    for field_name, field_info in model_class.model_fields.items():
        if field_name in exclude_fields:
            continue

        field_type = field_info.annotation
        pa_type = pydantic_to_pyarrow_type(field_type, field_info)
        fields.append(pa.field(field_name, pa_type))

    # Add additional fields
    if additional_fields:
        for field_name, pa_type in additional_fields.items():
            fields.append(pa.field(field_name, pa_type))

    return pa.schema(fields)


def get_chunk_schema(embedding_dimension: int) -> pa.Schema:
    """Generate chunk table schema from ChunkMetadata"""
    from .chunk import ChunkMetadata

    # Additional fields not in the Pydantic model
    additional_fields = {
        "text": pa.string(),
        "document_key": pa.string(),
        "vector": pa.list_(pa.float32(), embedding_dimension),
    }

    return pydantic_model_to_pyarrow_schema(
        ChunkMetadata, additional_fields=additional_fields
    )


def get_file_schema() -> pa.Schema:
    """Generate file table schema from FileMetadata"""
    from .file import FileMetadata

    # Additional fields for file-level storage that aren't in FileMetadata
    additional_fields = {
        "document_id": pa.string(),
        "page_content": pa.string(),
        "metadata_json": pa.string(),  # Additional metadata as JSON if needed
    }

    return pydantic_model_to_pyarrow_schema(
        FileMetadata, additional_fields=additional_fields
    )
