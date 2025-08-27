from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from pydantic import BaseModel


class FileMetadata(BaseModel):
    # Required fields
    filename: str
    # The uri of the file
    # When the file is a local file, the uri is the path to the file
    # When the file is a remote file, the uri is the url of the file
    uri: str
    # Whether the file is private
    private: bool
    knowledge_base_id: str
    workspace_id: str

    # Optional fields
    type: Optional[
        Literal[
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "jpg",
            "png",
            "zip",
            "txt",
            "csv",
            "text",
            "tsv",
            "html",
            "md",
        ]
    ] = None
    page_number: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    # New fields for enhanced file storage
    markdown_content: Optional[str] = None  # Full markdown representation
    table_of_contents: Optional[List[Dict[str, Any]]] = None  # Structured TOC


class File(Document, BaseModel):
    # "file-mdhash(filename)"
    id: str
    # The content of the file
    page_content: str
    # The metadata of the file
    metadata: FileMetadata
