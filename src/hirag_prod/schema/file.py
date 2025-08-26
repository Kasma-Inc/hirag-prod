from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from pydantic import BaseModel


class FileMetadata(BaseModel):
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
    filename: Optional[str] = None
    page_number: Optional[int] = None
    # The uri of the file
    # When the file is a local file, the uri is the path to the file
    # When the file is a remote file, the uri is the url of the file
    uri: Optional[str] = None
    # Whether the file is private
    private: Optional[bool] = None
    uploaded_at: Optional[datetime] = None
    knowledge_base_id: Optional[str] = (
        None  # The id of the knowledge base that the file is from
    )
    workspace_id: Optional[str] = None

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
