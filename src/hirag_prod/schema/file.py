from datetime import datetime
from typing import Any, Dict, Literal, Optional

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
    table_of_contents: Optional[str] = None  # Structured TOC in string format

    def to_dict(self, camelize: bool = False) -> Dict[str, Any]:
        if camelize:
            return {
                "filename": self.filename,
                "uri": self.uri,
                "private": self.private,
                "knowledgeBaseId": self.knowledge_base_id,
                "workspaceId": self.workspace_id,
                "type": self.type,
                "pageNumber": self.page_number,
                "uploadedAt": self.uploaded_at,
                "markdownContent": self.markdown_content,
                "tableOfContents": self.table_of_contents,
            }
        return {
            "filename": self.filename,
            "uri": self.uri,
            "private": self.private,
            "knowledge_base_id": self.knowledge_base_id,
            "workspace_id": self.workspace_id,
            "type": self.type,
            "page_number": self.page_number,
            "uploaded_at": self.uploaded_at,
            "markdown_content": self.markdown_content,
            "table_of_contents": self.table_of_contents,
        }


class File(Document, BaseModel):
    # "file-mdhash(filename)"
    id: str
    # The content of the file
    page_content: str
    # The metadata of the file
    metadata: FileMetadata
