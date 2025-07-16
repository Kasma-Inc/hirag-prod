from langchain_core.documents import Document
from pydantic import BaseModel

from .file import FileMetadata


class ChunkMetadata(FileMetadata):
    # The index of the chunk in the document
    chunk_idx: int
    # The id of the document that the chunk is from
    document_id: str
    # The type of the chunk
    chunk_type: str


class Chunk(Document, BaseModel):
    # "chunk-mdhash(chunk_content)"
    id: str
    # The content of the chunk
    page_content: str
    # The metadata of the chunk
    metadata: ChunkMetadata
