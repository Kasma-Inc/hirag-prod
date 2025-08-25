from typing import Optional

from langchain_core.documents import Document
from pydantic import BaseModel

from hirag_prod.schema.file import FileMetadata


class ChunkMetadata(FileMetadata):
    knowledge_base_id: Optional[
        str
    ]  # The id of the knowledge base that the chunk is from
    workspace_id: Optional[str]  # The id of the workspace that the chunk is from
    chunk_idx: Optional[int]  # the index of the chunk in the document
    chunk_type: Optional[str]  # the type of the chunk
    page_number: Optional[int]  # the page number of the chunk
    page_image_url: Optional[str]  # the image url of the page
    page_width: Optional[float]  # the width of the page
    page_height: Optional[float]  # the height of the page
    document_id: Optional[str]  # The id of the document that the chunk is from
    x_0: Optional[float]  # x coordinate of the top left corner of the bounding box
    y_0: Optional[float]  # y coordinate of the top left corner of the bounding box
    x_1: Optional[float]  # x coordinate of the bottom right corner of the bounding box
    y_1: Optional[float]  # y coordinate of the bottom right corner of the bounding box


class Chunk(Document, BaseModel):
    id: str  # unique identifier for the chunk
    page_content: str  # the content of the chunk
    metadata: ChunkMetadata
