from typing import List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel

from hirag_prod.schema.file import FileMetadata


# If you change this file, you may need to change src/hirag_prod/storage/pg_schema.py
# Also the hirag.py _chunk_props_for_vdb function
class ChunkMetadata(FileMetadata):
    document_id: str  # The id of the document that the chunk is from
    chunk_idx: Optional[int]  # the index of the chunk in the document
    chunk_type: Optional[str]  # the type of the chunk
    page_image_url: Optional[str]  # the image url of the page
    page_width: Optional[float]  # the width of the page
    page_height: Optional[float]  # the height of the page
    headers: Optional[List[str]]  # The header's chunk's id of the chunk
    children: Optional[List[str]]  # The children's chunk's id of the chunk
    caption: Optional[str]  # The caption of the chunk
    bbox: Optional[
        List[List[float]]
    ]  # The bounding box of the chunk, may contain multiple boxes


class Chunk(Document, BaseModel):
    id: str  # unique identifier for the chunk
    page_content: str  # the content of the chunk
    metadata: ChunkMetadata
