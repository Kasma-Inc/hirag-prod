from datetime import datetime
from typing import List, Optional
from pgvector.sqlalchemy import HALFVEC, Vector

from hirag_prod.schema.base import Base
from hirag_prod.schema.file import File
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, Column
from sqlalchemy.types import ARRAY
import dotenv
import os

# read halfvec and dim from env
dotenv.load_dotenv()
dim = int(os.getenv("EMBEDDING_DIMENSION", 1536))
use_halfvec = bool(os.getenv("USE_HALF_VEC", True))

vec_type = HALFVEC(dim) if use_halfvec else Vector(dim)

class Chunk(Base):
    __tablename__ = "Chunks"

    # Chunk Data
    documentKey: str = Column(
        String, primary_key=True, nullable=False
    )
    text: str = Column(Text, nullable=False)
    # From FileMetadata
    fileName: str = Column(String, nullable=False)
    uri: str = Column(String, nullable=False)
    private: bool = Column(Boolean, default=False, nullable=False)
    knowledgeBaseId: str = Column(String, nullable=False)
    workspaceId: str = Column(String, nullable=False)
    type: Optional[str] = Column(String, nullable=True)
    pageNumber: Optional[int] = Column(Integer, nullable=True)
    uploadedAt: Optional[datetime] = Column(DateTime, nullable=True)
    # From ChunkMetadata
    documentId: str = Column(String, nullable=False)
    chunkIdx: Optional[int] = Column(Integer, nullable=True)
    chunkType: Optional[str] = Column(String, nullable=True)
    pageImageUrl: Optional[str] = Column(String, nullable=True)
    pageWidth: Optional[float] = Column(Float, nullable=True)
    pageHeight: Optional[float] = Column(Float, nullable=True)
    headers: Optional[List[str]] = Column(ARRAY(String), nullable=True)
    children: Optional[List[str]] = Column(ARRAY(String), nullable=True)
    caption: Optional[str] = Column(String, nullable=True)
    bbox: Optional[List[float]] = Column(ARRAY(Float), nullable=True)
    # Computed Data
    vector: List[float] = Column(vec_type, nullable=False)
    updatedAt: datetime = Column(
        DateTime, default=datetime.now, nullable=False
    )

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)

def file_to_chunk(file: File, documentKey: str, text: str, documentId: str, chunkIdx) -> Chunk:
    return Chunk(
        # Given
        documentKey=documentKey,
        text=text,
        documentId=documentId,
        chunkIdx=chunkIdx,
        # Copy
        fileName=file.fileName,
        uri=file.uri,
        private=file.private,
        knowledgeBaseId=file.knowledgeBaseId,
        workspaceId=file.workspaceId,
        type=file.type,
        pageNumber=file.pageNumber,
    )
