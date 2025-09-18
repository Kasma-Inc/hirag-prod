from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.types import ARRAY

from hirag_prod.schema.base import Base
from hirag_prod.schema.vector_config import vec_type


class Chunk(Base):
    __tablename__ = "Chunks"

    # Chunk Data
    documentKey: str = Column(String, primary_key=True, nullable=False)
    knowledgeBaseId: str = Column(String, primary_key=True, nullable=False)
    workspaceId: str = Column(String, primary_key=True, nullable=False)
    text: str = Column(Text, nullable=False)
    # From FileMetadata
    fileName: str = Column(String, nullable=False)
    uri: str = Column(String, nullable=False)
    private: bool = Column(Boolean, default=False, nullable=False)
    type: Optional[str] = Column(String, nullable=True)
    pageNumber: Optional[List[int]] = Column(ARRAY(Integer), nullable=True)
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
    bbox = Column(ARRAY(Float, dimensions=2), nullable=True)
    # Computed Data
    vector: List[float] = Column(vec_type, nullable=False)
    updatedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)
