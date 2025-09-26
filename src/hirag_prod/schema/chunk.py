from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, cast
from sqlalchemy.orm import Mapped, column_property, mapped_column
from sqlalchemy.types import ARRAY

from hirag_prod.schema.base import Base
from hirag_prod.schema.vector_config import PGVECTOR


class Chunk(Base):
    __tablename__ = "Chunks"

    # Chunk Data
    documentKey: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # From FileMetadata
    fileName: Mapped[str] = mapped_column(String, nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pageNumber: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    # From ChunkMetadata
    documentId: Mapped[str] = mapped_column(String, nullable=False)
    chunkIdx: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunkType: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pageImageUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pageWidth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pageHeight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    headers: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    children: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    caption: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    bbox: Mapped[Optional[List[float]]] = mapped_column(
        ARRAY(Float, dimensions=2), nullable=True
    )
    # Computed Data
    vector: Mapped[List[float]] = mapped_column(PGVECTOR, nullable=False)
    vector_float_array: Mapped[List[float]] = column_property(
        cast(vector, ARRAY(Float(4)))
    )
    # Timestamps and Users
    createdAt: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    createdBy: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updatedAt: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updatedBy: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    def __iter__(self):
        for column_name in self.__table__.columns.keys():
            yield column_name, getattr(self, column_name)
