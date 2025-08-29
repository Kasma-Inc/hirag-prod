from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import ARRAY


class Base(DeclarativeBase):
    pass


def _vector_type(use_halfvec: bool, dim: int):
    return HALFVEC(dim) if use_halfvec else Vector(dim)


def create_chunks_model(use_halfvec: bool, dim: int):
    vec_col_type = _vector_type(use_halfvec, dim)

    class Chunks(Base):
        __tablename__ = "Chunks"

        # Chunk Data
        documentKey: Mapped[str] = mapped_column(
            String, primary_key=True, nullable=False
        )
        text: Mapped[str] = mapped_column(Text, nullable=False)
        # From FileMetadata
        fileName: Mapped[str] = mapped_column(String, nullable=False)
        uri: Mapped[str] = mapped_column(String, nullable=False)
        private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
        knowledgeBaseId: Mapped[str] = mapped_column(String, nullable=False)
        workspaceId: Mapped[str] = mapped_column(String, nullable=False)
        type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageNumber: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        # From ChunkMetadata
        documentId: Mapped[str] = mapped_column(String, nullable=False)
        chunkIdx: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        chunkType: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageImageUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageWidth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        pageHeight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        headers: Mapped[Optional[List[str]]] = mapped_column(
            ARRAY(String), nullable=True
        )
        children: Mapped[Optional[List[str]]] = mapped_column(
            ARRAY(String), nullable=True
        )
        caption: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        bbox: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)
        # Computed Data
        vector: Mapped[List[float]] = mapped_column(vec_col_type, nullable=False)
        updatedAt: Mapped[datetime] = mapped_column(
            DateTime, default=datetime.now, nullable=False
        )

    return Chunks


def create_file_model(use_halfvec: bool, dim: int):
    # keeping args to fit into factories
    class Files(Base):
        __tablename__ = "Files"

        # File Data
        id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
        pageContent: Mapped[str] = mapped_column(Text, nullable=False)
        # FileMetadata
        fileName: Mapped[str] = mapped_column(String, nullable=False)
        uri: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        private: Mapped[Optional[bool]] = mapped_column(
            Boolean, default=False, nullable=True
        )
        knowledgeBaseId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        workspaceId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageNumber: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        uploadedAt: Mapped[datetime] = mapped_column(
            DateTime, default=datetime.now, nullable=False
        )
        markdownContent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        tableOfContents: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        # Computed Data
        updatedAt: Mapped[datetime] = mapped_column(
            DateTime, default=datetime.now, nullable=False
        )

    return Files


def create_triplets_model(use_halfvec: bool, dim: int):
    vec_col_type = _vector_type(use_halfvec, dim)

    class Triplets(Base):
        __tablename__ = "Triplets"

        source: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        target: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        description: Mapped[Optional[str]] = mapped_column(
            Text, primary_key=True, nullable=False
        )
        fileName: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        knowledgeBaseId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        workspaceId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        vector: Mapped[List[float]] = mapped_column(vec_col_type, nullable=False)
        updatedAt: Mapped[datetime] = mapped_column(
            DateTime, default=datetime.now, nullable=False
        )

    return Triplets
