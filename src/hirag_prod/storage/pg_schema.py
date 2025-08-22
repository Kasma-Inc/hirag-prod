from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _vector_type(use_halfvec: bool, dim: int):
    return HALFVEC(dim) if use_halfvec else Vector(dim)


def create_chunks_model(use_halfvec: bool, dim: int):
    vec_col_type = _vector_type(use_halfvec, dim)

    class Chunks(Base):
        __tablename__ = "Chunks"

        documentKey: Mapped[str] = mapped_column(
            String, primary_key=True, nullable=False
        )
        text: Mapped[str] = mapped_column(Text, nullable=False)
        workspaceId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        knowledgeBaseId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        fileName: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        pageNumber: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        uri: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
        chunkIdx: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        documentId: Mapped[Optional[str]] = mapped_column(String, nullable=False)
        chunkType: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageImageUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        pageWidth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        pageHeight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        x0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        y0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        vector: Mapped[List[float]] = mapped_column(vec_col_type, nullable=False)
        updatedAt: Mapped[datetime] = mapped_column(
            DateTime, default=datetime.now, nullable=False
        )

    return Chunks


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
