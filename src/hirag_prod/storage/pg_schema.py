from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _vector_type(use_halfvec: bool, dim: int):
    return HALFVEC(dim) if use_halfvec else Vector(dim)


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
