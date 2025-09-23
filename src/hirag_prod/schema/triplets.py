from datetime import datetime
from typing import List

from sqlalchemy import ARRAY, DateTime, Float, String, Text, cast
from sqlalchemy.orm import Mapped, column_property, mapped_column

from hirag_prod.schema.base import Base
from hirag_prod.schema.vector_config import PGVECTOR


class Triplets(Base):
    __tablename__ = "Triplets"

    source: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    fileName: Mapped[str] = mapped_column(String, nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    documentId: Mapped[str] = mapped_column(String, nullable=False)
    vector: Mapped[List[float]] = mapped_column(PGVECTOR, nullable=False)
    vector_float_array: Mapped[List[float]] = column_property(
        cast(vector, ARRAY(Float(4)))
    )
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
