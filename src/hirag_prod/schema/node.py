from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import ARRAY

from hirag_prod.schema.base import Base


class Node(Base):
    __tablename__ = "Nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )

    entityName: Mapped[Optional[str]] = mapped_column(String, nullable=False)
    entityType: Mapped[Optional[str]] = mapped_column(
        String, nullable=False, default="entity"
    )
    chunkIds: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=False)

    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    uri: Mapped[str] = mapped_column(String, nullable=False)
    documentId: Mapped[str] = mapped_column(
        String, nullable=False
    )  # For tracing back to the source document

    def __iter__(self):
        for column_name in self.__table__.columns.keys():
            yield column_name, getattr(self, column_name)
