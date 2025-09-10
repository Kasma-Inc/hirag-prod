from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, String
from sqlalchemy.types import ARRAY

from hirag_prod.schema.base import Base


class Node(Base):
    __tablename__ = "Nodes"

    id: str = Column(String, primary_key=True, nullable=False)
    workspaceId: str = Column(String, primary_key=True, nullable=False)
    knowledgeBaseId: str = Column(String, primary_key=True, nullable=False)

    entityName: Optional[str] = Column(String, nullable=False)
    entityType: Optional[str] = Column(String, nullable=False, default="entity")
    chunkIds: Optional[List[str]] = Column(ARRAY(String), nullable=False)

    updatedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)
    documentId: str = Column(
        String, nullable=False
    )  # For tracing back to the source document

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)
