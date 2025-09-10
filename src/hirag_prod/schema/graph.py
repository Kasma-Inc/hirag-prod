from datetime import datetime

from sqlalchemy import Column, DateTime, String

from hirag_prod.schema.base import Base


class Graph(Base):
    __tablename__ = "Graph"

    source: str = Column(String, primary_key=True, nullable=False)
    target: str = Column(String, primary_key=True, nullable=False)
    workspaceId: str = Column(String, primary_key=True, nullable=False)
    knowledgeBaseId: str = Column(String, primary_key=True, nullable=False)
    updatedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)
    documentId: str = Column(
        String, nullable=False
    )  # For tracing back to the source document

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)
