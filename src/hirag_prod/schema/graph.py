from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from hirag_prod.schema.base import Base


class Graph(Base):
    __tablename__ = "Graph"

    source: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    target: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    documentId: Mapped[str] = mapped_column(
        String, nullable=False
    )  # For tracing back to the source document

    def __iter__(self):
        for column_name in self.__table__.columns.keys():
            yield column_name, getattr(self, column_name)
