from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text

from hirag_prod.schema.base import Base


class File(Base):
    __tablename__ = "Files"

    # File Data
    documentKey: str = Column(String, primary_key=True, nullable=False)
    text: str = Column(Text, nullable=False)
    # FileMetadata
    fileName: str = Column(String, nullable=False)
    uri: Optional[str] = Column(String, nullable=True)
    private: Optional[bool] = Column(Boolean, default=False, nullable=True)
    knowledgeBaseId: str = Column(String, nullable=False)
    workspaceId: str = Column(String, nullable=False)
    type: Optional[str] = Column(String, nullable=True)
    pageNumber: Optional[int] = Column(Integer, nullable=True)
    uploadedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)
    tableOfContents: Optional[list] = Column(JSON, nullable=True)
    # Computed Data
    updatedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)
