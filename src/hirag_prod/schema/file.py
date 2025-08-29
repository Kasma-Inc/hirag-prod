from datetime import datetime
from typing import Optional

from hirag_prod.schema.base import Base
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Column

class File(Base):
    __tablename__ = "Files"

    # File Data
    documentKey: str = Column(String, primary_key=True, nullable=False)
    text: str = Column(Text, nullable=False)
    # FileMetadata
    fileName: str = Column(String, nullable=False)
    uri: Optional[str] = Column(String, nullable=True)
    private: Optional[bool] = Column(
        Boolean, default=False, nullable=True
    )
    knowledgeBaseId: str = Column(String, nullable=False)
    workspaceId: str = Column(String, nullable=False)
    type: Optional[str] = Column(String, nullable=True)
    pageNumber: Optional[int] = Column(Integer, nullable=True)
    uploadedAt: datetime = Column(
        DateTime, default=datetime.now, nullable=False
    )
    markdownContent: Optional[str] = Column(Text, nullable=True)
    tableOfContents: Optional[str] = Column(Text, nullable=True)
    # Computed Data
    updatedAt: datetime = Column(
        DateTime, default=datetime.now, nullable=False
    )
