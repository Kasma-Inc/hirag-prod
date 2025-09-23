from datetime import datetime
from typing import Literal, Optional

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hirag_prod.schema.base import Base

file_types = Literal[
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "jpg",
    "png",
    "zip",
    "txt",
    "csv",
    "text",
    "tsv",
    "html",
    "md",
]


class File(Base):
    __tablename__ = "Files"

    # File Data
    documentKey: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # FileMetadata
    fileName: Mapped[str] = mapped_column(String, nullable=False)
    uri: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    private: Mapped[Optional[bool]] = mapped_column(
        Boolean, default=False, nullable=True
    )
    type: Mapped[Optional[file_types]] = mapped_column(String, nullable=True)
    pageNumber: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    uploadedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    tableOfContents: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Computed Data
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    def __iter__(self):
        for column_name in self.__table__.columns.keys():
            yield column_name, getattr(self, column_name)
