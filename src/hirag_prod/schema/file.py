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

    id: Mapped[str] = mapped_column(String, nullable=True)
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
    tableOfContents: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Timestamps and Users
    extractedTimestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    createdBy: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updatedBy: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    def __iter__(self):
        for column_name in self.__table__.columns.keys():
            yield column_name, getattr(self, column_name)


def create_file(metadata: dict, **kwargs) -> File:
    # Get all column names from the File table
    file_columns = set(File.__table__.columns.keys())

    # Combine metadata and kwargs, with kwargs taking precedence
    all_data = {**metadata, **kwargs}

    # Filter data to only include valid File attributes
    file_data = {}
    for key, value in all_data.items():
        if key in file_columns:
            file_data[key] = value

    # Check for required fields and set defaults if needed
    required_fields = [
        "documentKey",
        "knowledgeBaseId",
        "workspaceId",
        "text",
        "fileName",
    ]
    missing_required = [field for field in required_fields if field not in file_data]

    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")

    # Set default timestamps if not provided
    current_time = datetime.now()
    if "createdAt" not in file_data:
        file_data["createdAt"] = current_time
    if "updatedAt" not in file_data:
        file_data["updatedAt"] = current_time

    # Set default values for optional fields if not provided
    if "private" not in file_data:
        file_data["private"] = False

    # Validate file type if provided
    if "type" in file_data and file_data["type"] is not None:
        valid_types = [
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
        if file_data["type"] not in valid_types:
            raise ValueError(
                f"Invalid file type '{file_data['type']}'. Must be one of: {valid_types}"
            )

    # Create and return the File instance
    return File(**file_data)
