from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import ARRAY

from hirag_prod.schema.base import Base


class Node(Base):
    __tablename__ = "Nodes"

    id: Mapped[str] = mapped_column(String, nullable=True)
    node_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )

    entityName: Mapped[Optional[str]] = mapped_column(String, nullable=False)
    entityType: Mapped[Optional[str]] = mapped_column(
        String, nullable=False, default="entity"
    )
    chunkIds: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    documentId: Mapped[str] = mapped_column(
        String, nullable=False
    )  # For tracing back to the source document

    # Timestamps and Users
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


def create_node(metadata: dict, **kwargs) -> Node:
    # Get all column names from the Node table
    node_columns = set(Node.__table__.columns.keys())

    # Combine metadata and kwargs, with kwargs taking precedence
    all_data = {**metadata, **kwargs}

    # Filter data to only include valid Node attributes
    node_data = {}
    for key, value in all_data.items():
        if key in node_columns:
            node_data[key] = value

    # Check for required fields and set defaults if needed
    required_fields = [
        "node_id",
        "workspaceId",
        "knowledgeBaseId",
        "entityName",
        "chunkIds",
        "uri",
        "documentId",
    ]
    missing_required = [field for field in required_fields if field not in node_data]

    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")

    # Set default timestamps if not provided
    current_time = datetime.now()
    if "createdAt" not in node_data:
        node_data["createdAt"] = current_time
    if "updatedAt" not in node_data:
        node_data["updatedAt"] = current_time

    # Set default values for optional fields if not provided
    if "entityType" not in node_data:
        node_data["entityType"] = "entity"

    # Create and return the Node instance
    return Node(**node_data)
