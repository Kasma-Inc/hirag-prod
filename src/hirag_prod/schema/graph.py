from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from hirag_prod.schema.base import Base


class Graph(Base):
    __tablename__ = "Graph"

    id: Mapped[str] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    target: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    workspaceId: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    knowledgeBaseId: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    documentId: Mapped[str] = mapped_column(
        String, nullable=False
    )  # For tracing back to the source document

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


def create_graph(metadata: dict, **kwargs) -> Graph:
    # Get all column names from the Graph table
    graph_columns = set(Graph.__table__.columns.keys())

    # Combine metadata and kwargs, with kwargs taking precedence
    all_data = {**metadata, **kwargs}

    # Filter data to only include valid Graph attributes
    graph_data = {}
    for key, value in all_data.items():
        if key in graph_columns:
            graph_data[key] = value

    # Check for required fields and set defaults if needed
    required_fields = [
        "source",
        "target",
        "uri",
        "workspaceId",
        "knowledgeBaseId",
        "documentId",
    ]
    missing_required = [field for field in required_fields if field not in graph_data]

    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")

    # Set default timestamps if not provided
    current_time = datetime.now()
    if "createdAt" not in graph_data:
        graph_data["createdAt"] = current_time
    if "updatedAt" not in graph_data:
        graph_data["updatedAt"] = current_time

    # Create and return the Graph instance
    return Graph(**graph_data)
