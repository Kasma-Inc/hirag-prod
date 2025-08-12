# This is the basic utils for saving results in neondb
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import JSON, Engine, create_engine, text
from sqlmodel import Field, SQLModel


class ContextualResultTable(SQLModel, table=True):
    __tablename__ = "ContextualResultTable"
    job_id: str = Field(primary_key=True)
    file_name: str
    markdown_document: str
    pages: List[Dict[str, Any]] = Field(sa_type=JSON, nullable=True)
    hierarchy_blocks: List[Dict[str, Any]] = Field(sa_type=JSON, nullable=True)
    table_of_content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "markdown_document": self.markdown_document,
            "pages": self.pages,
            "hierarchy_blocks": self.hierarchy_blocks,
            "table_of_content": self.table_of_content,
        }


def create_db_engine(connection_string: str) -> Engine:
    """Create a new SQLAlchemy engine."""
    return create_engine(connection_string)


def _ensure_table(engine: Engine, table) -> None:
    """Ensure the ContextualResultTable exists in the database."""
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, table.__tablename__):
            SQLModel.metadata.create_all(engine, tables=[table.__table__])


def saveContextResult(engine: Engine, result: Dict[str, Any]) -> Dict[str, Any]:
    """Save a context result to the database."""
    from sqlmodel import Session

    _ensure_table(engine, ContextualResultTable)

    # Extract fields from Contextual AI API response structure
    document_metadata = result.get("document_metadata", {})
    hierarchy = document_metadata.get("hierarchy", {})

    result_record = ContextualResultTable(
        job_id=result.get("job_id"),
        file_name=result.get("file_name", ""),
        markdown_document=result.get("markdown_document", ""),
        pages=result.get("pages", []),
        hierarchy_blocks=hierarchy.get("blocks", []),
        table_of_content=hierarchy.get("table_of_contents", ""),
    )

    with Session(engine) as session:
        # Use merge to handle upsert (insert or update)
        session.merge(result_record)
        session.commit()

    return result_record.to_dict()


def queryContextResult(engine: Engine, job_id: str) -> Optional[Dict[str, Any]]:
    """Query context result from database."""
    from sqlmodel import Session, select

    _ensure_table(engine, ContextualResultTable)

    with Session(engine) as session:
        statement = select(ContextualResultTable).where(
            ContextualResultTable.job_id == job_id
        )
        result = session.exec(statement).first()

        if result:
            return result.to_dict()

        return None
