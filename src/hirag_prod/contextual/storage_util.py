# This is the basic utils for saving results in neondb
from typing import Any, Dict, List, Optional, Union

from sqlmodel import Field, SQLModel
from sqlalchemy import inspect, JSON
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import JSON, Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from asyncpg import DuplicateTableError


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


def create_db_engine(connection_string: str) -> AsyncEngine:
    """Create a new SQLAlchemy engine."""
    if connection_string.startswith("postgres://"):
        connection_string = connection_string.replace("postgres://", "postgresql+asyncpg://", 1)
    elif connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif connection_string.startswith("postgresql+asyncpg://"):
        pass
    else:
        raise ValueError(
            "Invalid PostgreSQL URL format. Must start with 'postgresql://' or 'postgresql+asyncpg://'."
        )

    db = create_async_engine(
        connection_string,
        pool_pre_ping=True,  # tests connections before use
    )
    
    return db


async def _ensure_table(session: AsyncSession, table) -> None:
    """Ensure the ContextualResultTable exists in the database."""
    def _sync_create(sync_session: AsyncSession):
        # Use the inspector from sqlalchemy to check if table exists
        engine = sync_session.get_bind()
        if not inspect(engine).has_table(table.__tablename__):
            try:
                SQLModel.metadata.create_all(engine, tables=[table.__table__])
            except ProgrammingError as e:
                if isinstance(e.__cause__.__cause__, DuplicateTableError):
                    pass
                else:
                    raise

    await session.run_sync(_sync_create)


async def saveContextResult(session: AsyncSession, result: Dict[str, Any]) -> Dict[str, Any]:
    """Save a context result to the database."""

    _ensure_table(session, ContextualResultTable)

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

    session.add(result_record)
    await session.commit()

    return result_record.to_dict()


async def queryContextResult(session: AsyncSession, job_id: str) -> Optional[Dict[str, Any]]:
    """Query context result from database."""

    _ensure_table(session, ContextualResultTable)

    statement = select(ContextualResultTable).where(
        ContextualResultTable.job_id == job_id
    )
    result = await session.exec(statement).first()

    if result:
        return result.to_dict()

    return None
