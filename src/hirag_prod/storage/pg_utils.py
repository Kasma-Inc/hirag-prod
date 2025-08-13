import argparse
import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from asyncpg import DuplicateTableError
from dotenv import load_dotenv
from sqlalchemy import inspect, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import JSON, Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

load_dotenv("/chatbot/.env")


class DatabaseClient:
    def __init__(self):
        postgres_url = os.getenv("POSTGRES_URL_NO_SSL_DEV")

        if postgres_url and postgres_url.startswith("postgres://"):
            postgres_url = postgres_url.replace("postgres://", "postgresql://")

        self.connection_string = postgres_url
        self.table_name = os.getenv("POSTGRES_TABLE_NAME", "KnowledgeBaseJobs")
        self.schema_name = os.getenv("POSTGRES_SCHEMA", "public")

    def create_db_engine(self, connection_string: Optional[str] = None) -> AsyncEngine:
        """Create a new SQLAlchemy engine."""
        if connection_string is None:
            connection_string = self.connection_string

        if connection_string.startswith("postgres://"):
            connection_string = connection_string.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
        elif connection_string.startswith("postgresql://"):
            connection_string = connection_string.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
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

    async def get_table_schema(
        self,
        session: AsyncSession,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return column-level schema for a table.

        - Defaults to env `POSTGRES_TABLE_NAME` when `table_name` is None.
        - Defaults to `public` when `schema` is None.
        """
        target_table = table_name or self.table_name
        target_schema = schema or "public"

        query = text(
            f"""
            SELECT
                c.ordinal_position AS position,
                c.column_name       AS name,
                c.data_type,
                (c.is_nullable = 'YES') AS is_nullable,
                c.column_default,
                EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema   = c.table_schema
                      AND tc.table_name     = c.table_name
                      AND tc.constraint_type = 'PRIMARY KEY'
                      AND kcu.column_name   = c.column_name
                ) AS is_primary_key
            FROM information_schema.columns c
            WHERE c.table_schema = '{target_schema}' AND c.table_name = '{target_table}'
            ORDER BY c.ordinal_position
        """
        )

        result = await session.exec(query)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def update_job_status(
        self,
        session: AsyncSession,
        job_id: str,
        status: str,
        *,
        updated_at: Optional[datetime] = None,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> int:
        """Update job status and updatedAt by primary key jobId."""
        target_table = table_name or self.table_name
        target_schema = schema or self.schema_name or "public"

        # Format the datetime parameter if provided
        updated_at_value = updated_at if updated_at is not None else datetime.now()

        query = text(
            f"""
            UPDATE "{target_schema}"."{target_table}"
               SET "status" = '{status}',
                   "updatedAt" = '{updated_at_value.isoformat()}'
             WHERE "jobId" = '{job_id}'
        """
        )

        result = await session.exec(query)
        await session.commit()
        return result.rowcount or 0

    async def insert_job(
        self,
        session: AsyncSession,
        job_id: str,
        workspace_id: str,
        status: str = "pending",
        *,
        updated_at: Optional[datetime] = None,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> int:
        """Insert a job row with keys (jobId, workspaceId, status, updatedAt).

        Returns number of affected rows (1 on success).
        """
        target_table = table_name or self.table_name
        target_schema = schema or self.schema_name or "public"

        # Format the datetime parameter if provided
        updated_at_value = updated_at if updated_at is not None else datetime.now()

        query = text(
            f"""
            INSERT INTO "{target_schema}"."{target_table}"("jobId", "workspaceId", "status", "updatedAt")
            VALUES ('{job_id}', '{workspace_id}', '{status}', '{updated_at_value.isoformat()}')
        """
        )

        result = await session.exec(query)
        await session.commit()
        return result.rowcount or 0

    async def delete_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> int:
        """Delete a job row by primary key jobId. Returns affected row count."""
        target_table = table_name or self.table_name
        target_schema = schema or self.schema_name or "public"

        query = text(
            f"""
            DELETE FROM "{target_schema}"."{target_table}"
            WHERE "jobId" = '{job_id}'
        """
        )

        result = await session.exec(query)
        await session.commit()
        return result.rowcount or 0

    async def get_all_records(
        self,
        session: AsyncSession,
        *,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return all rows from the target table.

        - Defaults to env `POSTGRES_TABLE_NAME` when `table_name` is None.
        - Defaults to env `POSTGRES_SCHEMA` or `public` when `schema` is None.
        - Optional `limit` to cap returned rows for safety.
        """
        target_table = table_name or self.table_name
        target_schema = schema or self.schema_name or "public"

        if isinstance(limit, int) and limit > 0:
            query = text(
                f'SELECT * FROM "{target_schema}"."{target_table}" LIMIT {limit}'
            )
            result = await session.exec(query)
            rows = result.all()
        else:
            query = text(f'SELECT * FROM "{target_schema}"."{target_table}"')
            result = await session.exec(query)
            rows = result.all()

        return [dict(row._mapping) for row in rows]

    async def _ensure_table(self, session: AsyncSession, table) -> None:
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

    async def saveContextResult(
        self, session: AsyncSession, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a context result to the database."""

        await self._ensure_table(session, self.ContextualResultTable)

        # Extract fields from Contextual AI API response structure
        document_metadata = result.get("document_metadata", {})
        hierarchy = document_metadata.get("hierarchy", {})

        result_record = self.ContextualResultTable(
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

    async def queryContextResult(
        self, session: AsyncSession, job_id: str
    ) -> Optional[Dict[str, Any]]:
        """Query context result from database."""

        await self._ensure_table(session, self.ContextualResultTable)

        statement = select(self.ContextualResultTable).where(
            self.ContextualResultTable.job_id == job_id
        )
        result = await session.exec(statement)
        result = result.first()

        if result:
            return result.to_dict()

        return None


async def main():
    parser = argparse.ArgumentParser(description="PostgreSQL table utilities")
    parser.add_argument(
        "--schema",
        "-s",
        help="Table schema (defaults to env POSTGRES_SCHEMA or 'public')",
    )
    parser.add_argument(
        "--show-rows",
        action="store_true",
        help="Show all rows from the target table (use --limit to cap)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of rows to display",
    )
    # Insert
    parser.add_argument(
        "--add-job",
        metavar="JOB_ID",
        help="Insert a job row with given JOB_ID (requires --workspace; optional --status, --updated-at)",
    )
    parser.add_argument(
        "--workspace",
        metavar="WORKSPACE_ID",
        help="Workspace id required when using --add-job",
    )
    parser.add_argument(
        "--status",
        default="pending",
        help="Status for --add-job (default: pending)",
    )
    parser.add_argument(
        "--updated-at",
        metavar="ISO_DATETIME",
        help="Optional ISO datetime for --add-job (e.g. 2025-01-01T12:34:56+00:00)",
    )
    # Delete
    parser.add_argument(
        "--delete-job",
        metavar="JOB_ID",
        help="Delete a job row by JOB_ID",
    )
    args = parser.parse_args()

    client = DatabaseClient()
    engine = client.create_db_engine()

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "YES" if value else "NO"
        return str(value)

    try:
        async with AsyncSession(engine) as session:
            resolved_schema = args.schema or os.getenv("POSTGRES_SCHEMA") or "public"

            # Action: add-job
            if args.add_job:
                if not args.workspace:
                    print(
                        "--workspace is required when using --add-job", file=sys.stderr
                    )
                    sys.exit(2)
                parsed_dt: Optional[datetime] = None
                if args.updated_at:
                    try:
                        # Accept ISO 8601 format
                        parsed_dt = datetime.fromisoformat(args.updated_at)
                    except Exception:
                        print(
                            "Invalid --updated-at format. Use ISO 8601 (e.g. 2025-01-01T12:34:56+00:00)",
                            file=sys.stderr,
                        )
                        sys.exit(2)
                affected = await client.insert_job(
                    session,
                    args.add_job,
                    args.workspace,
                    status=args.status or "pending",
                    updated_at=parsed_dt,
                    schema=resolved_schema,
                )
                print(f"Inserted rows: {affected}")
                sys.exit(0)

            # Action: delete-job
            if args.delete_job:
                affected = await client.delete_job(
                    session, args.delete_job, schema=resolved_schema
                )
                print(f"Deleted rows: {affected}")
                sys.exit(0)

            # Action: show-rows
            if args.show_rows:
                rows = await client.get_all_records(
                    session, schema=resolved_schema, limit=args.limit
                )
                if not rows:
                    print("No rows found.")
                    sys.exit(0)

                headers = list(rows[0].keys())
                col_widths = []
                for h in headers:
                    max_content = max((len(_fmt(r.get(h))) for r in rows), default=0)
                    col_widths.append(max(len(h), max_content))

                header_line = " | ".join(
                    h.ljust(w) for h, w in zip(headers, col_widths)
                )
                sep_line = "-+-".join("-" * w for w in col_widths)
                print(header_line)
                print(sep_line)
                for r in rows:
                    line = " | ".join(
                        _fmt(r.get(h)).ljust(w) for h, w in zip(headers, col_widths)
                    )
                    print(line)

            # Show table schema
            else:
                rows = await client.get_table_schema(session, schema=resolved_schema)
                if not rows:
                    print(
                        "No columns found. Check table name and schema.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                headers = [
                    ("#", "position"),
                    ("name", "name"),
                    ("type", "data_type"),
                    ("null?", "is_nullable"),
                    ("default", "column_default"),
                    ("pk?", "is_primary_key"),
                ]

                col_widths = []
                for header, key in headers:
                    max_content = max(
                        (len(_fmt(row.get(key))) for row in rows), default=0
                    )
                    col_widths.append(max(len(header), max_content))

                header_line = " | ".join(
                    header.ljust(width)
                    for (header, _), width in zip(headers, col_widths)
                )
                sep_line = "-+-".join("-" * width for width in col_widths)
                print(header_line)
                print(sep_line)

                for row in rows:
                    line = " | ".join(
                        _fmt(row.get(key)).ljust(width)
                        for (_, key), width in zip(headers, col_widths)
                    )
                    print(line)
    except Exception as exc:
        print(f"Failed to connect to PostgreSQL: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
