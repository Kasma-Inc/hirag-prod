import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import text
from sqlmodel import JSON, Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_envs, initialize_config_manager
from hirag_prod.resources.functions import (
    get_resource_manager,
    initialize_resource_manager,
)

load_dotenv("/chatbot/.env")


async def get_table_schema(
    session: AsyncSession,
    *,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return column-level schema for a table."""

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

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
        WHERE c.table_schema = '{table_name}' AND c.table_name = '{schema}'
        ORDER BY c.ordinal_position
    """
    )
    result = await session.exec(query)
    rows = result.all()
    return [dict(row._mapping) for row in rows]


async def update_job_status(
    session: AsyncSession,
    job_id: str,
    status: str,
    *,
    updated_at: Optional[datetime] = None,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> int:
    """Update job status and updatedAt by primary key jobId."""
    # Format the datetime parameter if provided

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

    updated_at_value = updated_at if updated_at is not None else datetime.now()

    query = text(
        f"""
        UPDATE "{schema}"."{table_name}"
           SET "status" = '{status}',
               "updatedAt" = '{updated_at_value.isoformat()}'
         WHERE "jobId" = '{job_id}'
    """
    )
    result = await session.exec(query)
    await session.commit()
    return result.rowcount or 0


async def insert_job(
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

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

    # Format the datetime parameter if provided
    updated_at_value = updated_at if updated_at is not None else datetime.now()

    query = text(
        f"""
        INSERT INTO "{schema}"."{table_name}"("jobId", "workspaceId", "status", "updatedAt")
        VALUES ('{job_id}', '{workspace_id}', '{status}', '{updated_at_value.isoformat()}')
    """
    )
    result = await session.exec(query)
    await session.commit()
    return result.rowcount or 0


async def delete_job(
    session: AsyncSession,
    job_id: str,
    *,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> int:
    """Delete a job row by primary key jobId. Returns affected row count."""

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

    query = text(
        f"""
        DELETE FROM "{schema}"."{table_name}"
        WHERE "jobId" = '{job_id}'
    """
    )

    result = await session.exec(query)
    await session.commit()
    return result.rowcount or 0


async def get_all_records(
    session: AsyncSession,
    *,
    limit: Optional[int] = None,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all rows from the target table.

    - Defaults to env `POSTGRES_TABLE_NAME` when `table_name` is None.
    - Defaults to env `POSTGRES_SCHEMA` or `public` when `schema` is None.
    - Optional `limit` to cap returned rows for safety.
    """

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

    query = text(
        f'SELECT * FROM "{schema}"."{table_name}" {f"LIMIT {limit}" if isinstance(limit, int) and (limit > 0) else ""}'
    )
    result = await session.exec(query)
    rows = result.all()

    return [dict(row._mapping) for row in rows]


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

    initialize_config_manager()
    await initialize_resource_manager()

    get_envs().POSTGRES_SCHEMA = args.schema

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "YES" if value else "NO"
        return str(value)

    try:
        async with get_resource_manager().get_session_maker()() as session:
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
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Invalid --updated-at format. Use ISO 8601 (e.g. 2025-01-01T12:34:56+00:00)",
                            e,
                        )
                        sys.exit(2)
                affected = await insert_job(
                    session,
                    args.add_job,
                    args.workspace,
                    status=args.status or "pending",
                    updated_at=parsed_dt,
                )
                print(f"Inserted rows: {affected}")
                sys.exit(0)

            # Action: delete-job
            if args.delete_job:
                affected = await delete_job(session, args.delete_job)
                print(f"Deleted rows: {affected}")
                sys.exit(0)

            # Action: show-rows
            if args.show_rows:
                rows = await get_all_records(session, limit=args.limit)
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
                rows = await get_table_schema(session)
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
    except Exception as e:
        log_error_info(logging.ERROR, "Failed to connect to Postgres", e)

    await get_resource_manager().cleanup()


if __name__ == "__main__":
    asyncio.run(main())
