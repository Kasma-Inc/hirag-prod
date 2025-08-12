import argparse
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

load_dotenv("/chatbot/.env")


class DatabaseClient:
    def __init__(self):
        postgres_url = os.getenv("POSTGRES_URL_NO_SSL_DEV")

        if postgres_url and postgres_url.startswith("postgres://"):
            postgres_url = postgres_url.replace("postgres://", "postgresql://")

        self.connection_string = postgres_url
        self.table_name = os.getenv("POSTGRES_TABLE_NAME", "KnowledgeBaseJobs")
        self.schema_name = os.getenv("POSTGRES_SCHEMA", "public")

    @contextmanager
    def get_connection(self):
        """Context manager for psycopg2 connection"""
        conn = psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def get_table_schema(
        self, table_name: Optional[str] = None, schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return column-level schema for a table.

        - Defaults to env `POSTGRES_TABLE_NAME` when `table_name` is None.
        - Defaults to `public` when `schema` is None.
        """
        target_table = table_name or self.table_name
        target_schema = schema or "public"

        sql = """
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
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (target_schema, target_table))
                return [dict(row) for row in cursor.fetchall()]

    def update_job_status(
        self,
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

        query = sql.SQL(
            """
            UPDATE {schema}.{table}
               SET {status_col} = %s,
                   {updated_col} = COALESCE(%s, NOW())
             WHERE {pk_col} = %s
            """
        ).format(
            schema=sql.Identifier(target_schema),
            table=sql.Identifier(target_table),
            status_col=sql.Identifier("status"),
            updated_col=sql.Identifier("updatedAt"),
            pk_col=sql.Identifier("jobId"),
        )

        params = (status, updated_at, job_id)

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount or 0

    def insert_job(
        self,
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

        query = sql.SQL(
            """
            INSERT INTO {schema}.{table}("jobId", "workspaceId", "status", "updatedAt")
            VALUES (%s, %s, %s, COALESCE(%s, NOW()))
            """
        ).format(
            schema=sql.Identifier(target_schema), table=sql.Identifier(target_table)
        )

        params = (job_id, workspace_id, status, updated_at)

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount or 0

    def delete_job(
        self,
        job_id: str,
        *,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> int:
        """Delete a job row by primary key jobId. Returns affected row count."""
        target_table = table_name or self.table_name
        target_schema = schema or self.schema_name or "public"

        query = sql.SQL(
            """
            DELETE FROM {schema}.{table}
            WHERE "jobId" = %s
            """
        ).format(
            schema=sql.Identifier(target_schema), table=sql.Identifier(target_table)
        )

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (job_id,))
                return cursor.rowcount or 0

    def get_all_records(
        self,
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

        base = sql.SQL("SELECT * FROM {schema}.{table}").format(
            schema=sql.Identifier(target_schema),
            table=sql.Identifier(target_table),
        )

        if isinstance(limit, int) and limit > 0:
            query = sql.SQL("{base} LIMIT %s").format(base=base)
            params: tuple = (limit,)
        else:
            query = base
            params = ()

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
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

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "YES" if value else "NO"
        return str(value)

    try:
        resolved_schema = args.schema or os.getenv("POSTGRES_SCHEMA") or "public"
        # Action: add-job
        if args.add_job:
            if not args.workspace:
                print("--workspace is required when using --add-job", file=sys.stderr)
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
            affected = client.insert_job(
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
            affected = client.delete_job(args.delete_job, schema=resolved_schema)
            print(f"Deleted rows: {affected}")
            sys.exit(0)

        # Action: show-rows
        if args.show_rows:
            rows = client.get_all_records(schema=resolved_schema, limit=args.limit)
            if not rows:
                print("No rows found.")
                sys.exit(0)

            headers = list(rows[0].keys())
            col_widths = []
            for h in headers:
                max_content = max((len(_fmt(r.get(h))) for r in rows), default=0)
                col_widths.append(max(len(h), max_content))

            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            sep_line = "-+-".join("-" * w for w in col_widths)
            print(header_line)
            print(sep_line)
            for r in rows:
                line = " | ".join(
                    _fmt(r.get(h)).ljust(w) for h, w in zip(headers, col_widths)
                )
                print(line)
        else:
            rows = client.get_table_schema(schema=resolved_schema)
            if not rows:
                print("No columns found. Check table name and schema.", file=sys.stderr)
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
                max_content = max((len(_fmt(row.get(key))) for row in rows), default=0)
                col_widths.append(max(len(header), max_content))

            header_line = " | ".join(
                header.ljust(width) for (header, _), width in zip(headers, col_widths)
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
    except psycopg2.OperationalError as exc:
        print(f"Failed to connect to PostgreSQL: {exc}", file=sys.stderr)
        sys.exit(2)
