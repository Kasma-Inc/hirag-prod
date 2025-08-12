"""
PostgreSQL utils test
"""

import os
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest
from psycopg2 import sql

from hirag_prod.storage.pg_utils import DatabaseClient


def test_database_connection():
    """Test database connection"""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 as result")
                result = cursor.fetchone()
                assert result["result"] == 1
    except psycopg2.OperationalError:
        pytest.skip("Database unavailable")


def test_update_job_status():
    """Insert a temp record, update it, verify, then delete."""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()

    temp_job_id = f"test-{int(datetime.now().timestamp())}"
    workspace_id = "ws-test"
    schema_ident = sql.Identifier(db.schema_name or "public")
    table_ident = sql.Identifier(db.table_name)

    insert_query = sql.SQL(
        """
        INSERT INTO {schema}.{table}("jobId", "workspaceId", "status", "updatedAt")
        VALUES (%s, %s, %s, NOW() - interval '1 day')
        """
    ).format(schema=schema_ident, table=table_ident)

    select_query = sql.SQL(
        """
        SELECT "status", "updatedAt" FROM {schema}.{table}
        WHERE "jobId" = %s
        """
    ).format(schema=schema_ident, table=table_ident)

    delete_query = sql.SQL(
        """
        DELETE FROM {schema}.{table} WHERE "jobId" = %s
        """
    ).format(schema=schema_ident, table=table_ident)

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_query, (temp_job_id, workspace_id, "pending"))

        affected = db.update_job_status(temp_job_id, "processing")
        assert affected == 1

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(select_query, (temp_job_id,))
                row = cur.fetchone()
                assert row["status"] == "processing"
                first_updated = row["updatedAt"]

        explicit_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
        affected = db.update_job_status(
            temp_job_id, "completed", updated_at=explicit_ts
        )
        assert affected == 1

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(select_query, (temp_job_id,))
                row = cur.fetchone()
                assert row["status"] == "completed"
                assert row["updatedAt"] == explicit_ts
                assert row["updatedAt"] != first_updated

    finally:
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query, (temp_job_id,))
        except Exception:
            pass


def test_fetch_records():
    """Insert a temp record, fetch all rows, ensure it is present, then delete."""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()

    temp_job_id = f"test-{int(datetime.now().timestamp())}"
    workspace_id = "ws-test"

    try:
        affected = db.insert_job(temp_job_id, workspace_id, status="pending")
        assert affected == 1

        rows = db.get_all_records()
        assert any(row.get("jobId") == temp_job_id for row in rows)

    finally:
        try:
            db.delete_job(temp_job_id)
        except Exception:
            pass
