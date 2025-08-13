"""
PostgreSQL utils test
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod.storage.pg_utils import DatabaseClient


@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection"""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()

    try:
        engine = db.create_db_engine()
        async with AsyncSession(engine) as session:
            # Simple test query to verify connection
            from sqlalchemy import text

            result = await session.exec(text("SELECT 1 as result"))
            row = result.first()
            assert row.result == 1
    except Exception:
        pytest.skip("Database unavailable")


@pytest.mark.asyncio
async def test_update_job_status():
    """Insert a temp record, update it, verify, then delete."""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()
    engine = db.create_db_engine()

    # Check if the table exists first
    try:
        async with AsyncSession(engine) as session:
            from sqlalchemy import text

            check_query = text(
                f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{db.schema_name or 'public'}' 
                    AND table_name = '{db.table_name}'
                );
            """
            )
            result = await session.exec(check_query)
            table_exists = result.first()[0]
            if not table_exists:
                pytest.skip(
                    f"Table {db.schema_name or 'public'}.{db.table_name} does not exist"
                )
    except Exception:
        pytest.skip("Unable to check table existence")

    temp_job_id = f"test-{int(datetime.now().timestamp())}"
    workspace_id = "ws-test"

    try:
        # Insert a test record
        async with AsyncSession(engine) as session:
            affected = await db.insert_job(
                session,
                temp_job_id,
                workspace_id,
                status="pending",
                updated_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
            assert affected == 1

        # Update the job status
        async with AsyncSession(engine) as session:
            affected = await db.update_job_status(session, temp_job_id, "processing")
            assert affected == 1

        # Verify the update
        async with AsyncSession(engine) as session:
            from sqlalchemy import text

            query = text(
                f"""
                SELECT "status", "updatedAt" FROM "{db.schema_name or 'public'}"."{db.table_name}"
                WHERE "jobId" = '{temp_job_id}'
            """
            )
            result = await session.exec(query)
            row = result.first()
            assert row.status == "processing"
            first_updated = row.updatedAt

        # Update with explicit timestamp
        explicit_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
        async with AsyncSession(engine) as session:
            affected = await db.update_job_status(
                session, temp_job_id, "completed", updated_at=explicit_ts
            )
            assert affected == 1

        # Verify the explicit timestamp update
        async with AsyncSession(engine) as session:
            query = text(
                f"""
                SELECT "status", "updatedAt" FROM "{db.schema_name or 'public'}"."{db.table_name}"
                WHERE "jobId" = '{temp_job_id}'
            """
            )
            result = await session.exec(query)
            row = result.first()
            assert row.status == "completed"
            # Note: comparing timestamps might have precision differences, so we check it's close
            assert abs((row.updatedAt - explicit_ts).total_seconds()) < 1
            assert row.updatedAt != first_updated

    finally:
        # Clean up
        try:
            async with AsyncSession(engine) as session:
                await db.delete_job(session, temp_job_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_fetch_records():
    """Test fetching all records."""
    if not os.getenv("POSTGRES_URL_NO_SSL_DEV"):
        pytest.skip("No database connection string")

    db = DatabaseClient()
    engine = db.create_db_engine()

    # Check if the table exists first
    try:
        async with AsyncSession(engine) as session:
            from sqlalchemy import text

            check_query = text(
                f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{db.schema_name or 'public'}' 
                    AND table_name = '{db.table_name}'
                );
            """
            )
            result = await session.exec(check_query)
            table_exists = result.first()[0]
            if not table_exists:
                pytest.skip(
                    f"Table {db.schema_name or 'public'}.{db.table_name} does not exist"
                )
    except Exception:
        pytest.skip("Unable to check table existence")
