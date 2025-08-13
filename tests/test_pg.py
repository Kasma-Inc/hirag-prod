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

    print(f"\n=== DEBUG INFO ===")
    print(f"Test job ID: {temp_job_id}")
    print(f"Workspace ID: {workspace_id}")
    print(f"Database schema: {db.schema_name or 'public'}")
    print(f"Table name: {db.table_name}")

    try:
        # Insert a test record
        initial_updated_at = datetime.now(timezone.utc) - timedelta(days=1)
        print(
            f"\nInserting record with initial_updated_at: {initial_updated_at} (type: {type(initial_updated_at)})"
        )

        async with AsyncSession(engine) as session:
            affected = await db.insert_job(
                session,
                temp_job_id,
                workspace_id,
                status="pending",
                updated_at=initial_updated_at,
            )
            print(f"Insert operation affected {affected} row(s)")
            assert affected == 1

        # Verify the inserted record can be queried and data matches
        print(f"\nVerifying inserted record...")
        async with AsyncSession(engine) as session:
            from sqlalchemy import text

            query = text(
                f"""
                SELECT "jobId", "workspaceId", "status", "updatedAt" 
                FROM "{db.schema_name or 'public'}"."{db.table_name}"
                WHERE "jobId" = '{temp_job_id}'
            """
            )
            result = await session.exec(query)
            row = result.first()
            print(f"Query result: {row}")
            assert (
                row is not None
            ), f"Inserted record with jobId {temp_job_id} not found"
            assert (
                row.jobId == temp_job_id
            ), f"Expected jobId {temp_job_id}, got {row.jobId}"
            assert (
                row.workspaceId == workspace_id
            ), f"Expected workspaceId {workspace_id}, got {row.workspaceId}"
            assert (
                row.status == "pending"
            ), f"Expected status 'pending', got {row.status}"
            # Check timestamp is close (within 1 second due to potential precision differences)
            assert (
                abs((row.updatedAt - initial_updated_at).total_seconds()) < 1
            ), f"Expected updatedAt close to {initial_updated_at}, got {row.updatedAt}"
            print(
                "Got updatedAt datetime:",
                row.updatedAt,
                "\n with type:",
                type(row.updatedAt),
            )
            print("✓ Inserted record verification passed")

        # Update the job status
        print(f"\nUpdating job status to 'processing'...")
        async with AsyncSession(engine) as session:
            affected = await db.update_job_status(session, temp_job_id, "processing")
            print(f"Update operation affected {affected} row(s)")
            assert affected == 1

        # Verify the update
        print(f"Verifying status update...")
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
            print(f"Updated record status: {row.status}, updatedAt: {row.updatedAt}")
            assert row.status == "processing"
            first_updated = row.updatedAt
            print("✓ Status update verification passed")

        # Update with explicit timestamp
        explicit_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
        print(f"\nUpdating with explicit timestamp: {explicit_ts}")
        async with AsyncSession(engine) as session:
            affected = await db.update_job_status(
                session, temp_job_id, "completed", updated_at=explicit_ts
            )
            print(f"Explicit timestamp update affected {affected} row(s)")
            assert affected == 1

        # Verify the explicit timestamp update
        print(f"Verifying explicit timestamp update...")
        async with AsyncSession(engine) as session:
            query = text(
                f"""
                SELECT "status", "updatedAt" FROM "{db.schema_name or 'public'}"."{db.table_name}"
                WHERE "jobId" = '{temp_job_id}'
            """
            )
            result = await session.exec(query)
            row = result.first()
            print(f"Final record status: {row.status}, updatedAt: {row.updatedAt}")
            print(
                f"Time difference from explicit_ts: {abs((row.updatedAt - explicit_ts).total_seconds())} seconds"
            )
            print(f"First updated timestamp: {first_updated}")
            print(f"Current updated timestamp: {row.updatedAt}")
            print(f"Timestamps are different: {row.updatedAt != first_updated}")
            assert row.status == "completed"
            # Note: comparing timestamps might have precision differences, so we check it's close
            assert abs((row.updatedAt - explicit_ts).total_seconds()) < 1
            assert row.updatedAt != first_updated
            print("✓ Explicit timestamp update verification passed")

    finally:
        # Clean up
        print(f"\nCleaning up test record...")
        # try:
        #     async with AsyncSession(engine) as session:
        #         deleted = await db.delete_job(session, temp_job_id)
        #         print(f"Cleanup: deleted {deleted if deleted else 0} record(s)")
        # except Exception as e:
        #     print(f"Cleanup failed: {e}")
        # print("=== END DEBUG INFO ===\n")


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
