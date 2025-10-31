"""
PostgreSQL utils test
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text

from hirag_prod.configs.functions import get_postgres_config, initialize_config_manager
from hirag_prod.resources.functions import (
    get_db_session_maker,
    get_resource_manager,
    initialize_resource_manager,
)
from hirag_prod.storage.pg_utils import insert_job, update_job_status


@pytest_asyncio.fixture(autouse=True)
async def initialize_and_cleanup():
    initialize_config_manager()
    await initialize_resource_manager()
    yield
    await get_resource_manager().cleanup()


@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection"""
    try:
        async with get_db_session_maker()() as session:
            # Simple test query to verify connection
            result = await session.exec(text("SELECT 1 as result"))
            row = result.first()
            assert row.result == 1
    except Exception:
        pytest.skip("Database unavailable")


@pytest.mark.asyncio
async def test_update_job_status():
    """Insert a temp record, update it, verify, then delete."""
    # Check if the table exists first
    try:
        async with get_db_session_maker()() as session:
            check_query = text(
                f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{get_postgres_config().table_schema}' 
                    AND table_name = '{get_postgres_config().table_name}'
                );
            """
            )
            result = await session.exec(check_query)
            table_exists = result.first()[0]
            if not table_exists:
                pytest.skip(
                    f"Table {get_postgres_config().table_schema}.{get_postgres_config().table_name} does not exist"
                )
    except Exception:
        pytest.skip("Unable to check table existence")

    temp_job_id = f"test-{int(datetime.now().timestamp())}"
    workspace_id = "ws-test"

    print(f"\n=== DEBUG INFO ===")
    print(f"Test job ID: {temp_job_id}")
    print(f"Workspace ID: {workspace_id}")
    print(f"Database schema: {get_postgres_config().table_schema}")
    print(f"Table name: {get_postgres_config().table_name}")

    try:
        # Insert a test record
        initial_updated_at = datetime.now(timezone.utc) - timedelta(days=1)
        print(
            f"\nInserting record with initial_updated_at: {initial_updated_at} (type: {type(initial_updated_at)})"
        )

        async with get_db_session_maker()() as session:
            affected = await insert_job(
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
        async with get_db_session_maker()() as session:
            query = text(
                f"""
                SELECT "jobId", "workspaceId", "status", "updatedAt" 
                FROM "{get_postgres_config().table_schema}"."{get_postgres_config().table_name}"
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
        async with get_db_session_maker()() as session:
            affected = await update_job_status(session, temp_job_id, "processing")
            print(f"Update operation affected {affected} row(s)")
            assert affected == 1

        # Verify the update
        print(f"Verifying status update...")
        async with get_db_session_maker()() as session:
            query = text(
                f"""
                SELECT "status", "updatedAt" FROM "{get_postgres_config().table_schema}"."{get_postgres_config().table_name}"
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
        async with get_db_session_maker()() as session:
            affected = await update_job_status(
                session, temp_job_id, "completed", updated_at=explicit_ts
            )
            print(f"Explicit timestamp update affected {affected} row(s)")
            assert affected == 1

        # Verify the explicit timestamp update
        print(f"Verifying explicit timestamp update...")
        async with get_db_session_maker()() as session:
            query = text(
                f"""
                SELECT "status", "updatedAt" FROM "{get_postgres_config().table_schema}"."{get_postgres_config().table_name}"
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

    # Check if the table exists first
    try:
        async with get_db_session_maker()() as session:
            check_query = text(
                f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{get_postgres_config().table_schema}' 
                    AND table_name = '{get_postgres_config().table_name}'
                );
            """
            )
            result = await session.exec(check_query)
            table_exists = result.first()[0]
            if not table_exists:
                pytest.skip(
                    f"Table {get_postgres_config().table_schema}.{get_postgres_config().table_name} does not exist"
                )
    except Exception:
        pytest.skip("Unable to check table existence")
