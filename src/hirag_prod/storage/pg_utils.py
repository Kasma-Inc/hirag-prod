from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod.configs.functions import get_envs


async def update_job_status(
    session: AsyncSession,
    file_id: str,
    status: str,
    *,
    updated_at: Optional[datetime] = None,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> int:
    """Update job status and updatedAt by primary key id (jobId)."""
    # Format the datetime parameter if provided

    table_name = table_name or get_envs().POSTGRES_TABLE_NAME
    schema = schema or get_envs().POSTGRES_SCHEMA

    updated_at_value = updated_at if updated_at is not None else datetime.now()

    query = text(
        f"""
        UPDATE "{schema}"."{table_name}"
           SET "status" = '{status}',
               "updatedAt" = '{updated_at_value.isoformat()}'
         WHERE id = '{file_id}'
    """
    )
    result = await session.exec(query)
    await session.commit()
    return result.rowcount or 0
