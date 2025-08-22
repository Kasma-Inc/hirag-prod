import os
import re

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()


@pytest.mark.asyncio
async def test_async_db_connection() -> None:
    db_url = os.getenv("POSTGRES_URL_NO_SSL_DEV")
    assert db_url is not None, "Environment variable POSTGRES_URL_NO_SSL_DEV is not set"

    async_url = re.sub(r"^postgresql:", "postgresql+asyncpg:", db_url)
    engine = create_async_engine(async_url, echo=True)

    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("select 'hello world'"))
            rows = result.fetchall()

            assert rows == [("hello world",)], f"Unexpected query result: {rows}"
    finally:
        await engine.dispose()
