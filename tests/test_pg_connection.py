import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text

from configs.functions import initialize_config_manager
from resources.functions import (
    get_db_engine,
    get_resource_manager,
    initialize_resource_manager,
)

load_dotenv()


@pytest_asyncio.fixture(autouse=True)
async def initialize_and_cleanup():
    initialize_config_manager()
    await initialize_resource_manager()
    yield
    await get_resource_manager().cleanup()


@pytest.mark.asyncio
async def test_async_db_connection() -> None:
    async with get_db_engine().connect() as conn:
        result = await conn.execute(text("select 'hello world'"))
        rows = result.fetchall()
        assert rows == [("hello world",)], f"Unexpected query result: {rows}"
