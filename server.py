"""HiRAG MCP Server"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from mcp.server.fastmcp import Context, FastMCP

from hirag_prod import HiRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hirag_mcp.server")


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[dict]:
    """manage lifespan of HiRAG instance"""
    # global hirag_instance
    logger.info("Initializing HiRAG instance...")
    hirag_instance = await HiRAG.create()
    document_path = f"tests/test.txt"
    content_type = "csv"
    document_meta = {
        "type": "csv",
        "filename": "test.txt",
        "uri": document_path,
        "private": False,
    }
    logger.info("HiRAG instance initialized successfully")
    try:
        yield {"hirag": hirag_instance}
    finally:
        logger.info("HiRAG MCP server connection closed")


mcp = FastMCP("HiRAG MCP Server", lifespan=lifespan)


@mcp.tool()
async def naive_search(
    query: str, max_tokens: Optional[int] = None, ctx: Context = None
) -> str:
    """
    Perform a naive search over the knowledge base.

    Args:
        query: The search query text
        max_tokens: Optional maximum tokens for the response

    Returns:
        The search results as text
    """
    try:
        hirag_instance = ctx.request_context.lifespan_context["hirag"]
    except Exception as e:
        logger.error(f"Error in HiRAG instance access: {e}")
        return f"{str(e)}"

    result = await hirag_instance.query_chunks(query)

    return result


@mcp.tool()
async def hi_search(
    query: str, max_tokens: Optional[int] = None, ctx: Context = None
) -> str:
    """
    Perform a hybrid search combining both local and global knowledge over the knowledge base. (default to use)

    Args:
        query: The search query text
        max_tokens: Optional maximum tokens for the response

    Returns:
        The search results as text
    """
    try:
        hirag_instance = ctx.request_context.lifespan_context["hirag"]
    except Exception as e:
        logger.error(f"Error in HiRAG instance access: {e}")
        return f"{str(e)}"

    timeout_seconds = 100

    try:
        result = await asyncio.wait_for(
            hirag_instance.query_all(query), timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Query timed out after {timeout_seconds} seconds")
        return f"Query timed out after {timeout_seconds} seconds. Please try a simpler query or increase the timeout."

    except Exception as e:
        logger.error(f"Error in hi_search: {e}")
        return f"Search error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
