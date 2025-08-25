"""HiRAG MCP Server"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Union

from mcp.server.fastmcp import Context, FastMCP

from hirag_prod.hirag import HiRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hirag_mcp.server")

DEFAULT_TIMEOUT = int(os.getenv("HIRAG_QUERY_TIMEOUT", "100"))


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[dict]:
    """manage lifespan of HiRAG instance"""
    # global hirag_instance
    logger.info("Initializing HiRAG instance...")
    hirag_instance = await HiRAG.create()
    try:
        yield {"hirag": hirag_instance}
    finally:
        await hirag_instance.clean_up()
        logger.info("HiRAG MCP server connection closed")


mcp = FastMCP("HiRAG MCP Server", lifespan=lifespan)


@mcp.tool()
async def hi_search(query: str, ctx: Context = None) -> Union[str, dict]:
    """
    Retrieval Augmented Generation search over the knowledge base.

    Args:
        query: The search query text

    Returns:
        The search results as text
    """
    # Validate the input
    if not query or not query.strip():
        return "Error: Query cannot be empty"

    try:
        hirag_instance = ctx.request_context.lifespan_context.get("hirag")
        if not hirag_instance:
            raise ValueError("HiRAG instance not initialized")
    except (KeyError, AttributeError) as e:
        logger.error(f"Context access error: {e}")
        return "Service temporarily unavailable"
    except Exception as e:
        logger.error(f"Unexpected error accessing HiRAG instance: {e}")
        return "Internal server error"

    try:
        result = await asyncio.wait_for(
            hirag_instance.query(query, summary=True), timeout=DEFAULT_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Query timed out after {DEFAULT_TIMEOUT} seconds")
        return f"Query timed out after {DEFAULT_TIMEOUT} seconds. Please try a simpler query or increase the timeout."

    except Exception as e:
        logger.error(f"Error in kb_search: {e}")
        return f"Search error: {str(e)}"


@mcp.tool()
async def hi_set_language(language: str, ctx: Context = None) -> str:
    """
    Set the language for HiRAG summary generation.

    Args:
        language: The language code ("en" for English, "cn" for Chinese)

    Returns:
        Confirmation message
    """
    if not language:
        return "Error: Language parameter cannot be empty"

    try:
        hirag_instance = ctx.request_context.lifespan_context.get("hirag")
        if not hirag_instance:
            raise ValueError("HiRAG instance not initialized")
    except (KeyError, AttributeError) as e:
        logger.error(f"Context access error: {e}")
        return "Service temporarily unavailable"
    except Exception as e:
        logger.error(f"Unexpected error accessing HiRAG instance: {e}")
        return "Internal server error"

    try:
        await hirag_instance.set_language(language)
        logger.info(f"Language successfully set to: {language}")
        return f"Language successfully set to: {language}"
    except ValueError as e:
        logger.error(f"Invalid language: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return f"Error setting language: {str(e)}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
