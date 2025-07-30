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
async def naive_search(query: str, ctx: Context = None) -> str:
    """
    Retrieve the chunks over the knowledge base. The retrieval information is not comprehensive.
    But the retrieval speed is faster than hi_search.

    Args:
        query: The search query text

    Returns:
        The search results as text
    """
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

    result = await hirag_instance.query_chunks(query)

    return result


@mcp.tool()
async def hi_search(query: str, ctx: Context = None) -> Union[str, dict]:
    """
    Search for the chunks, entities and relations over the knowledge base. The retrieval information is more comprehensive than naive_search.
    But the retrieval speed is slower than naive_search.

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
            hirag_instance.query_all(query, summary=True), timeout=DEFAULT_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Query timed out after {DEFAULT_TIMEOUT} seconds")
        return f"Query timed out after {DEFAULT_TIMEOUT} seconds. Please try a simpler query or increase the timeout."

    except Exception as e:
        logger.error(f"Error in hi_search: {e}")
        return f"Search error: {str(e)}"

@mcp.tool()
async def hi_insert_chat(
    chat_id: str, role: str, content: str, ctx: Context = None
) -> str:
    """
    Insert a User / Assistant / Tool message into the chat history.

    Args:
        chat_id: Unique identifier for the chat session
        role: Role of the message sender (user, assistant, tool)
        content: Content of the message

    Returns:
        Success or error message
    """
    # Validate inputs
    if not chat_id or not chat_id.strip() or not content or not content.strip() or not role or not role.strip():
        return "Error: chat_id, role, and content cannot be empty"
    
    # Validate role
    valid_roles = {"user", "assistant", "tool"}
    if role.lower() not in valid_roles:
        return f"Error: role must be one of {valid_roles}"

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
        metrics = await hirag_instance.search_chat_history(chat_id, role.lower(), content)
        logger.info(f"Chat message inserted: chat_id={chat_id}, role={role}, content_length={len(content)}")
        return f"Chat message inserted successfully. Total processed chats: {metrics.processed_chats}"
    except Exception as e:
        logger.error(f"Error inserting chat message: {e}")
        return f"Error inserting chat message: {str(e)}"

@mcp.tool()
async def hi_search_chat(
    user_query: str, chat_id: str, role: str = None, ctx: Context = None
) -> Union[str, dict]:
    """
    Search the chat history for messages related to the user's query.

    Args:
        user_query: The search query to find relevant chat messages
        chat_id: Unique identifier for the chat session
        role: Optional role filter (user, assistant, tool)

    Returns:
        Search results as formatted string or error message
    """
    # Validate inputs
    if not user_query or not user_query.strip() or not chat_id or not chat_id.strip():
        return "Error: user_query and chat_id cannot be empty"
    
    # Validate role if provided
    if role:
        valid_roles = {"user", "assistant", "tool"}
        if role.lower() not in valid_roles:
            return f"Error: role must be one of {valid_roles} or None"

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
        results = await hirag_instance.search_chat_history(
            user_query=user_query, 
            chat_id=chat_id, 
            role=role.lower() if role else None
        )
        
        if not results:
            return f"No chat messages found for query '{user_query}' in chat '{chat_id}'"
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching chat history: {e}")
        return f"Error searching chat history: {str(e)}"

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
