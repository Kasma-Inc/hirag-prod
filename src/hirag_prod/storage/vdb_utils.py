#!/usr/bin/env python3
import logging
from typing import Any, Dict, List, Optional

from rich.console import Console

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_hi_rag_config
from hirag_prod.storage.pgvector import PGVector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()


async def get_chunk_info(
    chunk_ids: list[str],
    knowledge_base_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Optional[list[Dict[str, Any]]]:
    """Get a list of chunk records with its headers from vector database by its ids.

    Note: the chunk identifier is stored in the `document_key` column of the
    `chunks` table for PostgreSQL.

    Args:
        chunk_ids: list of chunk ids to get info for
        knowledge_base_id: The id of the knowledge base that the chunk is from (optional)
        workspace_id: The id of the workspace that the chunk is from (optional)

    Returns:
        A list of dicts of the chunk rows with its headers if found, otherwise None.
    """
    if not chunk_ids:
        return []

    if get_hi_rag_config().vdb_type == "pgvector":
        try:
            # Create PGVector instance
            vdb = PGVector.create(
                embedding_func=None,
                vector_type="halfvec",
            )

            base_chunks = await vdb.query_by_keys(
                key_value=chunk_ids,
                workspace_id=workspace_id or "",
                knowledge_base_id=knowledge_base_id or "",
                table_name="Chunks",
                key_column="documentKey",
            )

            return base_chunks

        except Exception as e:
            log_error_info(
                logging.ERROR, f"Failed to get chunks info for ids={chunk_ids}", e
            )
            return []
    else:
        raise NotImplementedError("This VDB type is not supported yet")


async def get_file_info(
    knowledge_base_id: str,
    workspace_id: str,
    uri: str,
) -> list[dict]:
    """Get file rows from 'Files' by scope (knowledgeBaseId, workspaceId) and uri."""
    if not knowledge_base_id or not workspace_id:
        raise ValueError("knowledge_base_id and workspace_id are required")

    if not uri:
        raise ValueError("uri is required")

    if get_hi_rag_config().vdb_type == "pgvector":
        try:
            vdb = PGVector.create(
                embedding_func=None,
                vector_type="halfvec",
            )
            results = await vdb.query_by_keys(
                key_value=[uri],
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                table_name="Files",
                key_column="uri",
                columns_to_select=None,
                limit=None,
            )
            return results
        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Failed to get file info by scope (pgvector) kb={knowledge_base_id}, ws={workspace_id}, uri={uri}",
                e,
            )
            return []
    else:
        raise NotImplementedError("This VDB type is not supported yet")


async def get_table_info_by_scope(
    table_name: str,
    knowledge_base_id: str,
    workspace_id: str,
    columns_to_select: Optional[List[str]] = None,
    additional_data_to_select: Optional[Dict[str, Any]] = None,
    additional_where_clause_list: Optional[Any] = None,
    order_by: Optional[List[Any]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Get table info by scope (knowledgeBaseId and workspaceId).

    Args:
        table_name: The name of the table to get info for
        knowledge_base_id: The id of the knowledge base that the table is from
        workspace_id: The id of the workspace that the table is from
        columns_to_select: The columns to select from the table
        additional_data_to_select: Additional data to select from the table
        additional_where_clause_list: Additional where clause list to use
        order_by: The order by clause to use
        limit: The limit to use for pagination

    Returns:
        A list of dicts of the table rows if found, otherwise an empty list.
    """
    if not knowledge_base_id or not workspace_id:
        raise ValueError("knowledge_base_id and workspace_id are required")

    results: list[dict[str, Any]] = []

    try:
        vdb = PGVector.create(
            embedding_func=None,
            vector_type="halfvec",
        )
        results = await vdb.query_by_keys(
            key_value=[],
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            table_name=table_name,
            key_column="documentKey",
            columns_to_select=columns_to_select,
            additional_data_to_select=additional_data_to_select,
            additional_where_clause_list=additional_where_clause_list,
            order_by=order_by,
            limit=limit,
        )
        return results
    except Exception as e:
        log_error_info(
            logging.ERROR,
            f"Failed to get chunk info by scope (pgvector) kb={knowledge_base_id}, ws={workspace_id}",
            e,
        )
        return results


async def get_chunk_info_by_scope(
    knowledge_base_id: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    """Get chunk info by scope (knowledgeBaseId and workspaceId).

    Args:
        knowledge_base_id: The id of the knowledge base that the chunk is from
        workspace_id: The id of the workspace that the chunk is from

    Returns:
        A list of dicts of the chunk rows if found, otherwise an empty list.
    """
    return await get_table_info_by_scope(
        table_name="Chunks",
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
    )


async def get_item_info_by_scope(
    knowledge_base_id: str,
    workspace_id: str,
    columns_to_select: Optional[List[str]] = None,
    additional_data_to_select: Optional[Dict[str, Any]] = None,
    additional_where_clause_list: Optional[Any] = None,
    order_by: Optional[List[Any]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Get item info by scope (knowledgeBaseId and workspaceId)."""
    return await get_table_info_by_scope(
        table_name="Items",
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
        columns_to_select=columns_to_select,
        additional_data_to_select=additional_data_to_select,
        additional_where_clause_list=additional_where_clause_list,
        order_by=order_by,
        limit=limit,
    )
