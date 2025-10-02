import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select

from hirag_prod._utils import log_error_info, retry_async
from hirag_prod.configs.functions import get_hi_rag_config, get_init_config
from hirag_prod.exceptions import StorageError
from hirag_prod.resources.functions import get_resource_manager
from hirag_prod.schema import (
    Chunk,
    File,
    Item,
    Relation,
)
from hirag_prod.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
)
from hirag_prod.storage.pgvector import PGVector

logger = logging.getLogger("HiRAG")
TOPK = get_init_config().default_query_top_k
TOPN = get_init_config().default_query_top_n


class StorageManager:
    """Unified manager for vector database and graph database operations."""

    def __init__(
        self,
        vdb: BaseVDB,
        gdb: BaseGDB,
    ):
        self.vdb = vdb
        self.gdb = gdb
        self.files_table = None

    async def initialize(self) -> None:
        """Initialize storage tables"""
        try:
            await self.vdb._init_vdb(
                embedding_dimension=get_hi_rag_config().embedding_dimension
            )
        except Exception as e:
            log_error_info(
                logging.ERROR,
                "Failed to initialize VDB",
                e,
                raise_error=True,
                new_error_class=StorageError,
            )

    @retry_async()
    async def clean_vdb_document(self, where: Dict[str, Any]) -> None:
        await self.vdb.clean_table(table_name="Chunks", where=where)
        await self.vdb.clean_table(table_name="Triplets", where=where)
        await self.vdb.clean_table(table_name="Items", where=where)
        await self.vdb.clean_table(table_name="Graph", where=where)
        await self.vdb.clean_table(table_name="Nodes", where=where)

    @retry_async()
    async def clean_vdb_file(self, where: Dict[str, Any]) -> None:
        await self.vdb.clean_table(table_name="Files", where=where)

    @retry_async()
    async def upsert_chunks_to_vdb(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        def _embed_text(c: Chunk) -> str:
            if getattr(c, "chunkType", None) in ["excel_sheet", "table", "picture"]:
                cap = (getattr(c, "caption", "") or "").strip()
                if cap:
                    return cap
            return (getattr(c, "text", "") or "").strip()

        texts_to_embed = [_embed_text(c) for c in chunks]
        await self.vdb.upsert_texts(
            texts_to_upsert=texts_to_embed,
            properties_list=chunks,
            table_name="Chunks",
            mode="append",
        )

    @retry_async()
    async def upsert_items_to_vdb(self, items: List[Item]) -> None:
        if not items:
            return

        def _embed_text(c: Item) -> str:
            if getattr(c, "chunkType", None) in ["excel_sheet", "table", "picture"]:
                cap = (getattr(c, "caption", "") or "").strip()
                if cap:
                    return cap
            return (getattr(c, "text", "") or "").strip()

        texts_to_embed = [_embed_text(c) for c in items]
        await self.vdb.upsert_texts(
            texts_to_upsert=texts_to_embed,
            properties_list=items,
            table_name="Items",
            with_tokenization=True,
            with_translation=True,
            mode="append",
        )

    @retry_async()
    async def upsert_file_to_vdb(self, file: File) -> None:
        if not file:
            return
        await self.vdb.upsert_file(
            file=file,
            mode="append",
        )

    @retry_async()
    async def upsert_relations_to_vdb(self, relations: List[Relation]) -> None:
        if not relations:
            return
        filtered = [r for r in relations if not r.source.startswith("chunk-")]
        if not filtered:
            return
        texts_to_embed = [r.properties.get("description", "") for r in filtered]
        # Create properties list by mapping relation properties to Triplets schema
        properties_list = []
        for r in filtered:
            # Start with required Triplets fields
            triplet_properties = {
                "source": r.source,
                "target": r.target,
                "description": r.properties.get("description", ""),
            }

            # Map all other relation properties directly (now in camelCase)
            for key, value in r.properties.items():
                if key not in triplet_properties and value is not None:
                    triplet_properties[key] = value

            properties_list.append(triplet_properties)
        await self.vdb.upsert_texts(
            texts_to_upsert=texts_to_embed,
            properties_list=properties_list,
            table_name="Triplets",
            mode="append",
        )

    async def get_existing_chunks(
        self, uri: str, workspace_id: str, knowledge_base_id: str
    ) -> List[str]:
        try:
            return await self.vdb.get_existing_document_keys(
                uri=uri,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                table_name="Chunks",
            )
        except Exception as e:
            log_error_info(logging.WARNING, "Failed to get existing chunks", e)
            return []

    async def query_chunks(
        self,
        query: Union[str, List[str]],
        workspace_id: str,
        knowledge_base_id: str,
        topk: Optional[int] = None,
        topn: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = await self.vdb.query(
            query=query,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            table_name="Chunks",
            topk=topk or TOPK,
            topn=topn or TOPN,
            columns_to_select=[
                "text",
                "uri",
                "fileName",
                "private",
                "updatedAt",
                "documentKey",
                "extractedTimestamp",
            ],
        )
        return rows

    async def query_triplets(
        self,
        query: Union[str, List[str]],
        workspace_id: str,
        knowledge_base_id: str,
        topk: int = None,
        topn: int = None,
    ) -> List[Dict[str, Any]]:
        rows = await self.vdb.query(
            query=query,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            table_name="Triplets",
            topk=topk if topk else TOPK,
            topn=topn if topn else TOPN,
            columns_to_select=["source", "target", "description", "fileName"],
        )
        return rows

    async def query_by_keys(
        self,
        chunk_ids: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        columns_to_select: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        rows = await self.vdb.query_by_keys(
            key_value=chunk_ids,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            table_name="Chunks",
            key_column="documentKey",
            columns_to_select=columns_to_select,
        )
        return rows

    # ------------------------------ Health/Cleanup ------------------------------
    async def health_check(self) -> Dict[str, bool]:
        health: Dict[str, bool] = {}
        try:
            if isinstance(self.vdb, LanceDB):
                await self.vdb.db.table_names()
            elif isinstance(self.vdb, PGVector):
                async with get_resource_manager().get_session_maker()() as s:
                    await s.execute(select(1))
            health["vdb"] = True
        except Exception as e:
            log_error_info(logging.WARNING, "VDB health check failed", e)
            health["vdb"] = False

        try:
            await self.gdb.health_check() if hasattr(self.gdb, "health_check") else None
            health["gdb"] = True
        except Exception as e:
            log_error_info(logging.WARNING, "GDB health check failed", e)
            health["gdb"] = False
        return health

    async def cleanup(self) -> None:
        try:
            await self.gdb.clean_up()
        except Exception as e:
            log_error_info(logging.WARNING, "Cleanup failed", e)
