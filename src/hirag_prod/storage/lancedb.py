import logging
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import lancedb

from hirag_prod._utils import AsyncEmbeddingFunction
from hirag_prod.configs.functions import get_init_config
from hirag_prod.reranker.utils import apply_reranking
from hirag_prod.storage.base_vdb import BaseVDB
from hirag_prod.storage.lance_schema import get_chunks_schema, get_relations_schema
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

logger = logging.getLogger(__name__)

THRESHOLD_DISTANCE = get_init_config().default_distance_threshold
TOPK = get_init_config().default_query_top_k
TOPN = get_init_config().default_query_top_n


@dataclass
class LanceDB(BaseVDB):
    embedding_func: Optional[AsyncEmbeddingFunction]
    db: lancedb.AsyncConnection
    strategy_provider: RetrievalStrategyProvider

    @classmethod
    async def create(
        cls,
        embedding_func: Optional[AsyncEmbeddingFunction],
        db_url: str,
        strategy_provider: RetrievalStrategyProvider,
    ):
        db = await lancedb.connect_async(db_url)
        return cls(embedding_func, db, strategy_provider)

    async def _ensure_table_exists_and_add_data(
        self,
        table_name: str,
        data: List[dict],
    ) -> lancedb.AsyncTable:
        """
        Ensure table exists and add data to it. This method handles table creation
        and data insertion in a robust way that always appends data.

        Args:
            table_name: Name of the table
            data: List of dictionaries containing the data to insert

        Returns:
            The table instance with data added
        """
        # Try to open existing table first
        try:
            table = await self.db.open_table(table_name)
            logger.info(
                f"[_ensure_table_exists_and_add_data] Opened existing table '{table_name}'"
            )
            await table.add(data)
            return table
        except (ValueError, FileNotFoundError) as e:
            # Table doesn't exist, create it
            logger.info(
                f"[_ensure_table_exists_and_add_data] Creating new table '{table_name}' (table not found: {e})"
            )
            try:
                table = await self.db.create_table(table_name, data=data)
                logger.info(
                    f"[_ensure_table_exists_and_add_data] Successfully created table '{table_name}' with {len(data)} rows"
                )
                return table
            except ValueError as create_error:
                if "already exists" in str(create_error).lower():
                    # Race condition: table was created by another process
                    logger.info(
                        f"[_ensure_table_exists_and_add_data] Table '{table_name}' was created by another process, opening and adding data"
                    )
                    table = await self.db.open_table(table_name)
                    await table.add(data)
                    return table
                else:
                    raise create_error

    async def upsert_text(
        self,
        text_to_embed: str,
        properties: dict,
        table_name: str,
        mode: Literal["append", "overwrite"] = "append",
    ) -> lancedb.AsyncTable:
        """
        Insert a single text with its embedding into the table.
        This method ensures data is always appended, never overwritten.

        Args:
            text_to_embed: Text to generate embedding for
            properties: Metadata properties for the record
            table_name: Table name to upsert data into.
            mode: Mode to use for upserting data.

        Returns:
            The table instance
        """
        start = time.perf_counter()

        # Generate embedding
        embedding = await self.embedding_func([text_to_embed])
        properties["vector"] = embedding[0].tolist()

        # Ensure table exists and add data
        result = await self._ensure_table_exists_and_add_data(table_name, [properties])

        elapsed = time.perf_counter() - start
        logger.info(
            f"[upsert_text] Successfully added 1 record to table '{table_name}', elapsed={elapsed:.3f}s"
        )
        return result

    async def upsert_texts(
        self,
        texts_to_embed: list[str],
        properties_list: list[dict],
        table_name: str,
        mode: Literal["append", "overwrite"] = "append",
    ) -> lancedb.AsyncTable:
        """
        Batch insert multiple texts with embeddings.
        This method ensures data is always appended, never overwritten.

        Args:
            texts_to_embed: List of texts to embed.
            properties_list: Corresponding metadata for each text.
            table_name: Name of the table to upsert data into.
            mode: Mode to use for upserting data.

        Returns:
            The table where the data was inserted.
        """
        if len(texts_to_embed) != len(properties_list):
            raise ValueError(
                "texts_to_embed and properties_list must have the same length"
            )

        start = time.perf_counter()

        # Generate embeddings for all texts
        embeddings = await self.embedding_func(texts_to_embed)
        for props, emb in zip(properties_list, embeddings):
            props["vector"] = emb.tolist()

        # Ensure table exists and add data
        result = await self._ensure_table_exists_and_add_data(
            table_name, properties_list
        )

        elapsed = time.perf_counter() - start
        logger.info(
            f"[upsert_texts] Successfully added {len(properties_list)} records to table '{table_name}', elapsed={elapsed:.3f}s"
        )
        return result

    def add_filter_by_uri(self, uri_list: Optional[List[str]]) -> Optional[str]:
        if uri_list:
            values = ", ".join(map(repr, uri_list))
            return f"uri IN ({values})"
        return None

    def add_filter_by_require_access(
        self, require_access: Optional[Literal["private", "public"]]
    ) -> Optional[str]:
        if require_access is not None:
            return f"private = {require_access == 'private'}"
        return None

    def add_filter_by_scope(
        self,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> str:
        return (
            f"`workspaceId` = '{workspace_id}' AND "
            f"`knowledgeBaseId` = '{knowledge_base_id}'"
        )

    async def query(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topk: Optional[int] = TOPK,
        topn: Optional[int] = TOPN,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = ["filename", "text"],
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
        rerank: bool = False,
    ) -> List[dict]:
        """Search the chunk table by text and return the topk results

        Args:
            query (str): The query string.
            workspace_id (str): The workspace ID.
            knowledge_base_id (str): The knowledge base ID.
            table_name (str): The name of the table to search.
            topk (Optional[int]): The number of results to return. Defaults to 10.
            uri_list (Optional[List[str]]): The list of documents (by uri) to search in.
            require_access (Optional[Literal["private", "public"]]): The access level of the documents to search in.
            columns_to_select (Optional[List[str]]): The columns to select from the table.
            distance_threshold (Optional[float]): The distance (cosine) threshold to use.
                The distance is calculated by the cosine distance between the query and the embeddings.
                The distance is between 0 and 1, where 0 is the most similar and 1 is the least similar.
                The default value is 0.7.
                If the distance is greater than the threshold, the result will be excluded.
            topn (Optional[int]): The number of results to rerank. Defaults to 4.
            rerank (bool): Whether to rerank the results. Defaults to True.

        Returns:
            List[dict]: _description_
        """
        table = await self.db.open_table(table_name)

        query_text = query
        embedding = await self.embedding_func([query_text])
        embedding = embedding[0].tolist()
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "fileName",
                "private",
            ]

        if topk is None:
            topk = self.strategy_provider.default_topk

        if topn is None:
            topn = self.strategy_provider.default_topn

        if topn > topk:
            raise ValueError(f"topn ({topn}) must be <= topk ({topk})")

        q = table.query().nearest_to(embedding).distance_type("cosine")

        if hasattr(q, "nprobes"):
            q = q.nprobes(20)

        clauses = [
            self.add_filter_by_uri(uri_list),
            self.add_filter_by_require_access(require_access),
            self.add_filter_by_scope(workspace_id, knowledge_base_id),
        ]
        predicate = " AND ".join([c for c in clauses if c])
        if predicate:
            q = q.where(predicate)

        if distance_threshold is not None:
            q = q.distance_range(upper_bound=distance_threshold)
        q = q.select(columns_to_select).limit(topk)

        results = await q.to_list()
        return (
            await apply_reranking(query=query, results=results, topn=topn, topk=topk)
            if rerank
            else results
        )

    async def query_by_keys(
        self,
        key_value: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        key_column: str = "documentKey",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Query the table by document key and return matching results.

        Args:
            key_value (List[str]): The document key value to search for.
            workspace_id (str): The workspace ID.
            knowledge_base_id (str): The knowledge base ID.
            table_name (str): The name of the table to search.
            key_column (str): The name of the column containing document keys. Defaults to "documentKey".
            columns_to_select (Optional[List[str]]): The columns to select from the table.
                If None, defaults to common columns.
            limit (Optional[int]): Maximum number of results to return. If None, returns all matches.

        Returns:
            List[dict]: List of matching records from the table.
        """
        table = await self.db.open_table(table_name)

        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "fileName",
                "private",
                key_column,
            ]

        key_pred = f"`{key_column}` IN ({', '.join(map(repr, key_value))})"
        scope_pred = self.add_filter_by_scope(workspace_id, knowledge_base_id)
        predicate = f"{key_pred} AND {scope_pred}"

        q = table.query().where(predicate).select(columns_to_select)

        if limit is not None:
            q = q.limit(limit)

        return await q.to_list()

    async def get_existing_document_keys(
        self,
        uri: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
    ) -> List[str]:
        table = await self.db.open_table(table_name)
        where_clauses = [
            f"uri = '{uri}'",
            f"`workspaceId` = '{workspace_id}'",
            f"`knowledgeBaseId` = '{knowledge_base_id}'",
        ]
        where_expr = " and ".join(where_clauses)
        existing = await table.query().where(where_expr).to_list()
        return [row.get("documentKey") for row in existing]

    async def ensure_tables(
        self,
        embedding_dimension: int,
        chunks_table_name: str = "chunks",
        relations_table_name: str = "relations",
    ) -> None:
        names = await self.db.table_names()
        if chunks_table_name not in names:
            await self.db.create_table(
                chunks_table_name, schema=get_chunks_schema(embedding_dimension)
            )
        if relations_table_name not in names:
            await self.db.create_table(
                relations_table_name, schema=get_relations_schema(embedding_dimension)
            )

    async def get_table(self, table_name: str) -> str:
        """Get a table from the database."""
        table = await self.db.open_table(table_name)
        data = await table.to_arrow()
        return data

    async def _init_vdb(self, embedding_dimension: int, *args, **kwargs):
        names = await self.db.table_names()
        if "Chunks" not in names:
            await self.db.create_table(
                "Chunks", schema=get_chunks_schema(embedding_dimension)
            )
        if "Triplets" not in names:
            await self.db.create_table(
                "Triplets", schema=get_relations_schema(embedding_dimension)
            )
