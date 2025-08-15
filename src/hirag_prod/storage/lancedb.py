import logging
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import lancedb

from hirag_prod._utils import EmbeddingFunc
from hirag_prod.storage.base_vdb import BaseVDB

from .retrieval_strategy_provider import RetrievalStrategyProvider

logger = logging.getLogger(__name__)

THRESHOLD_DISTANCE = 0.8
TOPK = 5
TOPN = 4


@dataclass
class LanceDB(BaseVDB):
    embedding_func: EmbeddingFunc
    db: lancedb.AsyncConnection
    strategy_provider: RetrievalStrategyProvider

    @classmethod
    async def create(
        cls,
        embedding_func: EmbeddingFunc,
        db_url: str,
        strategy_provider: RetrievalStrategyProvider,
    ):
        db = await lancedb.connect_async(db_url)
        return cls(embedding_func, db, strategy_provider)

    async def _ensure_table_exists_and_add_data(
        self,
        table_name: str,
        data: List[dict],
        table: Optional[lancedb.AsyncTable] = None,
    ) -> lancedb.AsyncTable:
        """
        Ensure table exists and add data to it. This method handles table creation
        and data insertion in a robust way that always appends data.

        Args:
            table_name: Name of the table
            data: List of dictionaries containing the data to insert
            table: Optional existing table instance

        Returns:
            The table instance with data added
        """
        if table is not None:
            # Table instance provided, just add data
            await table.add(data)
            return table

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
        table: Optional[lancedb.AsyncTable] = None,
        table_name: Optional[str] = None,
        mode: Literal["append", "overwrite"] = "append",
    ) -> lancedb.AsyncTable:
        """
        Insert a single text with its embedding into the table.
        This method ensures data is always appended, never overwritten.

        Args:
            text_to_embed: Text to generate embedding for
            properties: Metadata properties for the record
            table: Optional existing table instance
            table_name: Table name (required if table is None)
            mode: Legacy parameter, always appends regardless of value

        Returns:
            The table instance
        """
        if table is None and table_name is None:
            raise ValueError("Either table or table_name must be provided")

        table_info = table_name if table is None else f"table_instance_{id(table)}"

        start = time.perf_counter()

        # Generate embedding
        embedding = await self.embedding_func([text_to_embed])
        properties["vector"] = embedding[0].tolist()

        # Ensure table exists and add data
        result = await self._ensure_table_exists_and_add_data(
            table_name if table_name else table.name, [properties], table
        )

        elapsed = time.perf_counter() - start
        logger.info(
            f"[upsert_text] Successfully added 1 record to table '{table_info}', elapsed={elapsed:.3f}s"
        )
        return result

    async def upsert_texts(
        self,
        texts_to_embed: list[str],
        properties_list: list[dict],
        table: Optional[lancedb.AsyncTable] = None,
        table_name: Optional[str] = None,
        mode: Literal["append", "overwrite"] = "append",
    ) -> lancedb.AsyncTable:
        """
        Batch insert multiple texts with embeddings.
        This method ensures data is always appended, never overwritten.

        Args:
            texts_to_embed: List of texts to embed.
            properties_list: Corresponding metadata for each text.
            table: Existing table instance. If None, table_name must be provided.
            table_name: Name of the table when table is None.
            mode: Legacy parameter, always appends regardless of value

        Returns:
            The table where the data was inserted.
        """
        if table is None and table_name is None:
            raise ValueError("Either table or table_name must be provided")

        if len(texts_to_embed) != len(properties_list):
            raise ValueError(
                "texts_to_embed and properties_list must have the same length"
            )

        table_info = table_name if table is None else f"table_instance_{id(table)}"

        start = time.perf_counter()

        # Generate embeddings for all texts
        embeddings = await self.embedding_func(texts_to_embed)
        for props, emb in zip(properties_list, embeddings):
            props["vector"] = emb.tolist()

        # Ensure table exists and add data
        result = await self._ensure_table_exists_and_add_data(
            table_name if table_name else table.name, properties_list, table
        )

        elapsed = time.perf_counter() - start
        logger.info(
            f"[upsert_texts] Successfully added {len(properties_list)} records to table '{table_info}', elapsed={elapsed:.3f}s"
        )
        return result

    def add_filter_by_uri(self, uri_list: Optional[List[str]], query):
        filter_expr = None
        if uri_list is not None and len(uri_list) > 0:
            uri_list = [f"'{uri}'" for uri in uri_list]
            filter_expr = f"uri in ({','.join(uri_list)})"
            # prefilter before searching the nearest neighbors
            query = query.where(filter_expr)
        return query

    def add_filter_by_require_access(
        self, require_access: Optional[Literal["private", "public"]], query
    ):
        if require_access is not None:
            query = query.where(f"private = {require_access == 'private'}")
        return query

    def add_filter_by_scope(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        query,
    ):
        if workspace_id:
            query = query.where(f"workspace_id == '{workspace_id}'")
        if knowledge_base_id:
            query = query.where(f"knowledge_base_id == '{knowledge_base_id}'")
        return query

    async def query(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        table: lancedb.AsyncTable,
        topk: Optional[int] = TOPK,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = ["filename", "text"],
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
        topn: Optional[int] = TOPN,
        rerank: bool = True,
    ) -> List[dict]:
        """Search the chunk table by text and return the topk results

        Args:
            query (str): The query string.
            table (Union[lancedb.AsyncTable, lancedb.table.Table]): The lancedb table to search.
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
        query_text = query
        embedding = await self.embedding_func([query_text])
        embedding = embedding[0].tolist()
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "filename",
                "private",
            ]

        if topk is None:
            topk = self.strategy_provider.default_topk

        # We use the cosine distance to calculate the distance between the query and the embeddings
        query = table.query().nearest_to(embedding).distance_type("cosine")

        # Set nprobes to avoid the warning - adjust the value based on your needs
        # Higher values = more accurate but slower, lower values = faster but less accurate
        # Common values: 20-50 for balanced performance
        if hasattr(query, "nprobes"):
            query = query.nprobes(20)  # adjust this value as needed

        query = self.add_filter_by_uri(uri_list, query)
        query = self.add_filter_by_require_access(require_access, query)
        query = self.add_filter_by_scope(workspace_id, knowledge_base_id, query)

        if distance_threshold is not None:
            query = query.distance_range(upper_bound=distance_threshold)
        query = query.select(columns_to_select).limit(topk)

        if topn is None:
            topn = self.strategy_provider.default_topn

        if rerank:
            reranked_query = self.strategy_provider.rerank_chunk_query(
                query, query_text, topn
            )
            return await reranked_query.to_list()

        return await query.to_list()

    async def query_by_keys(
        self,
        key_value: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table: lancedb.AsyncTable,
        key_column: str = "document_key",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Query the table by document key and return matching results.

        Args:
            document_key (str): The document key value to search for.
            table (lancedb.AsyncTable): The lancedb table to search.
            key_column (str): The name of the column containing document keys. Defaults to "document_key".
            columns_to_select (Optional[List[str]]): The columns to select from the table.
                If None, defaults to common columns.
            limit (Optional[int]): Maximum number of results to return. If None, returns all matches.

        Returns:
            List[dict]: List of matching records from the table.
        """
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "uri",
                "filename",
                "private",
                key_column,
            ]

        # Build the query with filter for the document key
        query = (
            table.query()
            .where(f"{key_column} IN ({', '.join(map(repr, key_value))})")
            .select(columns_to_select)
        )
        query = self.add_filter_by_scope(workspace_id, knowledge_base_id, query)

        # Apply limit if specified
        if limit is not None:
            query = query.limit(limit)

        return await query.to_list()

    async def get_table(self, table_name: str) -> str:
        """Get a table from the database."""
        table = await self.db.open_table(table_name)
        data = await table.to_arrow()
        return data

    async def clean_up(self):
        pass
