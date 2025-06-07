from dataclasses import dataclass
from typing import List, Literal, Optional

import lancedb

from hirag_prod._utils import EmbeddingFunc
from hirag_prod.storage.base_vdb import BaseVDB

from .retrieval_strategy_provider import RetrievalStrategyProvider

THRESHOLD_DISTANCE = 0.7
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

    async def upsert_text(
        self,
        text_to_embed: str,
        properties: dict,
        table: Optional[lancedb.AsyncTable] = None,
        table_name: Optional[str] = None,
        mode: Literal["append", "overwrite"] = "append",
    ) -> lancedb.AsyncTable:
        """Generate embedding for the text and add the embedding and metadata to the table

        Args:
            text_to_embed (str): the text to embed
            metadata (dict): other metadata to add to the table
            table (Optional[lancedb.AsyncTable]): If not None, use the existing table.
            table_name (Optional[str]): Required if table is None.
        """
        embedding = await self.embedding_func(text_to_embed)
        properties["vector"] = embedding[0].tolist()
        if table is None:
            if table_name is None:
                raise ValueError("table_name is required if table is None")
            try:
                return await self.db.create_table(table_name, data=[properties])
            except ValueError as e:
                if "already exists" in str(e):
                    table = await self.db.open_table(table_name)
                    await table.add([properties], mode=mode)
                    return table
                raise e
        else:
            await table.add([properties], mode=mode)
            return table

    def add_filter_by_document_keys(self, document_list: Optional[List[str]], query):
        filter_expr = None
        if document_list is not None and len(document_list) > 0:
            document_list = [f"'{doc}'" for doc in document_list]
            filter_expr = f"document_key in ({','.join(document_list)})"
            # prefilter before searching the nearest neighbors
            query = query.where(filter_expr)
        return query

    def add_filter_by_require_access(
        self, require_access: Optional[Literal["private", "public"]], query
    ):
        if require_access is not None:
            query = query.where(f"private = {require_access == 'private'}")
        return query

    async def query(
        self,
        query: str,
        table: lancedb.AsyncTable,
        topk: Optional[int] = TOPK,
        document_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = ["filename", "text"],
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
        topn: Optional[int] = TOPN,
    ) -> List[dict]:
        """Search the chunk table by text and return the topk results

        Args:
            query (str): The query string.
            table (Union[lancedb.AsyncTable, lancedb.table.Table]): The lancedb table to search.
            topk (Optional[int]): The number of results to return. Defaults to 10.
            document_list (Optional[List[str]]): The list of documents (by document_key url) to search in.
            require_access (Optional[Literal["private", "public"]]): The access level of the documents to search in.
            columns_to_select (Optional[List[str]]): The columns to select from the table.
            distance_threshold (Optional[float]): The distance (cosine) threshold to use.
                The distance is calculated by the cosine distance between the query and the embeddings.
                The distance is between 0 and 1, where 0 is the most similar and 1 is the least similar.
                The default value is 0.7.
                If the distance is greater than the threshold, the result will be excluded.
            topn (Optional[int]): The number of results to rerank. Defaults to 4.

        Returns:
            List[dict]: _description_
        """
        query_text = query
        embedding = await self.embedding_func(query_text)
        embedding = embedding[0].tolist()
        if columns_to_select is None:
            columns_to_select = [
                "text",
                "document_key",
                "filename",
                "private",
            ]

        if topk is None:
            topk = self.strategy_provider.default_topk

        # We use the cosine distance to calculate the distance between the query and the embeddings
        query = table.query().nearest_to(embedding).distance_type("cosine")
        query = self.add_filter_by_document_keys(document_list, query)
        query = self.add_filter_by_require_access(require_access, query)

        if distance_threshold is not None:
            query = query.distance_range(upper_bound=distance_threshold)
        query = query.select(columns_to_select).limit(topk)

        if topn is None:
            topn = self.strategy_provider.default_topn
        reranked_query = self.strategy_provider.rerank_chunk_query(
            query, query_text, topn
        )
        return await reranked_query.to_list()

    async def get_table(self, table_name: str) -> str:
        """Get a table from the database."""
        table = await self.db.open_table(table_name)
        data = await table.to_arrow()
        return data

    async def clean_up(self):
        pass
