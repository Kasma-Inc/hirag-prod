import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from hirag_prod._utils import AsyncEmbeddingFunction, log_error_info
from hirag_prod.reranker.utils import apply_reranking
from hirag_prod.resources.functions import get_db_engine, get_db_session_maker
from hirag_prod.schema import Base as PGBase
from hirag_prod.schema import Chunk, File, Item, Triplets
from hirag_prod.storage.base_vdb import BaseVDB
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

logger = logging.getLogger(__name__)

THRESHOLD_DISTANCE = 0.8
TOPK = 5
TOPN = 4


# extends to implement PostgreSQL-based vdb with pgvector support
class PGVector(BaseVDB):
    """A vector database interface using PostgreSQL with pgvector extension.

    Provides methods to upsert text embeddings and query them based on similarity
    or specific keys, leveraging SQLAlchemy for database operations.

    Attributes:
        embedding_func (Optional[AsyncEmbeddingFunction]): Function to generate text embeddings.
        strategy_provider (RetrievalStrategyProvider): Handles result ranking.
        vector_type (str): Type of vector storage ('vector' or 'halfvec').
        tables (dict): Mapping of table names to model classes.
    """

    def __init__(
        self,
        embedding_func: Optional[AsyncEmbeddingFunction],
        strategy_provider: "RetrievalStrategyProvider",
        vector_type: Literal["vector", "halfvec"] = "halfvec",
    ):
        self.embedding_func = embedding_func
        self.strategy_provider = strategy_provider
        self.vector_type = vector_type
        self.tables = {
            "Chunks": Chunk,
            "Files": File,
            "Triplets": Triplets,
            "Items": Item,
        }  # mapping of table names to model creation functions

    def _to_list(self, embedding):
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    # retrieves or creates a SQLAlchemy model for the specified table
    def get_model(self, table_name: str):
        model = self.tables.get(table_name)
        if not model:
            raise ValueError(f"No table found for table {table_name}")
        return model

    # create a PGVector instance
    @classmethod
    def create(
        cls,
        embedding_func: Optional[AsyncEmbeddingFunction],
        strategy_provider: "RetrievalStrategyProvider",
        vector_type: Literal["vector", "halfvec"] = "halfvec",
    ):
        instance = cls(embedding_func, strategy_provider, vector_type=vector_type)
        return instance

    # upserts a single text embedding into the specified table
    async def upsert_text(
        self,
        text_to_embed: str,
        properties: dict,
        table_name: str,
        mode: Literal["append", "overwrite"] = "append",
    ):
        return (
            await self.upsert_texts([text_to_embed], [properties], table_name, mode)
        )[0]

    # batch upserts text embeddings into the specified table
    async def upsert_texts(
        self,
        texts_to_embed: List[str],
        properties_list: List[dict],
        table_name: str,
        mode: Literal["append", "overwrite"] = "append",
    ):
        if len(texts_to_embed) != len(properties_list):
            raise ValueError(
                "texts_to_embed and properties_list must have the same length"
            )

        model = self.get_model(table_name)

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            embs = await self.embedding_func(texts_to_embed)
            now = datetime.now()
            rows = []
            for props, emb in zip(properties_list, embs):
                vec = self._to_list(emb)
                row = dict(props or {})
                row["vector"] = vec
                row["updatedAt"] = now
                rows.append(row)

            table = model.__table__
            pk_cols = [c.name for c in table.primary_key.columns]

            if not rows:
                elapsed = time.perf_counter() - start
                logger.info(
                    f"[upsert_texts] No rows to upsert into '{table_name}', mode={mode}, elapsed={elapsed:.3f}s"
                )
                return rows

            try:
                cols_per_row = len(rows[0])
            except Exception as e:
                log_error_info(logging.WARNING, "Rows are empty", e)
                cols_per_row = None

            if not cols_per_row:
                all_keys = set()
                for r in rows:
                    all_keys.update(r.keys())
                cols_per_row = max(1, len(all_keys))

            # Compute a safe batch size based on parameter budget to avoid exceeding
            # PostgreSQL's 65535 bind parameter limit for a single statement.
            # We conservatively cap at 60000 total parameters per statement.
            param_budget = 60000
            max_batch_size = max(1, param_budget // cols_per_row)

            total = len(rows)
            processed = 0
            for i in range(0, total, max_batch_size):
                batch = rows[i : i + max_batch_size]
                ins = insert(table).values(batch)
                stmt = ins.on_conflict_do_nothing(
                    index_elements=[table.c[name] for name in pk_cols]
                )
                await session.execute(stmt)
                processed += len(batch)

            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                "[upsert_texts] Upserted %d into '%s' in batches (batch_size<=%d), mode=%s, elapsed=%.3fs",
                len(rows),
                table_name,
                max_batch_size,
                mode,
                elapsed,
            )
            return rows

    async def clean_table(
        self,
        table_name: str,
        where: Dict[str, Any],
    ):
        # Clean all rows matching the where criteria
        # where {"key": "value"}
        model = self.get_model(table_name)

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            stmt = delete(model).where(
                *[getattr(model, k) == v for k, v in where.items()]
            )
            await session.execute(stmt)
            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                f"[clean_table] Cleaned table '{table_name}', elapsed={elapsed:.3f}s"
            )

    async def upsert_file(
        self,
        file: File,
        table_name: str = "Files",
        mode: Literal["append", "overwrite"] = "append",
    ):
        model = self.get_model(table_name)

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            now = datetime.now()
            row = dict(file)
            row["updatedAt"] = now

            table = model.__table__
            pk_cols = [c.name for c in table.primary_key.columns]
            ins = insert(table).values(row)
            stmt = ins.on_conflict_do_nothing(
                index_elements=[table.c[name] for name in pk_cols]
            )

            await session.execute(stmt)
            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                f"[upsert_texts] Upserted file information into '{table_name}', mode={mode}, elapsed={elapsed:.3f}s"
            )
            return row

    async def query(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topk: Optional[int] = TOPK,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
        topn: Optional[int] = TOPN,
        rerank: bool = False,
    ) -> List[dict]:
        if columns_to_select is None:
            columns_to_select = ["text", "uri", "fileName", "private"]

        if topk is None:
            topk = self.strategy_provider.default_topk
        if topn is None:
            topn = self.strategy_provider.default_topn

        if topn > topk:
            raise ValueError(f"topn ({topn}) must be <= topk ({topk})")

        model = self.get_model(table_name)

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            q_emb = (await self.embedding_func([query]))[0]
            q_emb = self._to_list(q_emb)

            distance_expr = model.vector.cosine_distance(q_emb).label("distance")

            stmt = select(model, distance_expr)

            if uri_list and hasattr(model, "uri"):
                stmt = stmt.where(model.uri.in_(uri_list))
            if require_access is not None and hasattr(model, "private"):
                stmt = stmt.where(model.private == (require_access == "private"))
            if workspace_id and hasattr(model, "workspaceId"):
                stmt = stmt.where(model.workspaceId == workspace_id)
            if knowledge_base_id and hasattr(model, "knowledgeBaseId"):
                stmt = stmt.where(model.knowledgeBaseId == knowledge_base_id)

            if distance_threshold is not None:
                stmt = stmt.where(distance_expr < float(distance_threshold))

            stmt = stmt.order_by(distance_expr.asc()).limit(topk)

            result = await session.execute(stmt)
            rows = result.all()

            scored = []
            for row, dist in rows:
                payload = {
                    col: getattr(row, col)
                    for col in columns_to_select
                    if hasattr(row, col)
                }
                payload["distance"] = dist
                scored.append(payload)

            scored = (
                await apply_reranking(query, scored, topn, topk) if rerank else scored
            )

            elapsed = time.perf_counter() - start
            logger.info(
                f"[query] Retrieved {len(scored)} records from '{table_name}', elapsed={elapsed:.3f}s"
            )
            return scored

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
        model = self.get_model(table_name)
        async with get_db_session_maker()() as session:
            stmt = select(model)
            if key_value:
                stmt = stmt.where(getattr(model, key_column).in_(key_value))
            if workspace_id and hasattr(model, "workspaceId"):
                stmt = stmt.where(model.workspaceId == workspace_id)
            if knowledge_base_id and hasattr(model, "knowledgeBaseId"):
                stmt = stmt.where(model.knowledgeBaseId == knowledge_base_id)
            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = list(result.scalars().all())

            if columns_to_select is None:  # query all if nothing provided
                columns_to_select = [c.name for c in model.__table__.columns]

            out: List[dict] = []
            for r in rows:
                rec = {}
                for col in columns_to_select:
                    if not hasattr(r, col):
                        continue
                    val = getattr(r, col)
                    # normalize HalfVector/Vector -> python list[float]
                    if hasattr(val, "to_list"):
                        rec[col] = val.to_list()
                    elif hasattr(val, "tolist"):
                        rec[col] = val.tolist()
                    else:
                        rec[col] = list(val) if isinstance(val, (list, tuple)) else val
                out.append(rec)
            return out

    async def get_existing_document_keys(
        self,
        uri: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
    ) -> List[str]:
        model = self.get_model(table_name)
        async with get_db_session_maker()() as s:
            stmt = (
                select(model)
                .where(model.uri == uri)
                .where(model.workspaceId == workspace_id)
                .where(model.knowledgeBaseId == knowledge_base_id)
            )
            result = await s.execute(stmt)
            rows = list(result.scalars().all())
            return [
                getattr(r, "documentKey", None)
                for r in rows
                if getattr(r, "documentKey", None)
            ]

    async def get_table(self, table_name: str) -> List[dict]:
        model = self.get_model(table_name)
        async with get_db_session_maker()() as session:
            result = await session.execute(select(model))
            rows = list(result.scalars().all())
            out = []
            for r in rows:
                rec = {}
                for col in r.__table__.columns.keys():  # type: ignore
                    val = getattr(r, col, None)
                    if hasattr(val, "to_list"):
                        rec[col] = val.to_list()
                    elif hasattr(val, "tolist"):
                        rec[col] = val.tolist()
                    else:
                        rec[col] = val
                out.append(rec)
            return out

    async def _init_vdb(self, embedding_dimension: int, *args, **kwargs):
        async with get_db_engine().begin() as conn:
            Chunks = self.get_model("Chunks")
            Files = self.get_model("Files")
            Triplets = self.get_model("Triplets")
            Items = self.get_model("Items")

            def _create(sync_conn):
                PGBase.metadata.create_all(
                    bind=sync_conn,
                    tables=[
                        Chunks.__table__,
                        Files.__table__,
                        Triplets.__table__,
                        Items.__table__,
                    ],
                )

            await conn.run_sync(_create)
