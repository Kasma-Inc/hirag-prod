import logging
import os
import time
from datetime import datetime
from typing import List, Literal, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from hirag_prod._utils import EmbeddingFunc
from hirag_prod.storage.base_vdb import BaseVDB
from hirag_prod.storage.pg_schema import Base as PGBase
from hirag_prod.storage.pg_schema import (
    create_chunks_model,
    create_file_model,
    create_triplets_model,
)
from hirag_prod.storage.pg_utils import DatabaseClient
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
        embedding_func (EmbeddingFunc): Function to generate text embeddings.
        db_client (DatabaseClient): Manages database connections.
        strategy_provider (RetrievalStrategyProvider): Handles result ranking.
        engine: SQLAlchemy async engine for database operations.
        vector_type (str): Type of vector storage ('vector' or 'halfvec').
        models (dict): Cache of SQLAlchemy table models.
        factories (dict): Mapping of table names to model factory functions.
    """

    def __init__(
        self,
        embedding_func: EmbeddingFunc,
        db_client: DatabaseClient,
        strategy_provider: "RetrievalStrategyProvider",
        vector_type: Literal["vector", "halfvec"] = "halfvec",
    ):
        self.embedding_func = embedding_func
        self.db_client = db_client  # manages the database connection
        self.strategy_provider = strategy_provider
        self.engine = (
            db_client.create_db_engine()
        )  # SQLAlchemy async engine for db operations
        self.vector_type = vector_type
        self.models = {}  # cache of models for each table
        self.factories = {
            "Chunks": create_chunks_model,
            "Files": create_file_model,
            "Triplets": create_triplets_model,
        }  # mapping of table names to model creation functions

    def _to_list(self, embedding):
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    # retrieves or creates a SQLAlchemy model for the specified table
    def get_model(self, table_name: str):
        model = self.models.get(table_name)
        if model:
            return model
        dim = int(os.getenv("EMBEDDING_DIMENSION"))
        factory = self.factories.get(table_name)
        if not factory:
            raise ValueError(f"No factory found for table {table_name}")
        model = factory(use_halfvec=(self.vector_type == "halfvec"), dim=dim)
        self.models[table_name] = model
        return model

    # create a PGVector instance
    @classmethod
    def create(
        cls,
        embedding_func: EmbeddingFunc,
        db_url: str,
        strategy_provider: "RetrievalStrategyProvider",
        vector_type: Literal["vector", "halfvec"] = "halfvec",
    ):
        db_client = DatabaseClient()
        db_client.connection_string = db_url
        instance = cls(
            embedding_func, db_client, strategy_provider, vector_type=vector_type
        )
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
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
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
            ins = insert(table).values(rows)
            stmt = ins.on_conflict_do_nothing(
                index_elements=[table.c[name] for name in pk_cols]
            )

            await session.execute(stmt)
            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                f"[upsert_texts] Upserted {len(rows)} into '{table_name}', mode={mode}, elapsed={elapsed:.3f}s"
            )
            return rows

    async def upsert_file(
        self,
        properties_list: List[dict],
        table_name: str = "Files",
        mode: Literal["append", "overwrite"] = "append",
    ):
        model = self.get_model(table_name)

        start = time.perf_counter()
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            now = datetime.now()
            rows = []
            for props in properties_list:
                row = dict(props or {})
                row["updatedAt"] = now
                rows.append(row)

            table = model.__table__
            pk_cols = [c.name for c in table.primary_key.columns]
            ins = insert(table).values(rows)
            stmt = ins.on_conflict_do_nothing(
                index_elements=[table.c[name] for name in pk_cols]
            )

            await session.execute(stmt)
            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                f"[upsert_texts] Upserted {len(rows)} into '{table_name}', mode={mode}, elapsed={elapsed:.3f}s"
            )
            return rows

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

        model = self.get_model(table_name)

        start = time.perf_counter()
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
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

            if rerank:
                pass  # TODO: refactor the rerank logic to directly rerank the retrieved content

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
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
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
        async with AsyncSession(self.engine, expire_on_commit=False) as s:
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
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
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
        async with AsyncSession(self.engine, expire_on_commit=False) as _:
            pass
        async with self.engine.begin() as conn:
            Chunks = self.get_model("Chunks")
            Files = self.get_model("Files")
            Triplets = self.get_model("Triplets")

            def _create(sync_conn):
                PGBase.metadata.create_all(
                    bind=sync_conn,
                    tables=[Chunks.__table__, Files.__table__, Triplets.__table__],
                )

            await conn.run_sync(_create)

    async def clean_up(self):
        await self.engine.dispose()
