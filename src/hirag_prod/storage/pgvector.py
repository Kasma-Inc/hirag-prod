import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import load_only
from tqdm import tqdm

from hirag_prod._utils import AsyncEmbeddingFunction, log_error_info
from hirag_prod.configs.functions import get_init_config
from hirag_prod.cross_language_search.functions import normalize_tokenize_text
from hirag_prod.resources.functions import (
    get_db_engine,
    get_db_session_maker,
    get_translator,
)
from hirag_prod.schema import Base as PGBase
from hirag_prod.schema import Chunk, Entity, File, Graph, Item, Node, Relation, Triplets
from hirag_prod.storage.base_vdb import BaseVDB
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

logger = logging.getLogger(__name__)

THRESHOLD_DISTANCE = get_init_config().default_distance_threshold
TOPK = get_init_config().default_query_top_k
TOPN = get_init_config().default_query_top_n


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
            "Graph": Graph,
            "Nodes": Node,
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

    # batch upserts text embeddings into the specified table
    async def upsert_texts(
        self,
        texts_to_upsert: List[str],
        properties_list: List[dict],
        table_name: str,
        with_tokenization: bool = False,
        with_translation: bool = False,
        mode: Literal["append", "overwrite"] = "append",
    ):
        if len(texts_to_upsert) != len(properties_list):
            raise ValueError(
                "texts_to_upsert and properties_list must have the same length"
            )

        model = self.get_model(table_name)

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            embs = await self.embedding_func(texts_to_upsert)
            now = datetime.now()
            rows = []
            with tqdm(
                total=len(properties_list), desc="Processing texts", leave=False
            ) as progress_bar:
                for i in range(len(properties_list)):
                    row = dict(properties_list[i] or {})
                    if with_tokenization:
                        (
                            row["token_list"],
                            row["token_start_index_list"],
                            row["token_end_index_list"],
                        ) = normalize_tokenize_text(texts_to_upsert[i])
                    if with_translation:
                        row["translation"] = (
                            await get_translator().translate(
                                texts_to_upsert[i], dest="en"
                            )
                        ).text
                        if with_tokenization:
                            (
                                row["translation_token_list"],
                                row["translation_token_start_index_list"],
                                row["translation_token_end_index_list"],
                            ) = normalize_tokenize_text(row["translation"])
                    vec = self._to_list(embs[i])
                    row["vector"] = vec
                    row["updatedAt"] = now
                    rows.append(row)
                    progress_bar.update(1)

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
            # We conservatively cap at 30000 total parameters per statement.
            param_budget = 30000
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

    # use pg to mimic graphdb
    async def upsert_graph(
        self,
        relations: List[Relation],
        table_name: str = "Graph",
        mode: Literal["append", "overwrite"] = "append",
    ):

        edge_rows = []
        node_map: Dict[tuple, Dict[str, Any]] = (
            {}
        )  # key=(id, workspaceId, knowledgeBaseId), value=row

        def is_chunk(node_id: str) -> bool:
            return str(node_id).startswith("chunk-")

        for rel in relations:
            props = rel.properties or {}
            workspace_id = props.get("workspace_id")
            knowledge_base_id = props.get("knowledge_base_id")
            chunk_id = props.get("chunk_id", [])
            source_id = rel.source
            target_id = rel.target
            source_name = None if is_chunk(source_id) else props.get("source")
            target_name = None if is_chunk(target_id) else props.get("target")
            document_id = props.get("document_id", "")
            uri = props.get("uri", "")

            edge_rows.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "uri": uri,
                    "workspaceId": workspace_id,
                    "knowledgeBaseId": knowledge_base_id,
                    "documentId": document_id,
                }
            )

            for node_id, name_hint in (
                (source_id, source_name),
                (target_id, target_name),
            ):
                if is_chunk(node_id):
                    continue
                key = (node_id, workspace_id, knowledge_base_id)
                rec = node_map.get(key)
                if rec is None:
                    rec = {
                        "id": node_id,
                        "workspaceId": workspace_id,
                        "knowledgeBaseId": knowledge_base_id,
                        "entityName": name_hint,
                        "entityType": "entity",
                        "chunkIds": [],
                        "documentId": document_id,
                        "uri": uri,
                    }
                    node_map[key] = rec
                if chunk_id and chunk_id not in rec["chunkIds"]:
                    rec["chunkIds"].append(chunk_id)

        GraphModel = self.get_model("Graph")
        NodeModel = self.get_model("Nodes")

        start = time.perf_counter()
        async with get_db_session_maker()() as session:
            now = datetime.now()

            if edge_rows:
                graph_table = GraphModel.__table__
                pk_cols = [c.name for c in graph_table.primary_key.columns]
                cols_per_row = len(edge_rows[0])
                param_budget = 30000
                max_batch_size = max(1, param_budget // cols_per_row)
                for i in range(0, len(edge_rows), max_batch_size):
                    batch = [
                        {**r, "updatedAt": now}
                        for r in edge_rows[i : i + max_batch_size]
                    ]
                    ins = insert(graph_table).values(batch)
                    stmt = ins.on_conflict_do_nothing(
                        index_elements=[graph_table.c[name] for name in pk_cols]
                    )
                    await session.execute(stmt)

            node_rows = list(node_map.values())
            if node_rows:
                node_table = NodeModel.__table__
                for nr in node_rows:
                    if nr.get("chunkIds"):
                        nr["chunkIds"] = list(dict.fromkeys(nr["chunkIds"]))
                cols_per_row = len(node_rows[0])
                param_budget = 30000
                max_batch_size = max(1, param_budget // cols_per_row)
                for i in range(0, len(node_rows), max_batch_size):
                    batch = [
                        {**nr, "updatedAt": now}
                        for nr in node_rows[i : i + max_batch_size]
                    ]
                    ins = insert(node_table).values(batch)
                    stmt = ins.on_conflict_do_update(
                        index_elements=[
                            node_table.c.id,
                            node_table.c.workspaceId,
                            node_table.c.knowledgeBaseId,
                        ],
                        set_={
                            "entityName": func.coalesce(
                                ins.excluded.entityName, node_table.c.entityName
                            ),
                            "entityType": func.coalesce(
                                ins.excluded.entityType, node_table.c.entityType
                            ),
                            "chunkIds": func.coalesce(
                                func.array_cat(
                                    node_table.c.chunkIds, ins.excluded.chunkIds
                                ),
                                node_table.c.chunkIds,
                                ins.excluded.chunkIds,
                            ),
                            "documentId": func.coalesce(
                                ins.excluded.documentId, node_table.c.documentId
                            ),
                            "uri": func.coalesce(ins.excluded.uri, node_table.c.uri),
                            "updatedAt": now,
                        },
                    )
                    await session.execute(stmt)

            await session.commit()
            elapsed = time.perf_counter() - start
            logger.info(
                "[upsert_graph] Upserted %d edges and %d nodes, elapsed=%.3fs",
                len(edge_rows),
                len(node_rows),
                elapsed,
            )
            return {"edges": len(edge_rows), "nodes": len(node_rows)}

    async def query_node(
        self,
        node_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> Entity:
        NodeModel = self.get_model("Nodes")
        async with get_db_session_maker()() as session:
            stmt = (
                select(NodeModel)
                .where(NodeModel.id == node_id)
                .where(NodeModel.workspaceId == workspace_id)
                .where(NodeModel.knowledgeBaseId == knowledge_base_id)
            )
            row = (await session.execute(stmt)).scalars().first()

        if not row:
            raise ValueError(f"Node not found: {node_id}")

        name = getattr(row, "entityName", None) or ""
        etype = getattr(row, "entityType", None) or "UNKNOWN"
        chunk_ids = getattr(row, "chunkIds", None) or []
        if isinstance(chunk_ids, list):
            chunk_ids = list(dict.fromkeys(chunk_ids))

        meta = {
            "entity_type": etype,
            "description": [],
            "chunk_ids": chunk_ids,
            "document_id": getattr(row, "documentId", ""),
            "workspace_id": getattr(row, "workspaceId", ""),
            "knowledge_base_id": getattr(row, "knowledgeBaseId", ""),
        }
        return Entity(id=node_id, page_content=name, metadata=meta)

    async def pagerank_top_chunks_with_reset(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: int,
        alpha: float = 0.85,
    ) -> List[Tuple[str, float]]:
        if topk <= 0:
            return []

        start = time.perf_counter()
        GraphModel = self.get_model("Graph")

        async with get_db_session_maker()() as session:
            stmt = (
                select(GraphModel)
                .where(GraphModel.workspaceId == workspace_id)
                .where(GraphModel.knowledgeBaseId == knowledge_base_id)
            )
            rows = list((await session.execute(stmt)).scalars().all())

        if not rows:
            return []

        G = nx.DiGraph()
        for r in rows:
            if getattr(r, "source", None) and getattr(r, "target", None):
                G.add_edge(r.source, r.target)

        if G.number_of_nodes() == 0:
            return []

        personalization: Dict[str, float] = {}
        total = 0.0
        for node, w in (reset_weights or {}).items():
            if node in G:
                try:
                    val = float(w)
                except Exception:
                    continue
                if not math.isfinite(val) or val <= 0:
                    continue
                personalization[node] = val
                total += val

        if total <= 0:
            return []
        for node in personalization:
            personalization[node] /= total

        pr = nx.pagerank(
            G.to_undirected(), alpha=alpha, personalization=personalization
        )
        chunk_scores = [
            (node, score)
            for node, score in pr.items()
            if str(node).startswith("chunk-")
        ]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        out = chunk_scores[:topk]

        elapsed = time.perf_counter() - start
        logger.info(
            "[pagerank_top_chunks_with_reset] nodes=%d, returned=%d, elapsed=%.3fs",
            G.number_of_nodes(),
            len(out),
            elapsed,
        )
        return out

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
        topn: Optional[int] = TOPN,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
    ) -> List[dict]:
        if topk is None:
            topk = self.strategy_provider.default_topk
        if topn is None:
            topn = self.strategy_provider.default_topn

        if topn > topk:
            raise ValueError(f"topn ({topn}) must be <= topk ({topk})")

        model = self.get_model(table_name)

        if columns_to_select is None:  # query all except vector if nothing provided
            columns_to_select = [
                c for c in model.__table__.columns.keys() if c != "vector"
            ]

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

            elapsed = time.perf_counter() - start
            logger.info(
                f"[query] Retrieved {len(scored)} records from '{table_name}', elapsed={elapsed:.3f}s"
            )
            return scored

    # Function overload to handle list of queries
    async def query(
        self,
        query: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topn: Optional[int] = TOPN,
        topk: Optional[int] = TOPK,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = THRESHOLD_DISTANCE,
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
            # Generate embeddings for all query strings
            q_embs = await self.embedding_func(query)
            q_embs = [self._to_list(emb) for emb in q_embs]

            # Calculate cosine distance for each query embedding and take the minimum
            distance_expressions = [
                model.vector.cosine_distance(q_emb) for q_emb in q_embs
            ]

            # Use func.least to get the minimum distance among all query embeddings
            if len(distance_expressions) == 1:
                distance_expr = distance_expressions[0].label("distance")
            else:
                distance_expr = func.least(*distance_expressions).label("distance")

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
        additional_data_to_select: Optional[Dict[str, Any]] = None,
        additional_where_clause_list: Optional[Any] = None,
        order_by: Optional[List[Any]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        model = self.get_model(table_name)
        entity_to_select_list: List[Any] = [model]
        additional_data_key_list: List[str] = []
        if additional_data_to_select is not None:
            for k, v in additional_data_to_select.items():
                additional_data_key_list.append(k)
                entity_to_select_list.append(v)
        if columns_to_select is None:  # query all if nothing provided
            columns_to_select = [
                c for c in model.__table__.columns.keys() if c != "vector"
            ]
        async with get_db_session_maker()() as session:
            stmt = select(*entity_to_select_list).options(
                load_only(*[getattr(model, column) for column in columns_to_select])
            )
            if key_value:
                stmt = stmt.where(getattr(model, key_column).in_(key_value))
            if workspace_id and hasattr(model, "workspaceId"):
                stmt = stmt.where(model.workspaceId == workspace_id)
            if knowledge_base_id and hasattr(model, "knowledgeBaseId"):
                stmt = stmt.where(model.knowledgeBaseId == knowledge_base_id)
            if additional_where_clause_list is not None:
                stmt = stmt.where(additional_where_clause_list)
            if order_by is not None:
                stmt = stmt.order_by(*order_by)
            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = list(result.all())

            out: List[dict] = []
            for r in rows:
                rec = {}
                for col in columns_to_select:
                    if not hasattr(r[0], col):
                        continue
                    rec[col] = getattr(r[0], col)
                for i in range(1, len(r)):
                    rec[additional_data_key_list[i - 1]] = r[i]
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
                for col in r.__table__.columns.keys():
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
            Graph = self.get_model("Graph")
            Nodes = self.get_model("Nodes")

            def _create(sync_conn):
                PGBase.metadata.create_all(
                    bind=sync_conn,
                    tables=[
                        Chunks.__table__,
                        Files.__table__,
                        Triplets.__table__,
                        Items.__table__,
                        Graph.__table__,
                        Nodes.__table__,
                    ],
                )

            await conn.run_sync(_create)
