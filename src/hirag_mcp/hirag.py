import logging
from dataclasses import dataclass, field
from typing import Optional


from ._llm import gpt_4o_mini_complete, openai_embedding
from .chunk import BaseChunk, FixTokenChunk
from .entity import BaseEntity, VanillaEntity
from .loader import load_document
from .storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

logger = logging.getLogger("HiRAG")


@dataclass
class HiRAG:
    # Chunk documents
    chunker: BaseChunk = field(
        default_factory=lambda: FixTokenChunk(chunk_size=1200, chunk_overlap=200)
    )

    # Entity extraction
    entity_extractor: BaseEntity = field(
        default_factory=lambda: VanillaEntity.create(extract_func=gpt_4o_mini_complete)
    )

    # Storage
    vdb: BaseVDB = field(default=None)
    gdb: BaseGDB = field(
        default_factory=lambda: NetworkXGDB.create(
            path="kb/hirag.gpickle",
            llm_func=gpt_4o_mini_complete,
        )
    )

    @classmethod
    async def create(cls, **kwargs):
        if kwargs.get("vdb") is None:
            lancedb = await LanceDB.create(
                embedding_func=openai_embedding,
                db_url="kb/hirag.db",
                strategy_provider=RetrievalStrategyProvider(),
            )
            kwargs["vdb"] = lancedb
        return cls(**kwargs)

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ):
        # Load the document from the document path
        logger.info(f"Loading the document from the document path: {document_path}")
        documents = load_document(
            document_path, content_type, document_meta, loader_configs
        )
        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Chunking the documents")
        # TODO: Handle the concurrent upsertion
        for document in documents:
            chunks = self.chunker.chunk(document)
            # TODO: Handle the concurrent upsertion
            for chunk in chunks:
                await self.vdb.upsert_text(
                    text_to_embed=chunk.page_content,
                    properties={
                        "document_key": chunk.id,
                        "text": chunk.page_content,
                        **chunk.metadata.__dict__,
                    },
                    table_name="chunks",
                    mode="overwrite",
                )
            entities = await self.entity_extractor.entity(chunks)
            # Store to lancedb for now
            for entity in entities:
                await self.vdb.upsert_text(
                    text_to_embed=entity.metadata.description,
                    properties={
                        "document_key": entity.id,
                        "text": entity.page_content,
                        **entity.metadata.__dict__,
                    },
                    table_name="entities",
                    mode="overwrite",
                )

            # extract relations
            relations = await self.entity_extractor.relation(chunks, entities)
            # TODO: handle the concurrent upsertion
            for relation in relations:
                await self.gdb.upsert_relation(relation)
