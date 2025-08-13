import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import json_repair

from .._utils import (
    _limited_gather_with_factory,
    compute_mdhash_id,
)
from ..prompt import PROMPTS
from ..schema import Chunk, Entity, Relation
from .base import BaseKG


@dataclass
class VanillaKG(BaseKG):
    """
    Production-ready knowledge graph construction pipeline using LLM models.
    """

    # === Core Components ===
    llm_model_name: str = field(default="gpt-4o")
    extract_func: Callable
    language: str = field(default="en")  # en | cn

    # === Entity Extraction Configuration ===
    entity_extract_prompt: str = field(init=False)

    # === Relation Extraction Configuration ===
    relation_extract_prompt: str = field(init=False)

    # === Concurrency Configuration ===
    chunk_processing_concurrency: int = field(default=16)

    def __post_init__(self):
        self._update_language_config()

    def _update_language_config(self):
        """Update all language-specific configurations based on current language setting."""

        self.entity_extract_prompt = PROMPTS[f"entity_extraction_{self.language}"]
        self.relation_extract_prompt = PROMPTS[f"triplet_extraction_{self.language}"]

    def set_language(self, language: str):
        """
        Update the language setting and refresh all language-specific configurations.

        Args:
            language: Language code ('en' or 'cn')
        """
        if language not in ["en", "cn"]:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: ['en', 'cn']"
            )

        self.language = language
        self._update_language_config()
        logging.info(f"[VanillaKG] Language updated to {language}")

    @classmethod
    def create(cls, language: str = "en", **kwargs) -> "VanillaKG":
        """Factory method to create a VanillaKG instance with custom configuration."""
        return cls(language=language, **kwargs)

    async def _dense_sparse_integration(
        self, entities: List[Entity], chunk: Chunk
    ) -> List[Relation]:
        """
        Dense Sparse Integration
        - Dense: chunk node
        - Sparse: entity node
        
        Each chunk in the corpus is treated as a dense node, with the context edge labeled "contains" connecting \
        the chunk to all entities derived from the chunk.

        Args:
            entities: List of extracted entities with chunk metadata
        """
        if not entities:
            return

        dense_sparse_relations = []
        for entity in entities:
            dense_sparse_relations.append(
                Relation(
                    source=chunk.id,
                    target=entity.id,
                    properties={
                        "source": chunk.id,
                        "relation": "contains",
                        "target": entity.page_content,
                        "description": f"Chunk {chunk.id} contains Entity {entity.page_content}",
                        "weight": 1.0,
                        "chunk_id": chunk.id,
                        "file_name": chunk.metadata.filename,
                    },
                )
            )

        return dense_sparse_relations

    async def _extract_entities_from_chunk(self, chunk: Chunk) -> List[Entity]:
        """
        Extract entities from a single text chunk using LLM with iterative gleaning.

        This method implements the complete extraction pipeline for a single chunk:
        1. Initial LLM-based extraction using configured prompts
        2. Iterative gleaning to improve extraction quality
        3. Parsing and validation of extracted entities

        Args:
            chunk: Text chunk to process

        Returns:
            List of entities extracted from the chunk
        """
        try:
            start_time = time.time()

            entity_prompt = self.entity_extract_prompt.format(
                input_text=chunk.page_content
            )

            entity_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=entity_prompt,
            )

            entities = await self._parse_entities_from_result(entity_result, chunk.id)

            elapsed = time.time() - start_time
            logging.info(
                f"[Entity] Extracted {len(entities)} entities from chunk {chunk.id} "
                f"in {elapsed:.2f}s"
            )

            return entities

        except Exception as e:
            logging.exception(f"[Entity] Extraction failed for chunk {chunk.id}")
            warnings.warn(f"Entity extraction failed for chunk {chunk.id}: {e}")
            return []

    async def _parse_entities_from_result(
        self, entity_result: str, chunk_id: str
    ) -> List[Entity]:
        """
        Parse entities from LLM output string.

        This method processes the raw LLM output, splitting it into individual
        entity records and parsing each into a structured Entity object.

        Args:
            entity_result: Raw LLM output containing entity information
            chunk_id: ID of the source chunk for tracking

        Returns:
            List of parsed Entity objects
        """
        decoded_obj = json_repair.repair_json(entity_result, return_objects=True)
        entity_list = decoded_obj.get("entities", [])
        entities = []
        for entity in entity_list:
            entity_id = compute_mdhash_id(entity, prefix="ent-")
            entities.append(
                Entity(
                    id=entity_id,
                    page_content=entity,
                    metadata={
                        "entity_type": "entity",
                        "description": [],
                        "chunk_ids": [chunk_id],
                    },
                )
            )
        return entities

    async def _extract_relations_from_chunk(
        self, chunk: Chunk, entities: List[Entity]
    ) -> List[Relation]:
        """
        Extract relations from a single chunk.

        Args:
            chunk: Text chunk to process
            entities: Available entities for relation extraction

        Returns:
            List of relations extracted from the chunk
        """
        try:
            start_time = time.time()

            if not entities:
                logging.info(
                    f"[Relation] No entities found for chunk {chunk.id}, skipping"
                )
                return []

            # Create entity list for prompt
            entity_names = [entity.page_content for entity in entities]

            relation_prompt = self.relation_extract_prompt.format(
                entity_list=entity_names,
                input_text=chunk.page_content,
            )

            # Step 1: Initial relation extraction
            relation_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=relation_prompt,
            )

            relations = await self._parse_relations_from_result(relation_result, chunk)

            elapsed = time.time() - start_time
            logging.info(
                f"[Relation] Extracted {len(relations)} relations from chunk {chunk.id} "
                f"in {elapsed:.2f}s"
            )

            return relations

        except Exception as e:
            logging.exception(f"[Relation] Extraction failed for chunk {chunk.id}")
            warnings.warn(f"Relation extraction failed for chunk {chunk.id}: {e}")
            return []

    async def _parse_relations_from_result(
        self, relation_result: str, chunk: Chunk
    ) -> List[Relation]:
        """
        Parse relations from LLM output string.

        Args:
            relation_result: Raw LLM output containing relation information
            chunk: Source chunk for the relations

        Returns:
            List of parsed Relation objects
        """
        try:
            decoded_obj = json_repair.repair_json(relation_result, return_objects=True)
            triplets = decoded_obj.get("triplets", [])
        except json_repair.JSONRepairError:
            return []

        relations = []
        for triplet in triplets:
            head = triplet.get("Head")
            relation = triplet.get("Relation")
            tail = triplet.get("Tail")
            if not all([head, relation, tail]):
                continue

            source_id = compute_mdhash_id(head, prefix="ent-")
            target_id = compute_mdhash_id(tail, prefix="ent-")

            properties = {
                "source": head,
                "relation": relation,
                "target": tail,
                "description": f"{head} {relation} {tail}",
                "weight": 1.0,
                "chunk_id": chunk.id,
                "file_name": chunk.metadata.filename,
            }

            rel = Relation(
                source=source_id,
                target=target_id,
                properties=properties,
            )
            relations.append(rel)

            source_entity = Entity(
                id=source_id,
                page_content=head,
                metadata={
                    "entity_type": "entity",
                    "description": [],
                    "chunk_ids": [chunk.id],
                },
            )
            target_entity = Entity(
                id=target_id,
                page_content=tail,
                metadata={
                    "entity_type": "entity",
                    "description": [],
                    "chunk_ids": [chunk.id],
                },
            )

            dense_sparse_relations = await self._dense_sparse_integration(
                entities=[source_entity, target_entity],
                chunk=chunk,
            )
            relations.extend(dense_sparse_relations)

        return relations

    async def construct_kg(
        self, chunks: List[Chunk]
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Process chunks to extract entities and relations concurrently.

        This is the main entry point that processes all chunks in parallel,
        extracting both entities and relations for each chunk simultaneously.

        Args:
            chunks: List of text chunks to process

        Returns:
            Tuple of (all_entities, all_relations) extracted from all chunks
        """
        logging.info(f"[ProcessChunks] Starting processing of {len(chunks)} chunks")

        if not chunks:
            return [], []

        # Create chunk processing factories for concurrent execution
        chunk_factories = [
            lambda chunk=chunk: self._process_single_chunk(chunk) for chunk in chunks
        ]

        # Process all chunks concurrently with progress bar
        chunk_results = await _limited_gather_with_factory(
            chunk_factories,
            self.chunk_processing_concurrency,
            desc=f"Processing {len(chunks)} chunks",
            show_progress=True,
        )

        # Aggregate results from all chunks
        all_entities = []
        all_relations = []

        for entities, relations in chunk_results:
            if entities:
                all_entities.extend(entities)
            if relations:
                all_relations.extend(relations)

        logging.info(
            f"[ProcessChunks] Completed processing: "
            f"{len(all_entities)} entities, {len(all_relations)} relations"
        )

        return all_entities, all_relations

    async def _process_single_chunk(
        self, chunk: Chunk
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Process a single chunk to extract both entities and relations.

        Args:
            chunk: Text chunk to process

        Returns:
            Tuple of (entities, relations) extracted from this chunk
        """
        try:
            start_time = time.time()

            # Extract entities first
            entities = await self._extract_entities_from_chunk(chunk)

            # Extract relations using the entities from this chunk
            relations = await self._extract_relations_from_chunk(chunk, entities)

            elapsed = time.time() - start_time
            logging.info(
                f"[SingleChunk] Processed chunk {chunk.id}: "
                f"{len(entities)} entities, {len(relations)} relations in {elapsed:.2f}s"
            )

            return entities, relations

        except Exception as e:
            logging.exception(f"[SingleChunk] Failed to process chunk {chunk.id}")
            warnings.warn(f"Chunk processing failed for {chunk.id}: {e}")
            return [], []
