import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from hirag_prod._utils import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _limited_gather_with_factory,
    compute_mdhash_id,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from hirag_prod.prompt import PROMPTS
from hirag_prod.schema import Chunk, Entity, Relation

from .base import BaseEntity


@dataclass
class VanillaEntity(BaseEntity):
    """
    Production-ready entity and relation extraction pipeline using LLM models.
    """

    # === Core Components ===
    llm_model_name: str = field(default="gpt-4o-mini")
    extract_func: Callable
    continue_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_continue_extraction"]
    )

    # === Entity Extraction Configuration ===
    entity_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["hi_entity_extraction"]
    )
    entity_extract_max_gleaning: int = field(default=1)
    entity_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_if_loop_extraction"]
    )
    entity_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        }
    )

    # === Relation Extraction Configuration ===
    relation_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["hi_relation_extraction"]
    )
    relation_extract_max_gleaning: int = field(default=1)
    relation_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["relation_if_loop_extraction"]
    )
    relation_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        }
    )

    # === Concurrency Configuration ===
    entity_extraction_concurrency: int = field(default=64)
    entity_merge_concurrency: int = field(default=64)
    relation_extraction_concurrency: int = field(default=64)

    @classmethod
    def create(cls, **kwargs) -> "VanillaEntity":
        """Factory method to create a VanillaEntity instance with custom configuration."""
        return cls(**kwargs)

    async def entity(self, chunks: List[Chunk]) -> List[Entity]:
        """
        Extract and merge entities from text chunks.

        This method orchestrates the complete entity extraction pipeline:
        1. Concurrent extraction from all chunks
        2. Flattening and validation of results
        3. Merging of duplicate entities across chunks

        Args:
            chunks: List of text chunks to process for entity extraction

        Returns:
            List of unique, merged entities extracted from all chunks
        """
        logging.info(f"[Entity] Starting extraction from {len(chunks)} chunks")

        # TODO: Add progress bar for entity extraction to track processing of chunks

        # Step 1: Extract entities from all chunks concurrently
        extraction_factories = [
            lambda chunk=chunk: self._extract_entities_from_chunk(chunk)
            for chunk in chunks
        ]
        entities_lists = await _limited_gather_with_factory(
            extraction_factories, self.entity_extraction_concurrency
        )

        # Step 2: Flatten and filter valid entities
        entities = self._flatten_entity_lists(entities_lists)
        logging.info(f"[Entity] Extracted {len(entities)} raw entities")

        # Step 3: Merge duplicate entities
        merged_entities = await self._merge_duplicate_entities(entities)
        logging.info(f"[Entity] Completed with {len(merged_entities)} final entities")

        return merged_entities

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

            # Step 1: Initial entity extraction
            entity_result, history = await self._perform_initial_entity_extraction(
                chunk
            )

            # Step 2: Iterative gleaning for improved quality
            entity_result = await self._perform_entity_gleaning(entity_result, history)

            # Step 3: Parse entities from LLM output
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

    async def _perform_initial_entity_extraction(
        self, chunk: Chunk
    ) -> Tuple[str, List]:
        """
        Perform the initial entity extraction using LLM.

        Args:
            chunk: Text chunk to process

        Returns:
            Tuple of (extraction_result, conversation_history)
        """
        entity_prompt = self.entity_extract_prompt.format(
            **self.entity_extract_context, input_text=chunk.page_content
        )

        entity_result = await self.extract_func(
            model=self.llm_model_name,
            prompt=entity_prompt,
        )

        history = pack_user_ass_to_openai_messages(entity_prompt, entity_result)
        return entity_result, history

    async def _perform_entity_gleaning(self, entity_result: str, history: List) -> str:
        """
        Perform iterative gleaning to improve entity extraction quality.

        Gleaning is an iterative process where the LLM is asked to continue
        extraction, potentially finding entities missed in the initial pass.

        Args:
            entity_result: Initial extraction result
            history: Conversation history for context

        Returns:
            Enhanced extraction result after gleaning
        """
        for glean_idx in range(self.entity_extract_max_gleaning):
            # Get additional entities through gleaning
            glean_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=self.continue_prompt,
                history_messages=history,
            )

            history += pack_user_ass_to_openai_messages(
                self.continue_prompt, glean_result
            )
            entity_result += glean_result

            # Check if we should continue gleaning (skip termination check on last iteration)
            if glean_idx < self.entity_extract_max_gleaning - 1:
                should_continue = await self._should_continue_extraction(
                    history, self.entity_extract_termination_prompt
                )
                if not should_continue:
                    break

        return entity_result

    async def _should_continue_extraction(
        self, history: List, termination_prompt: str
    ) -> bool:
        """
        Determine if extraction should continue based on LLM judgment.

        Args:
            history: Conversation history for context
            termination_prompt: Prompt to ask LLM if extraction should continue

        Returns:
            True if extraction should continue, False otherwise
        """
        termination_response = await self.extract_func(
            model=self.llm_model_name,
            prompt=termination_prompt,
            history_messages=history,
        )

        normalized_response = termination_response.strip().strip('"').strip("'").lower()
        return normalized_response == "yes"

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
        # Split the result into individual records
        records = split_string_by_multi_markers(
            entity_result,
            [
                self.entity_extract_context["record_delimiter"],
                self.entity_extract_context["completion_delimiter"],
            ],
        )

        entities = []
        for record in records:
            entity = await self._parse_single_entity_record(record, chunk_id)
            if entity:
                entities.append(entity)

        return entities

    async def _parse_single_entity_record(
        self, record: str, chunk_id: str
    ) -> Optional[Entity]:
        """
        Parse a single entity record from the LLM output.

        Args:
            record: Single entity record string from LLM output
            chunk_id: ID of the source chunk

        Returns:
            Parsed Entity object or None if parsing fails
        """
        # Extract content within parentheses using regex
        match = re.search(r"\((.*?)\)", record)
        if not match:
            return None

        record_content = match.group(1)
        record_attributes = split_string_by_multi_markers(
            record_content, [self.entity_extract_context["tuple_delimiter"]]
        )

        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_id
        )

        if not entity_data:
            return None

        return Entity(
            id=compute_mdhash_id(entity_data["entity_name"], prefix="ent-"),
            page_content=entity_data["entity_name"],
            metadata={
                "entity_type": entity_data["entity_type"],
                "description": [entity_data["description"]],
                "chunk_ids": [chunk_id],
            },
        )

    def _flatten_entity_lists(self, entities_lists: List[List[Entity]]) -> List[Entity]:
        """
        Flatten and filter entity lists from multiple chunks.

        Args:
            entities_lists: List of entity lists from different chunks

        Returns:
            Flattened list of all valid entities
        """
        return [
            entity
            for entity_list in entities_lists
            if entity_list  # Filter out None/empty lists
            for entity in entity_list
        ]

    async def _merge_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge entities that appear in multiple chunks.

        When the same entity is mentioned across multiple chunks, this method
        consolidates them into a single entity with merged metadata.

        Args:
            entities: List of all extracted entities

        Returns:
            List of unique entities with duplicates merged
        """
        # Count entity occurrences by name
        entity_counts = Counter(entity.page_content for entity in entities)

        # Separate unique and duplicate entities
        unique_entities = [
            entity for entity in entities if entity_counts[entity.page_content] == 1
        ]

        duplicate_entities = [
            entity for entity in entities if entity_counts[entity.page_content] > 1
        ]

        # Group duplicates by name for merging
        entities_by_name = defaultdict(list)
        for entity in duplicate_entities:
            entities_by_name[entity.page_content].append(entity)

        # Merge duplicate entities concurrently
        merge_factories = [
            lambda name=name, entity_list=entity_list: self._merge_entities_by_name(
                name, entity_list
            )
            for name, entity_list in entities_by_name.items()
        ]

        merged_entities = await _limited_gather_with_factory(
            merge_factories, self.entity_merge_concurrency
        )
        merged_entities = [entity for entity in merged_entities if entity]

        return unique_entities + merged_entities

    async def _merge_entities_by_name(
        self, entity_name: str, entities: List[Entity]
    ) -> Optional[Entity]:
        """
        Merge multiple entities with the same name into a single entity.

        Args:
            entity_name: Name of the entity to merge
            entities: List of entity instances to merge

        Returns:
            Merged Entity object or None if merge fails
        """
        try:
            logging.info(
                f"[Entity-Merge] Merging {len(entities)} instances of '{entity_name}'"
            )

            # Collect and align descriptions with chunk IDs
            merged_descriptions, merged_chunk_ids = self._collect_entity_descriptions(
                entities, entity_name
            )

            # Determine the most common entity type
            entity_type = self._get_most_common_entity_type(entities)

            merged_entity = Entity(
                id=compute_mdhash_id(entity_name, prefix="ent-"),
                page_content=entity_name,
                metadata={
                    "entity_type": entity_type,
                    "description": merged_descriptions,
                    "chunk_ids": merged_chunk_ids,
                },
            )

            logging.info(
                f"[Entity-Merge] Successfully merged '{entity_name}': "
                f"type={entity_type}, chunks={len(merged_chunk_ids)}, "
                f"descriptions={len(merged_descriptions)}"
            )

            return merged_entity

        except Exception as e:
            logging.exception(f"[Entity-Merge] Failed to merge '{entity_name}': {e}")
            warnings.warn(f"Entity merge failed for {entity_name}: {e}")
            return None

    def _collect_entity_descriptions(
        self, entities: List[Entity], entity_name: str
    ) -> Tuple[List[str], List[str]]:
        """
        Collect and align descriptions with chunk IDs from multiple entity instances.

        Args:
            entities: List of entity instances to collect from
            entity_name: Name of the entity for logging

        Returns:
            Tuple of (merged_descriptions, merged_chunk_ids)
        """
        merged_descriptions = []
        merged_chunk_ids = []

        for entity in entities:
            descriptions = entity.metadata.description
            chunk_ids = entity.metadata.chunk_ids

            # Ensure descriptions and chunk_ids have matching lengths
            if len(descriptions) != len(chunk_ids):
                logging.warning(
                    f"[Entity-Merge] Mismatched description/chunk_ids lengths "
                    f"for entity '{entity_name}'"
                )
                descriptions, chunk_ids = self._align_descriptions_and_chunks(
                    descriptions, chunk_ids
                )

            merged_descriptions.extend(descriptions)
            merged_chunk_ids.extend(chunk_ids)

        return merged_descriptions, merged_chunk_ids

    def _align_descriptions_and_chunks(
        self, descriptions: List[str], chunk_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Align descriptions and chunk IDs to have matching lengths.

        This method handles cases where descriptions and chunk_ids lists
        have mismatched lengths by extending the shorter list.

        Args:
            descriptions: List of entity descriptions
            chunk_ids: List of chunk IDs

        Returns:
            Tuple of aligned (descriptions, chunk_ids)
        """
        # Extend the shorter list to match the longer one
        while len(descriptions) < len(chunk_ids):
            descriptions.append(descriptions[-1] if descriptions else "")
        while len(chunk_ids) < len(descriptions):
            chunk_ids.append(chunk_ids[-1] if chunk_ids else "")

        return descriptions, chunk_ids

    def _get_most_common_entity_type(self, entities: List[Entity]) -> str:
        """
        Get the most common entity type from a list of entities.

        Args:
            entities: List of entities to analyze

        Returns:
            Most frequently occurring entity type
        """
        entity_types = [entity.metadata.entity_type for entity in entities]
        type_counts = Counter(entity_types)
        return type_counts.most_common(1)[0][0]

    async def relation(
        self, chunks: List[Chunk], entities: List[Entity]
    ) -> List[Relation]:
        """
        Extract relations between entities from text chunks.

        This method processes each chunk to identify relationships between the provided
        entities, using the LLM to understand contextual connections and semantic
        relationships in the text.

        Args:
            chunks: List of text chunks to process for relation extraction
            entities: List of available entities for relation identification

        Returns:
            List of relations extracted from all chunks
        """
        logging.info(
            f"[Relation] Starting extraction from {len(chunks)} chunks with {len(entities)} entities"
        )

        # TODO: Add progress bar for relation extraction to show chunk processing progress

        # Create relation extraction tasks for each chunk
        extraction_factories = [
            lambda chunk=chunk, entities=entities: self._extract_relations_from_chunk(
                chunk, entities
            )
            for chunk in chunks
        ]

        relations_lists = await _limited_gather_with_factory(
            extraction_factories, self.relation_extraction_concurrency
        )

        # Flatten and filter valid relations
        relations = self._flatten_relation_lists(relations_lists)
        logging.info(f"[Relation] Completed with {len(relations)} relations")

        return relations

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

            # Create entity lookup for this chunk
            chunk_entities = {
                entity.page_content: entity
                for entity in entities
                if chunk.id in entity.metadata.chunk_ids
            }

            if not chunk_entities:
                logging.info(
                    f"[Relation] No entities found for chunk {chunk.id}, skipping"
                )
                return []

            # Step 1: Initial relation extraction
            relation_result, history = await self._perform_initial_relation_extraction(
                chunk, chunk_entities
            )

            # Step 2: Iterative gleaning
            relation_result = await self._perform_relation_gleaning(
                relation_result, history
            )

            # Step 3: Parse relations from LLM output
            relations = await self._parse_relations_from_result(
                relation_result, chunk, chunk_entities
            )

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

    async def _perform_initial_relation_extraction(
        self, chunk: Chunk, entities_dict: Dict[str, Entity]
    ) -> Tuple[str, List]:
        """
        Perform initial relation extraction using LLM.

        Args:
            chunk: Text chunk to process
            entities_dict: Available entities for relation identification

        Returns:
            Tuple of (extraction_result, conversation_history)
        """
        relation_prompt = self.relation_extract_prompt.format(
            **self.relation_extract_context,
            entities=list(entities_dict.keys()),
            input_text=chunk.page_content,
        )

        relation_result = await self.extract_func(
            model=self.llm_model_name,
            prompt=relation_prompt,
        )

        history = pack_user_ass_to_openai_messages(relation_prompt, relation_result)
        return relation_result, history

    async def _perform_relation_gleaning(
        self, relation_result: str, history: List
    ) -> str:
        """
        Perform iterative gleaning for relation extraction.

        Args:
            relation_result: Initial extraction result
            history: Conversation history for context

        Returns:
            Enhanced extraction result after gleaning
        """
        for glean_idx in range(self.relation_extract_max_gleaning):
            glean_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=self.continue_prompt,
                history_messages=history,
            )

            history += pack_user_ass_to_openai_messages(
                self.continue_prompt, glean_result
            )
            relation_result += glean_result

            # Check if we should continue gleaning (skip termination check on last iteration)
            if glean_idx < self.relation_extract_max_gleaning - 1:
                should_continue = await self._should_continue_extraction(
                    history, self.relation_extract_termination_prompt
                )
                if not should_continue:
                    break

        return relation_result

    async def _parse_relations_from_result(
        self, relation_result: str, chunk: Chunk, entities_dict: Dict[str, Entity]
    ) -> List[Relation]:
        """
        Parse relations from LLM output string.

        Args:
            relation_result: Raw LLM output containing relation information
            chunk: Source chunk for the relations
            entities_dict: Available entities for relation validation

        Returns:
            List of parsed Relation objects
        """
        records = split_string_by_multi_markers(
            relation_result,
            [
                self.relation_extract_context["record_delimiter"],
                self.relation_extract_context["completion_delimiter"],
            ],
        )

        relations = []
        for record in records:
            relation = await self._parse_single_relation_record(
                record, chunk, entities_dict
            )
            if relation:
                relations.append(relation)

        return relations

    async def _parse_single_relation_record(
        self, record: str, chunk: Chunk, entities_dict: Dict[str, Entity]
    ) -> Optional[Relation]:
        """
        Parse a single relation record from LLM output.

        Args:
            record: Single relation record string from LLM output
            chunk: Source chunk for the relation
            entities_dict: Available entities for validation

        Returns:
            Parsed Relation object or None if parsing fails
        """
        # Extract content within parentheses
        match = re.search(r"\((.*)\)", record)
        if not match:
            return None

        record_content = match.group(1)
        record_attributes = split_string_by_multi_markers(
            record_content, [self.relation_extract_context["tuple_delimiter"]]
        )

        relation_data = await _handle_single_relationship_extraction(
            record_attributes, chunk.id
        )

        if not relation_data:
            return None

        # Validate that both source and target entities exist
        source_entity = entities_dict.get(relation_data["src_id"])
        target_entity = entities_dict.get(relation_data["tgt_id"])

        if not source_entity:
            logging.warning(
                f"Source entity '{relation_data['src_id']}' not found, "
                f"skipping relation in chunk {chunk.id}"
            )
            return None

        if not target_entity:
            logging.warning(
                f"Target entity '{relation_data['tgt_id']}' not found, "
                f"skipping relation in chunk {chunk.id}"
            )
            return None

        return Relation(
            source=source_entity.id,
            target=target_entity.id,
            properties={
                "description": relation_data["description"],
                "weight": relation_data["weight"],
                "chunk_id": chunk.id,
            },
        )

    def _flatten_relation_lists(
        self, relations_lists: List[List[Relation]]
    ) -> List[Relation]:
        """
        Flatten and filter relation lists from multiple chunks.

        Args:
            relations_lists: List of relation lists from different chunks

        Returns:
            Flattened list of all valid relations
        """
        return [
            relation
            for relation_list in relations_lists
            if relation_list  # Filter out None/empty lists
            for relation in relation_list
        ]
