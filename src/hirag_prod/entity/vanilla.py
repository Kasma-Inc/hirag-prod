import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from hirag_prod._utils import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _limited_gather,
    compute_mdhash_id,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from hirag_prod.prompt import PROMPTS
from hirag_prod.schema import Chunk, Entity, Relation
from hirag_prod.summarization import BaseSummarizer, TrancatedAggregateSummarizer

from .base import BaseEntity


@dataclass
class VanillaEntity(BaseEntity):
    # === Common Components ===
    # Core function for LLM-based extraction
    llm_model_name: str = field(default="gpt-4o-mini")
    extract_func: Callable
    # Summarizer for entity descriptions
    entity_description_summarizer: BaseSummarizer = field(default=None)
    # Prompt for continuing entity extraction
    continue_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_continue_extraction"]
    )

    # === Entity Extraction Parameters ===
    # Main prompt for entity extraction
    entity_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_extraction"]
    )
    # Max iterations for entity extraction
    entity_extract_max_gleaning: int = field(default_factory=lambda: 1)
    # Prompt to determine when to stop entity extraction
    entity_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["entity_if_loop_extraction"]
    )
    # Context variables for entity extraction
    entity_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        }
    )

    # === Relation Extraction Parameters ===
    # Main prompt for relation extraction
    relation_extract_prompt: str = field(
        default_factory=lambda: PROMPTS["hi_relation_extraction"]
    )
    # Max iterations for relation extraction
    relation_extract_max_gleaning: int = field(default_factory=lambda: 1)
    # Prompt to determine when to stop relation extraction
    relation_extract_termination_prompt: str = field(
        default_factory=lambda: PROMPTS["relation_if_loop_extraction"]
    )
    # Context variables for relation extraction
    relation_extract_context: dict = field(
        default_factory=lambda: {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        }
    )

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def __post_init__(self):
        if self.entity_description_summarizer is None:
            self.entity_description_summarizer = TrancatedAggregateSummarizer(
                llm_model_name=self.llm_model_name,
                extract_func=self.extract_func,
            )

    async def entity(self, chunks: List[Chunk]) -> List[Entity]:
        async def _process_single_content_entity(
            chunk: Chunk,
        ) -> List[Entity]:
            """
            This function is used to extract entities from a single chunk.
            """
            chunk_key = chunk.id
            content = chunk.page_content
            # 1. initial extraction
            entity_extraction_prompt = self.entity_extract_prompt.format(
                **self.entity_extract_context, input_text=content
            )  # fill in the parameter
            entity_string_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=entity_extraction_prompt,
            )  # feed into LLM with the prompt

            content_history = pack_user_ass_to_openai_messages(
                entity_extraction_prompt, entity_string_result
            )  # concat the prompt and result as history for the next iteration

            # 2. continue to extract entities for higher quality entities entraction, normally we only need 1 iteration
            for glean_idx in range(self.entity_extract_max_gleaning):
                glean_result = await self.extract_func(
                    model=self.llm_model_name,
                    prompt=self.continue_prompt,
                    history_messages=content_history,
                )

                content_history += pack_user_ass_to_openai_messages(
                    self.continue_prompt, glean_result
                )  # add to history
                entity_string_result += glean_result
                if glean_idx == self.entity_extract_max_gleaning - 1:
                    break

                entity_extraction_termination_str: str = (
                    await self.extract_func(  # judge if we still need the next iteration
                        model=self.llm_model_name,
                        prompt=self.entity_extract_termination_prompt,
                        history_messages=content_history,
                    )
                )
                entity_extraction_termination_str = (
                    entity_extraction_termination_str.strip()
                    .strip('"')
                    .strip("'")
                    .lower()
                )
                if entity_extraction_termination_str != "yes":
                    break

            # 3. split entities from entity_string_result, which is the output of llm --> list of entities
            records: List[str] = split_string_by_multi_markers(
                entity_string_result,
                [
                    self.entity_extract_context["record_delimiter"],
                    self.entity_extract_context["completion_delimiter"],
                ],
            )

            # 4. Use regrex to extract the entity,
            # entities is a list of entity, where entity is a Entity object
            entities = []
            for record in records:
                record = re.search(r"\((.*?)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(  # split entity
                    record, [self.entity_extract_context["tuple_delimiter"]]
                )
                entity = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                    record_attributes, chunk_key
                )
                if entity is not None:
                    # entities_dict[entity["entity_name"]].append(entity)

                    entities.append(
                        Entity(
                            id=compute_mdhash_id(entity["entity_name"], prefix="ent-"),
                            page_content=entity["entity_name"],
                            metadata={
                                "entity_type": entity["entity_type"],
                                "description": entity["description"],
                                "chunk_ids": [chunk_key],
                            },
                        )
                    )
            return entities

        async def _merge_entities(entity_name: str, entities: List[Entity]) -> Entity:
            description_list = [e.metadata.description for e in entities]
            chunk_ids = [e.metadata.chunk_ids for e in entities]
            chunk_ids = [item for sublist in chunk_ids for item in sublist]
            entity_types = [e.metadata.entity_type for e in entities]

            # description aggregation
            description = await self.entity_description_summarizer.summarize_entity(
                entity_name, list(set(description_list))
            )
            # merge chunk_ids
            chunk_ids = list(set(chunk_ids))
            # merge entity_types
            entity_types = sorted(
                Counter(entity_types).items(),
                key=lambda x: x[1],
                reverse=True,
            )[0][0]

            entity = Entity(
                id=compute_mdhash_id(entity_name, prefix="ent-"),
                page_content=entity_name,
                metadata={
                    "entity_type": entity_types,
                    "description": description,
                    "chunk_ids": chunk_ids,
                },
            )
            return entity

        entity_extraction_concurrency: int = 4
        entity_merge_concurrency: int = 2

        extraction_coros = [_process_single_content_entity(chunk) for chunk in chunks]

        # entities_list is a list of list of entities
        # because _process_single_content_entity returns a list of entities
        # TODO: handle the concurrent entity extraction
        entities_list = await _limited_gather(
            extraction_coros, entity_extraction_concurrency
        )
        # Flatten the list of entity lists into a single list of entities
        entities = [entity for entity_list in entities_list for entity in entity_list]
        # merge entities with the same id
        # Check if the entities are shared by multiple chunks
        # If so, merge them into a single entity
        entities_name_count = Counter([entity.page_content for entity in entities])
        entities_unique = [
            e for e in entities if entities_name_count[e.page_content] == 1
        ]
        entities_to_merge = [
            e for e in entities if entities_name_count[e.page_content] > 1
        ]
        # TODO: handle the concurrent merge
        entities_to_merge_by_name = defaultdict(list)
        for entity in entities_to_merge:
            entities_to_merge_by_name[entity.page_content].append(entity)
        merge_coros = [
            _merge_entities(name, ents)
            for name, ents in entities_to_merge_by_name.items()
        ]
        merged_entities = await _limited_gather(merge_coros, entity_merge_concurrency)
        return entities_unique + merged_entities

    async def relation(
        self, chunks: List[Chunk], entities: List[Entity]
    ) -> List[Relation]:
        async def _process_single_content_relation(
            chunk: Chunk, entities_dict: Dict[str, Entity]
        ) -> List[Relation]:  # for each chunk, run the func
            chunk_key = chunk.id
            content = chunk.page_content
            # 1. initial extraction
            relation_extract_prompt = self.relation_extract_prompt.format(
                **self.relation_extract_context,
                entities=[e.page_content for e in entities_dict.values()],
                input_text=content,
            )  # fill in the parameter
            relation_string_result = await self.extract_func(
                model=self.llm_model_name,
                prompt=relation_extract_prompt,
            )  # feed into LLM with the prompt

            content_history = pack_user_ass_to_openai_messages(
                relation_extract_prompt, relation_string_result
            )  # set as history

            # 2. continue to extract relations for higher quality relations extraction, normally we only need 1 iteration
            for glean_idx in range(self.relation_extract_max_gleaning):
                glean_result = await self.extract_func(
                    model=self.llm_model_name,
                    prompt=self.continue_prompt,
                    history_messages=content_history,
                )

                content_history += pack_user_ass_to_openai_messages(
                    self.continue_prompt, glean_result
                )  # add to history
                relation_string_result += glean_result
                if glean_idx == self.relation_extract_max_gleaning - 1:
                    break

                relation_extraction_termination_str: str = (
                    await self.extract_func(  # judge if we still need the next iteration
                        model=self.llm_model_name,
                        prompt=self.relation_extraction_termination_prompt,
                        history_messages=content_history,
                    )
                )
                relation_extraction_termination_str = (
                    relation_extraction_termination_str.strip()
                    .strip('"')
                    .strip("'")
                    .lower()
                )
                if relation_extraction_termination_str != "yes":
                    break

            # 3. split relations from relation_string_result, which is the output of llm --> list of relations
            records = split_string_by_multi_markers(  # split entities from result --> list of entities
                relation_string_result,
                [
                    self.relation_extract_context["record_delimiter"],
                    self.relation_extract_context["completion_delimiter"],
                ],
            )

            # 4. Use regrex to extract the relation,
            # relations is a list of relations, where relation is a Relation object
            relations = []
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(  # split entity
                    record, [self.relation_extract_context["tuple_delimiter"]]
                )
                relation = await _handle_single_relationship_extraction(
                    record_attributes, chunk_key
                )
                if relation is not None:
                    try:
                        source = entities_dict[relation["src_id"]]
                    except KeyError:
                        warnings.warn(
                            f"Source entity {relation['src_id']} not found in entities_dict, skipping relation {relation}"
                        )
                        continue
                    try:
                        target = entities_dict[relation["tgt_id"]]
                    except KeyError:
                        warnings.warn(
                            f"Target entity {relation['tgt_id']} not found in entities_dict, skipping relation {relation}"
                        )
                        continue
                    relation = Relation(
                        source=source.id,
                        target=target.id,
                        properties={
                            "description": relation["description"],
                            "weight": relation["weight"],
                            "chunk_id": chunk.id,
                        },
                    )
                    relations.append(relation)
            return relations

        relation_extraction_concurrency: int = 5

        relation_coros = [
            _process_single_content_relation(
                chunk,
                {
                    e.page_content: e
                    for e in entities
                    if chunk.id in e.metadata.chunk_ids
                },
            )
            for chunk in chunks
        ]

        relations_list = await _limited_gather(
            relation_coros, relation_extraction_concurrency
        )

        relations = [
            relation for relation_list in relations_list for relation in relation_list
        ]
        # We do not merge relations here, because relations represents facts/relationships between entities
        # and it is supposed to have multiple relations between the same entities
        return relations
