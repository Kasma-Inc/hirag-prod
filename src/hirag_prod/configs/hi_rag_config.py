from functools import cached_property
from typing import Literal, get_args

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class HiRAGConfig(BaseSettings):
    """HiRAG system configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: f"hi_rag_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    language: Literal["en", "cn-s", "cn-t"] = Field(
        "en", alias="HI_RAG_LANGUAGE", description="The language used for prompts."
    )

    # Database configuration
    vdb_type: Literal["pgvector"] = "pgvector"

    # TODO(tatiana): check whether the default values are updated to best experience values
    # Chunking configuration
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # whether to construct graph
    construct_graph: bool = Field(
        False,
        description="Whether to construct graph for indexing."
        " Constructing graph improves the retrieval quality at the cost of token usage.",
    )

    # Batch processing configuration
    embedding_batch_size: int = 1000
    entity_upsert_concurrency: int = 32
    relation_upsert_concurrency: int = 32

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    similarity_threshold: float = 0.5
    similarity_max_difference: float = 0.15
    max_references: int = 3

    # Basic Query Configuration
    max_chunk_ids_per_query: int = 10
    default_query_top_k: int = 10
    default_query_top_n: int = 5
    # Pagerank Configuration
    default_link_top_k: int = 30
    default_passage_node_weight: float = 0.6
    default_pagerank_damping: float = 0.5
    # Clustering Configuration
    clustering_n_clusters: int = 3
    clustering_distance_threshold: float = 0.5
    clustering_linkage_method: Literal["ward", "complete", "average", "single"] = "ward"
    clustering_n_type: Literal["fixed", "distance"] = "fixed"  # 'fixed' or 'distance'
    # Similarity search Configuration
    default_distance_threshold: float = 0.8

    @cached_property
    def supported_languages(self) -> list[str]:
        return list(get_args(self.model_fields.get("language").annotation))
