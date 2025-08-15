from typing import List

from pydantic import BaseModel, field_validator


class EntityMetadata(BaseModel):
    entity_type: str
    description: List[str]
    chunk_ids: List[str]
    workspace_id: str
    knowledge_base_id: str


class Entity(BaseModel):
    id: str
    page_content: str
    metadata: EntityMetadata

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v):
        if isinstance(v, dict):
            return EntityMetadata(**v)
        return v

    def to_document(self):
        from langchain_core.documents import Document

        return Document(page_content=self.page_content, metadata=self.metadata.dict())

    def to_flat_dict(self):
        return {
            "id": self.id,
            "page_content": self.page_content,
            **self.metadata.dict(),
        }
