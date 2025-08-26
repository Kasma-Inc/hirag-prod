import pyarrow as pa


def get_chunks_schema(embedding_dimension: int) -> pa.Schema:
    """Return LanceDB chunks table schema.

    Kept external so LanceDB schema initialization mirrors pgvector's external ORM models.
    """
    return pa.schema(
        [
            pa.field("knowledgeBaseId", pa.string()),
            pa.field("workspaceId", pa.string()),
            pa.field("text", pa.string()),
            pa.field("documentKey", pa.string()),
            pa.field("type", pa.string()),
            pa.field("fileName", pa.string()),
            pa.field("pageNumber", pa.int32()),
            pa.field("uri", pa.string()),
            pa.field("private", pa.bool_()),
            pa.field("chunkIdx", pa.int32()),
            pa.field("documentId", pa.string()),
            pa.field("chunkType", pa.string()),
            # Optional metadata fields from ChunkMetadata
            pa.field("pageImageUrl", pa.string()),
            pa.field("pageWidth", pa.float32()),
            pa.field("pageHeight", pa.float32()),
            pa.field("x0", pa.float32()),
            pa.field("y0", pa.float32()),
            pa.field("x1", pa.float32()),
            pa.field("y1", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dimension)),
            pa.field("updatedAt", pa.timestamp("us")),
        ]
    )


def get_relations_schema(embedding_dimension: int) -> pa.Schema:
    """Return LanceDB relations table schema."""
    return pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("target", pa.string()),
            pa.field("description", pa.string()),
            pa.field("fileName", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dimension)),
            pa.field("knowledgeBaseId", pa.string()),
            pa.field("workspaceId", pa.string()),
        ]
    )
