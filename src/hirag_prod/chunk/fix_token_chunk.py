import tiktoken

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.schema import Chunk, File

from .base_chunk import BaseChunk


class FixTokenChunk(BaseChunk):
    def __init__(
        self, chunk_size: int, chunk_overlap: int, encoding_name="cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, document: File) -> list[Chunk]:
        text = document.page_content
        metadata = document.metadata
        document_id = document.id

        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap

        return [
            Chunk(
                id=compute_mdhash_id(chunk, prefix="chunk-"),
                page_content=chunk,
                metadata={
                    **metadata.__dict__,  # Get all attributes from metadata object
                    "chunk_idx": chunk_idx,
                    "document_id": document_id,
                },
            )
            for chunk_idx, chunk in enumerate(chunks)
        ]
