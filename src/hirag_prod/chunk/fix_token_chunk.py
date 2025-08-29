from langchain_text_splitters import Tokenizer
from langchain_text_splitters.base import split_text_on_tokens

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.chunk.base_chunk import BaseChunk
from hirag_prod.schema import Chunk, File, file_to_chunk


class FixTokenChunk(BaseChunk):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: File) -> list[Chunk]:
        # TODO: Implement semantic-aware chunking to preserve context boundaries
        tokenizer = Tokenizer(
            tokens_per_chunk=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            decode=(lambda it: "".join(chr(i) for i in it)),
            encode=(lambda it: [ord(c) for c in it]),
        )
        chunks = split_text_on_tokens(
            text=document.text,
            tokenizer=tokenizer,
        )
        document_id = document.documentKey

        return [
            file_to_chunk(
                file=document,
                documentKey=compute_mdhash_id(chunk, prefix="chunk-"),
                text=chunk,
                chunk_idx=chunk_idx,
                document_id=document_id,
            )
            for chunk_idx, chunk in enumerate(chunks)
        ]
