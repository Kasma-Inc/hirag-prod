from enum import Enum
from typing import List, Optional

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.schema.chunk import Chunk, ChunkMetadata
from hirag_prod.schema.file import File

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SEPARATORS = ["=+=+=+=+=+=+=+=+="]


class ChunkType(Enum):
    """Enumeration of chunk types based on DocItem labels."""

    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    CODE = "code"
    CAPTION = "caption"
    FORMULA = "formula"
    PICTURE = "picture"
    CHART = "chart"
    FOOTNOTE = "footnote"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    DOCUMENT_INDEX = "document_index"
    CHECKBOX = "checkbox"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"
    GRADING_SCALE = "grading_scale"
    HANDWRITTEN_TEXT = "handwritten_text"
    EMPTY_VALUE = "empty_value"
    REFERENCE = "reference"
    MIXED = "mixed"
    UNKNOWN = "unknown"


LABEL_TO_CHUNK_TYPE = {
    DocItemLabel.TEXT: ChunkType.TEXT,
    DocItemLabel.TABLE: ChunkType.TABLE,
    DocItemLabel.LIST_ITEM: ChunkType.LIST,
    DocItemLabel.TITLE: ChunkType.TITLE,
    DocItemLabel.SECTION_HEADER: ChunkType.SECTION_HEADER,
    DocItemLabel.PARAGRAPH: ChunkType.PARAGRAPH,
    DocItemLabel.CODE: ChunkType.CODE,
    DocItemLabel.CAPTION: ChunkType.CAPTION,
    DocItemLabel.FORMULA: ChunkType.FORMULA,
    DocItemLabel.PICTURE: ChunkType.PICTURE,
    DocItemLabel.CHART: ChunkType.CHART,
    DocItemLabel.FOOTNOTE: ChunkType.FOOTNOTE,
    DocItemLabel.PAGE_FOOTER: ChunkType.PAGE_FOOTER,
    DocItemLabel.PAGE_HEADER: ChunkType.PAGE_HEADER,
    DocItemLabel.DOCUMENT_INDEX: ChunkType.DOCUMENT_INDEX,
    DocItemLabel.CHECKBOX_SELECTED: ChunkType.CHECKBOX,
    DocItemLabel.CHECKBOX_UNSELECTED: ChunkType.CHECKBOX,
    DocItemLabel.FORM: ChunkType.FORM,
    DocItemLabel.KEY_VALUE_REGION: ChunkType.KEY_VALUE_REGION,
    DocItemLabel.GRADING_SCALE: ChunkType.GRADING_SCALE,
    DocItemLabel.HANDWRITTEN_TEXT: ChunkType.HANDWRITTEN_TEXT,
    DocItemLabel.EMPTY_VALUE: ChunkType.EMPTY_VALUE,
    DocItemLabel.REFERENCE: ChunkType.REFERENCE,
}


# ======================== docling chunker ========================
def determine_docling_chunk_type(chunk) -> ChunkType:
    """
    Determine the type of a chunk based on its doc_items.

    Args:
        chunk: A DocChunk object containing doc_items with labels

    Returns:
        ChunkType: The determined type of the chunk
    """
    if not chunk.meta.doc_items:
        return ChunkType.UNKNOWN

    # Get all unique labels from doc_items
    labels = {item.label for item in chunk.meta.doc_items}

    # If only one type of item, return that type
    if len(labels) == 1:
        label = labels.pop()
        return LABEL_TO_CHUNK_TYPE.get(label, ChunkType.UNKNOWN)

    # If multiple types, return MIXED
    return ChunkType.MIXED


def chunk_docling_document(docling_doc: DoclingDocument, doc_md: File) -> List[Chunk]:
    """
    Split a docling document into chunks and return a list of Chunk objects.
    Each chunk will inherit metadata from the original document.

    Args:
        docling_doc: The docling document to be chunked
        doc_md: File object containing file information
               (type, filename, uri, etc.) that will be inherited by each chunk

    Returns:
        List[Chunk]: A list of Chunk objects with proper metadata including
                    chunk-specific metadata and inherited file metadata
    """
    # Initialize the chunker
    chunker = HierarchicalChunker()

    # Generate chunks from the document
    doc_chunks = chunker.chunk(docling_doc)

    # Convert to Chunk objects
    chunks = []
    for idx, chunk in enumerate(doc_chunks):
        chunk_type = determine_docling_chunk_type(chunk)

        metadata = ChunkMetadata(
            chunk_idx=idx,
            document_id=doc_md.id,
            chunk_type=chunk_type.value,
            # Inherit file metadata from doc_md
            type=doc_md.metadata.type,
            filename=doc_md.metadata.filename,
            page_number=doc_md.metadata.page_number,
            uri=doc_md.metadata.uri,
            private=doc_md.metadata.private,
            uploaded_at=doc_md.metadata.uploaded_at,
        )

        chunk_obj = Chunk(
            id=compute_mdhash_id(chunk.text, prefix="chunk-"),
            page_content=chunk.text,
            metadata=metadata,
        )

        chunks.append(chunk_obj)

    return chunks


# ======================== langchain chunker ========================
def chunk_langchain_document(
    langchain_doc: File,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Optional[List[str]] = SEPARATORS,
    keep_separator: bool = True,
) -> List[Chunk]:
    """
    Split a langchain document into chunks and return a list of Chunk objects.
    Each chunk will inherit metadata from the original document.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        keep_separator=keep_separator,
        is_separator_regex=False,
    )
    chunk_texts = text_splitter.split_text(langchain_doc.page_content)

    chunks = []
    for idx, chunk in enumerate(chunk_texts):
        metadata = ChunkMetadata(
            chunk_idx=idx,
            document_id=langchain_doc.id,
            chunk_type="text",
            # Inherit file metadata from doc_md
            type=langchain_doc.metadata.type,
            filename=langchain_doc.metadata.filename,
            page_number=langchain_doc.metadata.page_number,
            uri=langchain_doc.metadata.uri,
            private=langchain_doc.metadata.private,
            uploaded_at=langchain_doc.metadata.uploaded_at,
        )

        chunk_obj = Chunk(
            id=compute_mdhash_id(chunk, prefix="chunk-"),
            page_content=chunk,
            metadata=metadata,
        )

        chunks.append(chunk_obj)

    return chunks


# ======================== OCR chunker ========================
# TODO: Implement OCR chunker
