from enum import Enum
from typing import Any, Dict, List, Optional

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.chunk import DotsHierarchicalChunker
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
def _extract_docling_chunk_meta(chunk) -> dict:
    """Extract page number and merged bbox from a Docling chunk."""
    min_l = float("inf")
    max_r = float("-inf")
    max_t = float("-inf")
    min_b = float("inf")
    page_no = None
    headers = chunk.meta.headings

    for item in chunk.meta.doc_items or []:
        for prov in item.prov or []:
            if page_no is None:
                page_no = prov.page_no
            bb = prov.bbox
            if bb.l < min_l:
                min_l = bb.l
            if bb.r > max_r:
                max_r = bb.r
            if bb.t > max_t:
                max_t = bb.t
            if bb.b < min_b:
                min_b = bb.b

    has_bbox = min_l != float("inf")
    return {
        "page_number": int(page_no) if page_no else None,
        "x_0": float(min_l) if has_bbox else None,
        "y_0": float(max_t) if has_bbox else None,
        "x_1": float(max_r) if has_bbox else None,
        "y_1": float(min_b) if has_bbox else None,
        "headers": headers if headers else None,
    }


def _build_doc_pages_map(doc: DoclingDocument) -> dict[int, tuple[float, float]]:
    """
    Build a map of page numbers to their bounding boxes.
    """
    size_map: dict[int, tuple[float, float]] = {}
    for k, pg in doc.pages.items():
        pn = pg.page_no if hasattr(pg, "page_no") else int(k)
        size_map[int(pn)] = (pg.size.width, pg.size.height)
    return size_map


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

    doc_pages_map = _build_doc_pages_map(docling_doc)

    # Convert to Chunk objects
    chunks = []
    for idx, chunk in enumerate(doc_chunks):
        chunk_type = determine_docling_chunk_type(chunk)
        docling_chunk_meta = _extract_docling_chunk_meta(chunk)

        page_number = docling_chunk_meta["page_number"]
        page_width = None
        page_height = None
        if page_number is not None:
            page_size = doc_pages_map.get(page_number)
            if page_size is not None:
                page_width, page_height = page_size

        # Convert x_0, y_0, x_1, y_1 to bbox format if coordinates are available
        bbox = None
        if all(
            coord is not None
            for coord in [
                docling_chunk_meta["x_0"],
                docling_chunk_meta["y_0"],
                docling_chunk_meta["x_1"],
                docling_chunk_meta["y_1"],
            ]
        ):
            bbox = [
                [
                    docling_chunk_meta["x_0"],
                    docling_chunk_meta["y_0"],
                    docling_chunk_meta["x_1"],
                    docling_chunk_meta["y_1"],
                ]
            ]

        metadata = ChunkMetadata(
            chunk_idx=idx,
            document_id=doc_md.id,
            chunk_type=chunk_type.value,
            page_number=page_number,
            page_image_url=None,
            page_width=float(page_width) if page_width is not None else None,
            page_height=float(page_height) if page_height is not None else None,
            bbox=bbox,
            caption=None,
            # TODO: If using docling in the future, may need to do indexing for headers
            headers=docling_chunk_meta["headers"],
            # inherit file metadata
            type=doc_md.metadata.type,
            filename=doc_md.metadata.filename,
            uri=doc_md.metadata.uri,
            private=doc_md.metadata.private,
            uploaded_at=doc_md.metadata.uploaded_at,
            knowledge_base_id=doc_md.metadata.knowledge_base_id,
            workspace_id=doc_md.metadata.workspace_id,
            children=None,
        )

        chunk_obj = Chunk(
            id=compute_mdhash_id(chunk.text, prefix="chunk-"),
            page_content=chunk.text,
            metadata=metadata,
        )

        chunks.append(chunk_obj)

    return chunks


# ======================== Dots OCR chunker ========================

"""
    Generate format: [{page_no: int, full_layout_info: [{bbox:[int, int, int, int], category: str, text: str}, ...boxes]}, ...pages ]
    Possible types: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
"""


def _dots_category_to_chunk_type(category: str) -> ChunkType:
    """
    Convert a dots OCR category to a ChunkType.

    Args:
        category: The category from dots OCR

    Returns:
        ChunkType: The corresponding chunk type
    """
    category_mapping = {
        "Caption": ChunkType.CAPTION,
        "Footnote": ChunkType.FOOTNOTE,
        "Formula": ChunkType.FORMULA,
        "List-item": ChunkType.LIST,
        "Page-footer": ChunkType.PAGE_FOOTER,
        "Page-header": ChunkType.PAGE_HEADER,
        "Picture": ChunkType.PICTURE,
        "Section-header": ChunkType.SECTION_HEADER,
        "Table": ChunkType.TABLE,
        "Text": ChunkType.TEXT,
        "Title": ChunkType.TITLE,
    }
    return category_mapping.get(category, ChunkType.UNKNOWN)


def _transform_bbox_dims(bboxes: List[List[float]], height) -> List[List[float]]:
    # The dim from dots is from top left, transform to bottom left
    transformed = []
    for bbox in bboxes:
        x_0, y_0, x_1, y_1 = bbox
        transformed.append([x_0, height - y_1, x_1, height - y_0])
    return transformed


def get_ToC_from_chunks(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    ToC = []
    vis_chunks = set()
    chunk_to_index = {}
    for idx, chunk in enumerate(chunks):
        chunk_to_index[chunk.id] = idx

    def _is_header(chunk: Chunk) -> bool:
        return chunk.metadata.chunk_type in [
            ChunkType.TITLE.value,
            ChunkType.SECTION_HEADER.value,
        ]

    def _extract_term(chunk: Chunk) -> Dict[str, Any]:
        if not _is_header(chunk):
            return None
        term = {
            "title": chunk.page_content,
            "chunk_id": chunk.id,
        }
        # Go through children
        valid_children = []
        for child_id in chunk.metadata.children:
            child_idx = chunk_to_index.get(child_id)
            vis_chunks.add(child_id)
            extracted_child = _extract_term(chunks[child_idx])
            if extracted_child:
                valid_children.append(extracted_child)

        term["children"] = valid_children
        return term

    for chunk in chunks:
        if chunk.id in vis_chunks:
            continue
        vis_chunks.add(chunk.id)
        term = _extract_term(chunk)
        if term:
            ToC.append(term)

    return ToC


def chunk_dots_document(
    json_doc: List[Dict[str, Any]],
    md_doc: File,
    left_bottom_origin: bool = True,
) -> List[Chunk]:
    """
    Split a dots document into chunks and return a list of Chunk objects.
    Each chunk will inherit metadata from the original document.
    """

    chunker = DotsHierarchicalChunker()

    # Get chunks from the hierarchical chunker
    dots_chunks = chunker.chunk(json_doc)

    # Convert DotsChunk objects to Chunk objects
    chunks = []

    # Mapping for tmp_chunk_idx to chunk_id
    chunk_id_mapping = {}

    # Try to obtain page dimensions
    page_dimensions = {}
    for page in json_doc:
        page_no = page.get("page_no")
        if page_no is not None:
            page_dimensions[page_no] = {
                "width": page.get("width"),
                "height": page.get("height"),
            }

    for tmp_chunk_idx, dots_chunk in dots_chunks.items():
        # Convert dots category to chunk type
        chunk_type = _dots_category_to_chunk_type(dots_chunk.category)
        page_size = page_dimensions.get(dots_chunk.page_no, {})
        page_width = page_size.get("width", None)
        page_height = page_size.get("height", None)

        bboxes = dots_chunk.bbox

        if left_bottom_origin and page_height is not None:
            bboxes = _transform_bbox_dims(bboxes, page_height)

        metadata = ChunkMetadata(
            chunk_idx=tmp_chunk_idx,
            document_id=md_doc.id,
            chunk_type=chunk_type.value,
            page_number=dots_chunk.page_no,
            page_image_url=None,
            page_width=page_width,
            page_height=page_height,
            bbox=bboxes,
            caption=dots_chunk.caption,
            # Terms using temporary idx, would be filled later after chunks created
            headers=None,
            children=None,
            # inherit file metadata
            type=md_doc.metadata.type,
            filename=md_doc.metadata.filename,
            uri=md_doc.metadata.uri,
            private=md_doc.metadata.private,
            uploaded_at=md_doc.metadata.uploaded_at,
            knowledge_base_id=md_doc.metadata.knowledge_base_id,
            workspace_id=md_doc.metadata.workspace_id,
        )

        # Create the chunk content, including caption if available
        content = dots_chunk.text

        chunk_id = compute_mdhash_id(content, prefix="chunk-")

        chunk_obj = Chunk(
            id=chunk_id,
            page_content=content,
            metadata=metadata,
        )

        chunk_id_mapping[tmp_chunk_idx] = chunk_id

        chunks.append(chunk_obj)

    chunks.sort(key=lambda c: c.metadata.chunk_idx)

    # For all headings & children, do a mapping
    for chunk in chunks:
        chunk_meta = chunk.metadata
        tmp_idx = chunk_meta.chunk_idx
        raw_headers = dots_chunks[tmp_idx].headings if tmp_idx in dots_chunks else []
        if raw_headers:
            # Map the chunk ID to the heading text
            header_ids = [chunk_id_mapping[h] for h in raw_headers]
            chunk_meta.headers = header_ids

        raw_children = dots_chunks[tmp_idx].children if tmp_idx in dots_chunks else []
        if raw_children:
            # Map the chunk ID to the children
            child_ids = [chunk_id_mapping[c] for c in raw_children]
            chunk_meta.children = child_ids

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
            # Fields required by ChunkMetadata but not applicable for langchain chunks
            page_image_url=None,
            page_width=None,
            page_height=None,
            bbox=None,
            caption=None,
            headers=None,
            knowledge_base_id=langchain_doc.metadata.knowledge_base_id,
            workspace_id=langchain_doc.metadata.workspace_id,
            children=None,
        )

        chunk_obj = Chunk(
            id=compute_mdhash_id(chunk, prefix="chunk-"),
            page_content=chunk,
            metadata=metadata,
        )

        chunks.append(chunk_obj)

    return chunks
