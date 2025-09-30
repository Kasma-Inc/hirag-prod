from enum import Enum
from typing import Any, Dict, List, Optional, Union

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hirag_prod._utils import (
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
)
from hirag_prod.chunk import DotsHierarchicalChunker, UnifiedRecursiveChunker
from hirag_prod.schema import Chunk, File, Item

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


# ======================== Helper Functions ========================
def _copy_chunk_with_updates(source_chunk: Chunk, **updates) -> Chunk:
    """Create a new Chunk by copying all fields from source and applying updates."""
    new_chunk = Chunk()
    for col in dict(source_chunk):
        if hasattr(new_chunk, col):
            setattr(new_chunk, col, getattr(source_chunk, col))

    # Apply updates
    for key, value in updates.items():
        if hasattr(new_chunk, key):
            setattr(new_chunk, key, value)

    return new_chunk


def split_long_text_chunks(chunks: List[Chunk], chunk_max_tokens: int = 8192):
    """Split chunks that exceed token limit into smaller chunks."""
    if not chunks or not chunk_max_tokens:
        return chunks

    splitted_chunks: List[Chunk] = []

    for ck in chunks:
        # Skip chunks that embed from caption, not text
        if getattr(ck, "chunkType", None) in ("table", "excel_sheet"):
            splitted_chunks.append(ck)
            continue

        text = ck.text or ""
        toks = encode_string_by_tiktoken(text)
        if len(toks) <= chunk_max_tokens:
            splitted_chunks.append(ck)
            continue

        # Split into contiguous token windows (no overlap)
        for i in range(0, len(toks), chunk_max_tokens):
            part_text = decode_tokens_by_tiktoken(toks[i : i + chunk_max_tokens])
            new_key = compute_mdhash_id(part_text, prefix="chunk-")

            # Create new chunk by copying all fields and updating text & documentKey
            splitted_chunks.append(
                _copy_chunk_with_updates(
                    ck,
                    documentKey=new_key,
                    text=part_text,
                )
            )

    return splitted_chunks


def _inherit_file_metadata(source_file: File) -> Dict[str, Any]:
    """Extract inheritable metadata from a File object."""
    return {
        "id": source_file.id,
        "type": source_file.type,
        "fileName": source_file.fileName,
        "uri": source_file.uri,
        "private": source_file.private,
        "createdAt": source_file.createdAt,
        "updatedAt": source_file.updatedAt,
        "createdBy": source_file.createdBy,
        "updatedBy": source_file.updatedBy,
        "knowledgeBaseId": source_file.knowledgeBaseId,
        "workspaceId": source_file.workspaceId,
    }


def _create_bbox_from_coords(
    x_0: float, y_0: float, x_1: float, y_1: float
) -> Optional[List[float]]:
    """Create bbox list from coordinates if all are not None."""
    if all(coord is not None for coord in [x_0, y_0, x_1, y_1]):
        return [x_0, y_0, x_1, y_1]
    return None


def _extract_page_dimensions(
    page_info: Dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Extract page width and height from page info dictionary."""
    page_width = page_info.get("width")
    page_height = page_info.get("height")
    return (
        float(page_width) if page_width is not None else None,
        float(page_height) if page_height is not None else None,
    )


def _create_item_base(
    text: str,
    chunk_idx: int,
    document_id: str,
    chunk_type: str,
    source_file: File,
    page_number: Optional[int] = None,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
    bbox: Optional[List[float]] = None,
    caption: Optional[str] = None,
    headers: Optional[List[str]] = None,
    children: Optional[List[str]] = None,
) -> Item:
    """Create an Item object with standard fields and inherited metadata."""
    file_metadata = _inherit_file_metadata(source_file)

    return Item(
        documentKey=compute_mdhash_id(text, prefix="item-"),
        text=text,
        chunkIdx=chunk_idx,
        documentId=document_id,
        chunkType=chunk_type,
        pageNumber=page_number,
        pageImageUrl=None,
        pageWidth=page_width,
        pageHeight=page_height,
        bbox=bbox,
        caption=caption,
        headers=headers,
        children=children,
        **file_metadata,
    )


def _create_chunk_base(
    text: str,
    chunk_idx: int,
    document_id: str,
    chunk_type: str,
    file_metadata: Dict[str, Any],
    page_number: Optional[Union[int, List[int]]] = None,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
    bbox: Optional[List[List[float]]] = None,
    caption: Optional[str] = None,
    headers: Optional[List[str]] = None,
    children: Optional[List[str]] = None,
) -> Chunk:
    """Create a Chunk object with standard fields and provided metadata."""
    return Chunk(
        documentKey=compute_mdhash_id(text, prefix="chunk-"),
        text=text,
        chunkIdx=chunk_idx,
        documentId=document_id,
        chunkType=chunk_type,
        pageNumber=page_number,
        pageImageUrl=None,
        pageWidth=page_width,
        pageHeight=page_height,
        bbox=bbox,
        caption=caption,
        headers=headers,
        children=children,
        **file_metadata,
    )


def _find_text_positions(
    texts: List[str], original_text: str, chunk_overlap: int = 0
) -> Dict[int, Optional[tuple[int, int]]]:
    """Find character positions of text chunks in original content."""
    id2pos = {}
    search_start = 0

    for idx, chunk_text in enumerate(texts):
        start_pos = original_text.find(chunk_text, search_start)

        if start_pos == -1:
            id2pos[idx] = None
            continue

        end_pos = start_pos + len(chunk_text)
        id2pos[idx] = (start_pos, end_pos)
        search_start = start_pos + len(chunk_text) - chunk_overlap

    return id2pos


# ======================== docling chunker ========================
def _extract_docling_chunk_meta(chunk) -> dict:
    """Extract page number and merged bbox from a Docling chunk."""
    chunk_idx = chunk.meta.chunk_idx
    min_l = float("inf")
    max_r = float("-inf")
    max_t = float("-inf")
    min_b = float("inf")
    page_no = None
    headers = chunk.meta.headings
    children = chunk.meta.children

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
        "chunk_idx": chunk_idx,
        "page_number": int(page_no) if page_no else None,
        "x_0": float(min_l) if has_bbox else None,
        "y_0": float(max_t) if has_bbox else None,
        "x_1": float(max_r) if has_bbox else None,
        "y_1": float(min_b) if has_bbox else None,
        "headers": headers if headers else None,
        "children": children if children else None,
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


def chunk_docling_document(
    docling_doc: DoclingDocument, doc_md: File
) -> tuple[List[Item], set[str], List[int]]:
    """
    Split a docling document into chunks and return a list of Item objects.
    Each chunk will inherit metadata from the original document.

    Args:
        docling_doc: The docling document to be chunked
        doc_md: File object containing file information
            (type, filename, uri, etc.) that will be inherited by each chunk

    Returns:
        (items, header_set, table_items_idx)
    """
    # Initialize the chunker
    chunker = HierarchicalChunker()

    # Generate chunks from the document
    doc_chunks = chunker.chunk(docling_doc)

    doc_pages_map = _build_doc_pages_map(docling_doc)

    # Convert to Chunk objects
    chunks = []
    chunk_id_mapping = {}
    table_items_idx = []

    for _, chunk in enumerate(doc_chunks):
        chunk_type = determine_docling_chunk_type(chunk)
        docling_chunk_meta = _extract_docling_chunk_meta(chunk)

        page_number = docling_chunk_meta["page_number"]
        page_width, page_height = None, None
        if page_number is not None:
            page_size = doc_pages_map.get(page_number)
            if page_size is not None:
                page_width, page_height = _extract_page_dimensions(
                    {"width": page_size[0], "height": page_size[1]}
                )

        # Convert x_0, y_0, x_1, y_1 to bbox format if coordinates are available
        bbox = _create_bbox_from_coords(
            docling_chunk_meta["x_0"],
            docling_chunk_meta["y_0"],
            docling_chunk_meta["x_1"],
            docling_chunk_meta["y_1"],
        )

        text = chunk.text
        chunk_obj = _create_item_base(
            text=text,
            chunk_idx=docling_chunk_meta["chunk_idx"],
            document_id=doc_md.documentKey,
            chunk_type=chunk_type.value,
            source_file=doc_md,
            page_number=(
                page_number - 1 if page_number is not None else None
            ),  # Convert to 0-based index
            page_width=page_width,
            page_height=page_height,
            bbox=bbox,
            caption=None,
            headers=(
                docling_chunk_meta["headers"] if docling_chunk_meta["headers"] else None
            ),
            children=(
                docling_chunk_meta["children"]
                if docling_chunk_meta["children"]
                else None
            ),
        )
        if chunk_type == ChunkType.TABLE:
            table_items_idx.append(chunk_obj.chunkIdx)
        chunks.append(chunk_obj)
        chunk_id_mapping[chunk_obj.chunkIdx] = chunk_obj.documentKey

    # Translate all chunk IDs to their document keys
    header_set = set()

    for chunk in chunks:
        if chunk.headers:
            chunk.headers = [
                chunk_id_mapping.get(header_id) for header_id in chunk.headers
            ]
            header_set.update([h for h in chunk.headers if h is not None])
        if chunk.children:
            chunk.children = [
                chunk_id_mapping.get(child_id) for child_id in chunk.children
            ]

    chunks.sort(key=lambda c: c.chunkIdx)

    return chunks, header_set, table_items_idx


def obtain_docling_md_bbox(
    docling_doc: DoclingDocument,
    raw_md: str,
    items: List[Item] = None,
) -> List[Item]:
    """
    Finish the bbox for items after docling chunking if the file is markdown format.
    This function adds character position information as bbox for markdown text chunks.
    Uses the original file content instead of exported markdown for consistency.
    """
    if not items:
        return []

    original_content = raw_md

    if not original_content:
        # Fallback to exported markdown if original content is not available
        original_content = docling_doc.export_to_markdown()

    # Create a mapping to store character positions for each item
    id2pos = {}

    # Search for each item's text in the original content to get positions
    search_start = 0
    for item in items:
        # Clean the item text for better matching (remove extra whitespace)
        clean_item_text = item.text.strip()
        if not clean_item_text:
            id2pos[item.documentKey] = None
            continue

        # Try exact match first
        start_pos = original_content.find(clean_item_text, search_start)

        if start_pos == -1:
            # If exact match fails, try with normalized
            normalized_item = "".join(c.strip() for c in clean_item_text.split())
            normalized_content = "".join(
                o.strip() for o in original_content[search_start:].split()
            )

            relative_pos = normalized_content.find(normalized_item)
            if relative_pos != -1:
                start_pos = search_start + relative_pos
            else:
                id2pos[item.documentKey] = None
                continue

        end_pos = start_pos + len(clean_item_text)
        id2pos[item.documentKey] = (start_pos, end_pos)

        # Move search start forward to avoid overlapping matches
        search_start = start_pos + len(clean_item_text)

    # Update items with bbox information (character positions)
    updated_items = []
    for item in items:
        # Create a copy of the item with updated bbox
        pos_info = id2pos.get(item.documentKey)

        # Update the bbox field with character positions if found
        if pos_info:
            item.bbox = list(pos_info)  # [start_pos, end_pos]

        updated_items.append(item)

    # Sort by chunk index to maintain order
    updated_items.sort(key=lambda c: c.chunkIdx)
    return updated_items


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


def _transform_bbox_dims(bbox: List[float], height) -> List[float]:
    # The dim from dots is from top left, transfer to dim bottom left but keep same order of dims that new y0 > new y1
    x_0, y_0, x_1, y_1 = bbox
    return [x_0, height - y_0, x_1, height - y_1]


def _transform_bbox_dims_list(
    bboxes: Optional[List[List[float]]],
    height: Optional[float],
) -> Optional[List[List[float]]]:
    return [_transform_bbox_dims(bbox, height) for bbox in bboxes]


def get_toc_from_items(items: List[Item]) -> List[Dict[str, Any]]:
    ToC = []
    vis_items = set()
    item_to_index = {}
    for idx, item in enumerate(items):
        item_to_index[item.documentKey] = idx

    def _is_header(item: Item) -> bool:
        return item.chunkType in [
            ChunkType.TITLE.value,
            ChunkType.SECTION_HEADER.value,
        ]

    def _extract_term(item: Item) -> Dict[str, Any]:
        if not _is_header(item):
            return None

        term = {
            "title": item.text,
            "chunk_id": item.documentKey,
        }

        # Go through children
        valid_children = []
        if item.children:
            for child_id in item.children:
                child_idx = item_to_index.get(child_id)
                if child_id in vis_items or child_idx is None:
                    continue
                vis_items.add(child_id)
                extracted_child = _extract_term(items[child_idx])
                if extracted_child:
                    valid_children.append(extracted_child)

        term["children"] = valid_children
        return term

    for item in items:
        if item.documentKey in vis_items:
            continue
        vis_items.add(item.documentKey)
        term = _extract_term(item)
        if term:
            ToC.append(term)

    return ToC


def build_rich_toc(items: List[Item], file: File) -> Dict[str, Any]:
    id2item = {i.documentKey: i for i in items}
    tree = get_toc_from_items(items)
    blocks: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], level: int) -> None:
        iid = node.get("chunk_id")
        if not iid:
            return
        i = id2item.get(iid)
        if not i:
            return

        bbox = i.bbox or [0, 0, 0, 0]
        source_bbox = {}
        if len(bbox) == 4:
            source_bbox = {
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
            }
        elif len(bbox) == 2:
            source_bbox = {
                "start_char": bbox[0],
                "end_char": bbox[1],
            }

        blocks.append(
            {
                "type": i.chunkType,
                "hierarchyLevel": level,
                "id": i.documentKey,
                "sourceBoundingBox": source_bbox,
                "markdown": i.text or "",
                "pageIndex": i.pageNumber or 0,
                "fileUrl": i.uri or "",
            }
        )

        for child in node.get("children", []):
            visit(child, level + 1)

    for root in tree:
        visit(root, 0)

    content = "\n".join(b.get("markdown", "") for b in blocks if b.get("markdown"))

    return {
        "fileName": file.fileName or "",
        "markdownDocument": (file.text or ""),
        "hierarchy": {
            "content": content,
            "blocks": blocks,
        },
    }


def chunk_dots_document(
    json_doc: List[Dict[str, Any]],
    md_doc: File,
    dots_left_bottom_origin: bool = True,
) -> tuple[List[Item], set[str], List[int]]:
    """
    Split a dots document into chunks and return a list of Chunk objects.
    Each chunk will inherit metadata from the original document.

    Returns:
        (items, header_set, table_items_idx)
    """

    chunker = DotsHierarchicalChunker()

    # Get chunks from the hierarchical chunker
    dots_chunks = chunker.chunk(json_doc)

    # Convert DotsChunk objects to Chunk objects
    chunks = []
    table_items_idx = []
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
        page_width, page_height = _extract_page_dimensions(page_size)

        bbox = dots_chunk.bbox
        if dots_left_bottom_origin and page_height is not None:
            bbox = _transform_bbox_dims(bbox, page_height)

        # Create the chunk content, including caption if available
        content = dots_chunk.text

        chunk_obj = _create_item_base(
            text=content,
            chunk_idx=tmp_chunk_idx,
            document_id=md_doc.documentKey,
            chunk_type=chunk_type.value,
            source_file=md_doc,
            page_number=dots_chunk.page_no,
            page_width=page_width,
            page_height=page_height,
            bbox=bbox,
            caption=dots_chunk.caption,
            headers=None,  # Will be filled later after chunks created
            children=None,
        )
        if chunk_type == ChunkType.TABLE:
            table_items_idx.append(chunk_obj.chunkIdx)
        chunk_id_mapping[tmp_chunk_idx] = chunk_obj.documentKey
        chunks.append(chunk_obj)

    chunks.sort(key=lambda c: c.chunkIdx)

    header_set = set()
    # For all headings & children, do a mapping
    for chunk in chunks:
        tmp_idx = chunk.chunkIdx
        raw_headers = dots_chunks[tmp_idx].headings if tmp_idx in dots_chunks else []
        if raw_headers:
            # Map the chunk ID to the heading text
            header_ids = [chunk_id_mapping[h] for h in raw_headers]
            chunk.headers = header_ids
            header_set.update(header_ids)

        raw_children = dots_chunks[tmp_idx].children if tmp_idx in dots_chunks else []
        if raw_children:
            # Map the chunk ID to the children
            child_ids = [chunk_id_mapping[c] for c in raw_children]
            chunk.children = child_ids

    return chunks, header_set, table_items_idx


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
    chunk_texts = text_splitter.split_text(langchain_doc.text)

    chunks = []
    original_text = langchain_doc.text
    print("Original text length:", len(original_text))

    # Find text positions for all chunks
    id2pos = _find_text_positions(chunk_texts, original_text, chunk_overlap)

    for idx, chunk in enumerate(chunk_texts):
        bbox = list(id2pos.get(idx)) if id2pos.get(idx) else None

        chunk_obj = _create_item_base(
            text=chunk,
            chunk_idx=idx,
            document_id=langchain_doc.documentKey,
            chunk_type="text",
            source_file=langchain_doc,
            page_number=langchain_doc.pageNumber,
            bbox=bbox,
        )

        chunks.append(chunk_obj)

    return chunks


# ======================== Unified Chunkers ========================


def items_to_chunks_recursive(
    items: Optional[List[Item]] = None,
    header_set: Optional[set[str]] = None,
    chunk_max_tokens: Optional[int] = 8192,
) -> List[Chunk]:
    """
    Split a dots document into chunks using UnifiedRecursiveChunker and return a list of Chunk objects.
    This produces aggregated chunks that may span multiple pages, with pageNumber and bbox aligned by page.
    """
    chunker = UnifiedRecursiveChunker()
    dense_chunks = chunker.chunk(items, header_set)

    chunks: List[Chunk] = []
    for dchunk in dense_chunks:
        tbbox_list = dchunk.bbox

        file_metadata = {
            "id": dchunk.id,
            "type": dchunk.document_type,
            "fileName": dchunk.file_name,
            "uri": dchunk.uri,
            "private": dchunk.private,
            "createdAt": dchunk.created_at,
            "updatedAt": dchunk.updated_at,
            "createdBy": dchunk.created_by,
            "updatedBy": dchunk.updated_by,
            "knowledgeBaseId": dchunk.knowledge_base_id,
            "workspaceId": dchunk.workspace_id,
        }

        chunk_obj = _create_chunk_base(
            text=dchunk.text,
            chunk_idx=dchunk.chunk_idx,
            document_id=dchunk.document_id,
            chunk_type=dchunk.category,
            file_metadata=file_metadata,
            page_number=dchunk.pages_span,
            page_width=dchunk.page_width,
            page_height=dchunk.page_height,
            bbox=tbbox_list if tbbox_list else None,
            caption=dchunk.caption,
            headers=dchunk.headings,
        )
        chunks.append(chunk_obj)

    chunks.sort(key=lambda c: c.chunkIdx)
    if chunk_max_tokens and chunk_max_tokens > 0:
        chunks = split_long_text_chunks(chunks, chunk_max_tokens)
    return chunks
