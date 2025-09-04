from enum import Enum
from typing import Any, Dict, List, Optional

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.chunk import DotsHierarchicalChunker, DotsRecursiveChunker
from hirag_prod.schema.chunk import Chunk
from hirag_prod.schema.file import File
from hirag_prod.schema.item import Item

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
    chunk_id_mapping = {}

    for _, chunk in enumerate(doc_chunks):
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
                docling_chunk_meta["x_0"],
                docling_chunk_meta["y_0"],
                docling_chunk_meta["x_1"],
                docling_chunk_meta["y_1"],
            ]

        chunk_obj = Chunk(
            documentKey=compute_mdhash_id(chunk.text, prefix="chunk-"),
            text=chunk.text,
            chunkIdx=docling_chunk_meta["chunk_idx"],
            documentId=doc_md.documentKey,
            chunkType=chunk_type.value,
            pageNumber=page_number,
            pageImageUrl=None,
            pageWidth=float(page_width) if page_width is not None else None,
            pageHeight=float(page_height) if page_height is not None else None,
            bbox=bbox,
            caption=None,
            # TODO: If using docling in the future, may need to do indexing for headers
            headers=(
                docling_chunk_meta["headers"] if docling_chunk_meta["headers"] else None
            ),
            # inherit file metadata
            type=doc_md.type,
            fileName=doc_md.fileName,
            uri=doc_md.uri,
            private=doc_md.private,
            uploadedAt=doc_md.uploadedAt,
            knowledgeBaseId=doc_md.knowledgeBaseId,
            workspaceId=doc_md.workspaceId,
            children=(
                docling_chunk_meta["children"]
                if docling_chunk_meta["children"]
                else None
            ),
        )

        chunks.append(chunk_obj)
        chunk_id_mapping[chunk_obj.chunkIdx] = chunk_obj.documentKey

    # Translate all chunk IDs to their document keys
    for chunk in chunks:
        if chunk.headers:
            chunk.headers = [
                chunk_id_mapping.get(header_id) for header_id in chunk.headers
            ]
        if chunk.children:
            chunk.children = [
                chunk_id_mapping.get(child_id) for child_id in chunk.children
            ]

    chunks.sort(key=lambda c: c.chunkIdx)

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


def _transform_bbox_dims(bbox: List[float], height) -> List[float]:
    # The dim from dots is from top left, transfer to dim bottom left but keep same order of dims that new y0 > new y1
    x_0, y_0, x_1, y_1 = bbox
    return [x_0, height - y_0, x_1, height - y_1]


def _collect_dots_page_dimensions(
    json_doc: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    page_dimensions: Dict[int, Dict[str, Any]] = {}
    for page in json_doc:
        page_no = page.get("page_no")
        if page_no is not None:
            page_dimensions[page_no] = {
                "width": page.get("width"),
                "height": page.get("height"),
            }
    return page_dimensions


def _transform_bbox_dims_list(
    bboxes: Optional[List[List[float]]],
    pages: Optional[List[int]],
    page_dimensions: Dict[int, Dict[str, Any]],
) -> Optional[List[List[float]]]:
    if not bboxes or not pages:
        return bboxes
    out: List[List[float]] = []
    for bb, pn in zip(bboxes, pages):
        ph = (page_dimensions.get(pn) or {}).get("height")
        if ph is None:
            out.append(bb)
        else:
            out.append(_transform_bbox_dims(bb, ph))
    return out


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
                if child_id in vis_items:
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

        blocks.append(
            {
                "type": i.chunkType,
                "hierarchyLevel": level,
                "id": i.documentKey,
                "sourceBoundingBox": {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                },
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
    left_bottom_origin: bool = True,
) -> List[Item]:
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

        bbox = dots_chunk.bbox

        if left_bottom_origin and page_height is not None:
            bbox = _transform_bbox_dims(bbox, page_height)

        # Create the chunk content, including caption if available
        content = dots_chunk.text

        chunk_id = compute_mdhash_id(content, prefix="chunk-")

        chunk_obj = Item(
            documentKey=chunk_id,
            text=content,
            chunkIdx=tmp_chunk_idx,
            documentId=md_doc.documentKey,
            chunkType=chunk_type.value,
            pageNumber=dots_chunk.page_no,
            pageImageUrl=None,
            pageWidth=page_width,
            pageHeight=page_height,
            bbox=bbox,
            caption=dots_chunk.caption,
            # Terms using temporary idx, would be filled later after chunks created
            headers=None,
            children=None,
            # inherit file metadata
            type=md_doc.type,
            fileName=md_doc.fileName,
            uri=md_doc.uri,
            private=md_doc.private,
            uploadedAt=md_doc.uploadedAt,
            knowledgeBaseId=md_doc.knowledgeBaseId,
            workspaceId=md_doc.workspaceId,
        )

        chunk_id_mapping[tmp_chunk_idx] = chunk_id

        chunks.append(chunk_obj)

    chunks.sort(key=lambda c: c.chunkIdx)

    # For all headings & children, do a mapping
    for chunk in chunks:
        tmp_idx = chunk.chunkIdx
        raw_headers = dots_chunks[tmp_idx].headings if tmp_idx in dots_chunks else []
        if raw_headers:
            # Map the chunk ID to the heading text
            header_ids = [chunk_id_mapping[h] for h in raw_headers]
            chunk.headers = header_ids

        raw_children = dots_chunks[tmp_idx].children if tmp_idx in dots_chunks else []
        if raw_children:
            # Map the chunk ID to the children
            child_ids = [chunk_id_mapping[c] for c in raw_children]
            chunk.children = child_ids

    return chunks


def _fetch_header_texts_from_items(
    anchor_key: str, items: Optional[List[Item]]
) -> tuple[list[str], dict[str, str]]:
    """
    Get headers and their texts for the given anchor from items in memory.
    Returns (headers, {header_id: text}); returns ([], {}) if not found.
    """
    if not anchor_key or not items:
        return [], {}

    id2item = {it.documentKey: it for it in items if getattr(it, "documentKey", None)}
    anchor = id2item.get(anchor_key)
    if not anchor or not anchor.headers:
        return [], {}

    headers: List[str] = [h for h in anchor.headers if h in id2item]
    texts = {h: id2item[h].text for h in headers}
    return headers, texts


def chunk_dots_document_recursive(
    json_doc: List[Dict[str, Any]],
    md_doc: File,
    left_bottom_origin: bool = True,
    items: Optional[List[Item]] = None,
) -> List[Chunk]:
    """
    Split a dots document into chunks using DotsRecursiveChunker and return a list of Chunk objects.
    This produces aggregated chunks that may span multiple pages, with pageNumber and bbox aligned by page.
    """
    chunker = DotsRecursiveChunker()
    dense_chunks = chunker.chunk(json_doc)

    page_dimensions = _collect_dots_page_dimensions(json_doc)

    header_cache: Dict[str, tuple[list[str], dict[str, str]]] = {}

    chunks: List[Chunk] = []
    for dchunk in dense_chunks:
        chunk_type = _dots_category_to_chunk_type(dchunk.category)

        pages = dchunk.pages_span or []
        bbox_list = dchunk.bbox or []

        tbbox_list = bbox_list
        if left_bottom_origin:
            tbbox_list = _transform_bbox_dims_list(bbox_list, pages, page_dimensions)

        page_width = None
        page_height = None
        if pages:
            first_page_dim = page_dimensions.get(pages[0], {})
            page_width = first_page_dim.get("width", None)
            page_height = first_page_dim.get("height", None)

        # Get header texts from anchor_key and prepend them
        final_text = dchunk.text or ""
        headers_list: Optional[List[str]] = None
        anchor_key = getattr(dchunk, "anchor_key", None)
        if anchor_key:
            if anchor_key not in header_cache:
                header_cache[anchor_key] = _fetch_header_texts_from_items(
                    anchor_key, items
                )
            hdrs, hdr_texts = header_cache[anchor_key]
            if hdrs:
                headers_list = hdrs
                ordered_header_texts = [
                    hdr_texts.get(h, "") for h in hdrs if hdr_texts.get(h)
                ]
                if ordered_header_texts:
                    final_text = (
                        "\n".join(ordered_header_texts + [final_text])
                        if final_text
                        else "\n".join(ordered_header_texts)
                    )

        chunk_obj = Chunk(
            documentKey=compute_mdhash_id(final_text, prefix="chunk-"),
            text=final_text,
            chunkIdx=dchunk.chunk_idx,
            documentId=md_doc.documentKey,
            chunkType=chunk_type.value,
            pageNumber=pages if pages else None,
            pageImageUrl=None,
            pageWidth=page_width,
            pageHeight=page_height,
            bbox=tbbox_list if tbbox_list else None,
            caption=dchunk.caption,
            headers=headers_list,
            children=None,
            type=md_doc.type,
            fileName=md_doc.fileName,
            uri=md_doc.uri,
            private=md_doc.private,
            uploadedAt=md_doc.uploadedAt,
            knowledgeBaseId=md_doc.knowledgeBaseId,
            workspaceId=md_doc.workspaceId,
        )
        chunks.append(chunk_obj)

    chunks.sort(key=lambda c: c.chunkIdx)
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
    chunk_texts = text_splitter.split_text(langchain_doc.text)

    chunks = []
    for idx, chunk in enumerate(chunk_texts):
        chunk_obj = Chunk(
            documentKey=compute_mdhash_id(chunk, prefix="chunk-"),
            text=chunk,
            chunkIdx=idx,
            documentId=langchain_doc.documentKey,
            chunkType="text",
            # Inherit file metadata from doc_md
            type=langchain_doc.type,
            fileName=langchain_doc.fileName,
            pageNumber=langchain_doc.pageNumber,
            uri=langchain_doc.uri,
            private=langchain_doc.private,
            uploadedAt=langchain_doc.uploadedAt,
            # Fields required by ChunkMetadata but not applicable for langchain chunks
            pageImageUrl=None,
            pageWidth=None,
            pageHeight=None,
            bbox=None,
            caption=None,
            headers=None,
            knowledgeBaseId=langchain_doc.knowledgeBaseId,
            workspaceId=langchain_doc.workspaceId,
            children=None,
        )

        chunks.append(chunk_obj)

    return chunks
