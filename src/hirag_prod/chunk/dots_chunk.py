# This is a hierachical and recursive chunker for the JSON by Dots OCR

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from hirag_prod.schema.item import Item


class DotsChunkType(Enum):
    """Chunk types for Dots OCR documents."""

    TITLE = "Title"
    SECTION_HEADER = "Section-header"
    TEXT = "Text"
    TABLE = "Table"
    LIST_ITEM = "List-item"
    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    PICTURE = "Picture"
    PAGE_HEADER = "Page-header"
    PAGE_FOOTER = "Page-footer"


@dataclass
class DotsChunk:
    """A chunk from Dots OCR processing."""

    chunk_idx: int
    text: str
    category: str
    page_no: int
    bbox: List[float]
    headings: List[int]  # Hierarchical context
    caption: Optional[str] = None
    children: Optional[List[int]] = None


class DotsHierarchicalChunker:
    """Hierarchical chunker for Dots OCR JSON documents."""

    MAX_LEVEL = 6  # Maximum heading level to consider

    def __init__(self):
        self.hierarchy_types = [DotsChunkType.TITLE, DotsChunkType.SECTION_HEADER]

    def _get_level(self, text: str) -> int:
        """Get the heading level from the text."""
        level = 0
        stripped_text = text.lstrip()
        while stripped_text.startswith("#"):
            level += 1
            stripped_text = stripped_text[1:].lstrip()

        # If no # found, treat as level 1 (basic section header)
        if level == 0:
            return 1
        # Cap at maximum level
        if level > self.MAX_LEVEL:
            return self.MAX_LEVEL
        return level

    def chunk(self, json_doc: List[Dict[str, Any]]) -> Dict[int, DotsChunk]:
        """
        Chunk a Dots OCR document while maintaining hierarchical context.

        Args:
            json_doc: List of pages with layout information

        Yields:
            DotsChunk: Chunks with hierarchical context
        """
        heading_by_level: Dict[int, int] = {}
        used_captions: set = set()
        sorted_boxes: List[Dict[str, Any]] = []
        header_boxes: Dict[int, Dict[str, Any]] = {}
        parsed_chunks: Dict[int, DotsChunk] = {}

        # Collect all boxes sorted by order
        for page in json_doc:
            page_no = page.get("page_no", 0)
            layout_info = page.get("full_layout_info", [])

            for box in layout_info:
                # Add the box to the sorted boxes
                box["page_no"] = page_no
                box["idx"] = len(sorted_boxes)  # Assign a box idx for simplicity
                sorted_boxes.append(box)

        # Chunk by hierachy & handle captions for tables & images
        for box in sorted_boxes:
            idx = box.get("idx", -1)

            text = box.get("text", "").strip()
            if not text:
                continue

            category = box.get("category", "Text")
            bbox = box.get("bbox", [])
            page_no = box.get("page_no", 0)

            def _get_caption() -> Optional[Any]:
                """Extract caption from surrounding boxes."""
                previous_box = sorted_boxes[idx - 1] if idx > 0 else None
                next_box = (
                    sorted_boxes[idx + 1] if idx < len(sorted_boxes) - 1 else None
                )

                # TODO: Use LLM as judgement for caption extraction

                # Now using simple heuristics
                # Logic: 1. If previous is caption and not used, more likely the caption for this box
                if (
                    previous_box
                    and previous_box.get("category") == DotsChunkType.CAPTION.value
                    and (idx - 1) not in used_captions
                ):
                    used_captions.add(idx - 1)
                    return previous_box
                # Logic: 2. If next is caption and not used, use caption for this box
                elif (
                    next_box
                    and next_box.get("category") == DotsChunkType.CAPTION.value
                    and (idx + 1) not in used_captions
                ):
                    used_captions.add(idx + 1)
                    return next_box
                return None

            def _get_headers_and_register() -> List[int]:
                """Register headers in the chunk."""
                if not header_boxes:
                    return []

                heading_ids = [
                    heading_by_level[k] for k in sorted(heading_by_level.keys())
                ]

                # Add self to the children of its direct parent
                if heading_by_level:
                    # Get the most recent (deepest) heading
                    deepest_level = max(heading_by_level.keys())
                    parent_idx = heading_by_level[deepest_level]
                    if parent_idx in header_boxes:
                        header_boxes[parent_idx]["children"].append(idx)

                return heading_ids

            caption_block = None

            # Heading hierarchy, don't parse immediately
            if category in [
                DotsChunkType.TITLE.value,
                DotsChunkType.SECTION_HEADER.value,
            ]:
                level = 0  # Default to highest priority for titles

                if category == DotsChunkType.SECTION_HEADER.value:
                    # Section header, level is number of #s at the beginning
                    level = self._get_level(text)

                # Remove all deeper and same level headings
                keys_to_del = [k for k in heading_by_level if k >= level]
                for k in keys_to_del:
                    heading_by_level.pop(k, None)

                heading_ids = _get_headers_and_register()

                header_boxes[idx] = {
                    "text": text,
                    "level": level,
                    "bbox": bbox,
                    "page_no": page_no,
                    "headers": heading_ids,
                    "children": [],
                }

                # Update heading hierarchy
                heading_by_level[level] = idx

                continue

            # Raw Text and List Items
            elif category in [DotsChunkType.TEXT.value, DotsChunkType.LIST_ITEM.value]:
                pass

            # Table with potential caption
            elif category == DotsChunkType.TABLE.value:
                caption_block = _get_caption()

            # Picture with potential caption
            elif category == DotsChunkType.PICTURE.value:
                caption_block = _get_caption()

            # Formula content
            elif category == DotsChunkType.FORMULA.value:
                caption_block = _get_caption()

            # Footnote content
            elif category == DotsChunkType.FOOTNOTE.value:
                pass

            # Page headers and footers (usually skip or handle differently)
            elif category in [
                DotsChunkType.PAGE_HEADER.value,
                DotsChunkType.PAGE_FOOTER.value,
            ]:
                pass

            # Handle unknown categories as text
            else:
                pass

            # Get current heading hierarchy as list of heading texts
            heading_ids = _get_headers_and_register()

            if caption_block:
                # If caption is found, add it to the bbox list
                caption = caption_block.get("text", None)
            else:
                caption = None

            chunk = DotsChunk(
                chunk_idx=idx,
                text=text,
                category=category,
                page_no=page_no,
                bbox=bbox,
                headings=heading_ids,
                caption=caption,
            )

            parsed_chunks[idx] = chunk

        # Add headers to parsed_chunks
        for header_idx, header_info in header_boxes.items():
            # Get heading hierarchy up to this header's level
            category = (
                DotsChunkType.SECTION_HEADER.value
                if header_info["level"] > 0
                else DotsChunkType.TITLE.value
            )

            if header_info["children"] == []:
                category = DotsChunkType.TEXT.value

            chunk = DotsChunk(
                chunk_idx=header_idx,
                text=header_info["text"],
                category=category,
                page_no=header_info["page_no"],
                bbox=header_info["bbox"],
                headings=header_info["headers"],
                caption=None,
                children=header_info["children"],
            )

            parsed_chunks[header_idx] = chunk

        return parsed_chunks


@dataclass
class DotsDenseChunk:
    """Dense/recursive chunk schema aligned to this module's style."""

    chunk_idx: int = None
    text: str = None
    category: str = None
    bbox: List[List[float]] = None
    headings: List[int] = None
    caption: Optional[str] = None
    children: Optional[List[int]] = None
    pages_span: List[int] = None
    page_height: float = None
    page_width: float = None
    document_id: str = None
    document_type: str = None
    file_name: str = None
    uri: str = None
    private: bool = None
    knowledge_base_id: str = None
    workspace_id: str = None
    uploaded_at: datetime = None


class DotsRecursiveChunker:
    """Recursive chunker for Dots OCR JSON documents (dense aggregation)."""

    def _is_table(self, category: str) -> bool:
        return category == "table"

    def _build_table_chunk(self, item: Item, chunk_idx: int) -> DotsDenseChunk:
        return DotsDenseChunk(
            chunk_idx=chunk_idx,
            text=item.text,
            category="table",
            bbox=[item.bbox],
            pages_span=[item.pageNumber],
            children=None,
            caption=item.caption,
            headings=None,
            page_height=item.pageHeight,
            page_width=item.pageWidth,
            document_id=item.documentId,
            document_type=item.type,
            file_name=item.fileName,
            uri=item.uri,
            private=item.private,
            knowledge_base_id=item.knowledgeBaseId,
            workspace_id=item.workspaceId,
            uploaded_at=item.uploadedAt,
        )

    def chunk(
        self, items: Optional[List[Item]], header_set: Optional[set[str]]
    ) -> List[DotsDenseChunk]:
        if not items:
            return []

        header_set = header_set or set()

        id2item = {item.documentKey: item for item in items}

        chunks: List[DotsDenseChunk] = []
        chunk_idx = 1
        i = 0
        while i < len(items):
            item = items[i]
            item_type = item.chunkType

            if self._is_table(item_type):
                merged_item = self._build_table_chunk(item, chunk_idx)

                chunks.append(merged_item)
                chunk_idx += 1
                i += 1
                continue

            if item.documentKey in header_set:
                i += 1
                continue

            non_header_items = []
            while i < len(items):
                cur_item = items[i]
                if cur_item.documentKey in header_set:
                    break
                elif self._is_table(cur_item.chunkType):
                    merged_item = self._build_table_chunk(cur_item, chunk_idx)
                    chunks.append(merged_item)
                    chunk_idx += 1
                    i += 1
                    continue
                else:
                    non_header_items.append(cur_item)
                    i += 1

            if not non_header_items:
                continue

            first_non = non_header_items[0]
            header_ids = first_non.headers

            if not header_ids:  # the first non-header items block
                merged_text = " ".join(n.text for n in non_header_items)
                non_header_pages = sorted(set(n.pageNumber for n in non_header_items))
                pages_span = [p for p in non_header_pages]
                bbox_list = []
                for page in non_header_pages:
                    page_bboxes = [
                        n.bbox for n in non_header_items if n.pageNumber == page
                    ]
                    if page_bboxes:
                        min_x = min(b[0] for b in page_bboxes)
                        min_y = min(b[1] for b in page_bboxes)
                        max_x = max(b[2] for b in page_bboxes)
                        max_y = max(b[3] for b in page_bboxes)
                        bbox_list.append([min_x, min_y, max_x, max_y])

                merged_item = DotsDenseChunk(
                    chunk_idx=chunk_idx,
                    text=merged_text,
                    category="text",
                    bbox=bbox_list,
                    pages_span=pages_span,
                    children=None,
                    caption=None,
                    headings=None,
                    page_height=item.pageHeight,
                    page_width=item.pageWidth,
                    document_id=item.documentId,
                    document_type=item.type,
                    file_name=item.fileName,
                    uri=item.uri,
                    private=item.private,
                    knowledge_base_id=item.knowledgeBaseId,
                    workspace_id=item.workspaceId,
                    uploaded_at=item.uploadedAt,
                )

                chunks.append(merged_item)
                chunk_idx += 1
                i += 1
                continue

            header_items = [id2item[hid] for hid in header_ids if hid in id2item]
            non_header_texts = [h.text for h in non_header_items]
            header_texts = [h.text for h in header_items]
            merged_text = " ".join(header_texts + non_header_texts)

            pages_span = [h.pageNumber for h in header_items]
            non_header_pages = sorted(set(n.pageNumber for n in non_header_items))
            pages_span.extend(p for p in non_header_pages)

            bbox_list = [h.bbox for h in header_items]
            for page in non_header_pages:
                page_bboxes = [n.bbox for n in non_header_items if n.pageNumber == page]
                if page_bboxes:
                    min_x = min(b[0] for b in page_bboxes)
                    min_y = min(b[1] for b in page_bboxes)
                    max_x = max(b[2] for b in page_bboxes)
                    max_y = max(b[3] for b in page_bboxes)
                    bbox_list.append([min_x, min_y, max_x, max_y])

            merged_item = DotsDenseChunk(
                chunk_idx=chunk_idx,
                text=merged_text,
                category="text",
                bbox=bbox_list,
                pages_span=pages_span,
                children=None,
                caption=None,
                headings=None,
                page_height=item.pageHeight,
                page_width=item.pageWidth,
                document_id=item.documentId,
                document_type=item.type,
                file_name=item.fileName,
                uri=item.uri,
                private=item.private,
                knowledge_base_id=item.knowledgeBaseId,
                workspace_id=item.workspaceId,
                uploaded_at=item.uploadedAt,
            )

            chunks.append(merged_item)
            chunk_idx += 1

        return chunks
