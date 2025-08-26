# This is a hierachical chunker for the JSON by Dots OCR

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


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
    bbox: List[List[float]]
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

            bbox_list = [bbox]

            if caption_block:
                # If caption is found, add it to the bbox list
                bbox_list.append(caption_block.get("bbox", []))
                caption = caption_block.get("text", None)
            else:
                caption = None

            chunk = DotsChunk(
                chunk_idx=idx,
                text=text,
                category=category,
                page_no=page_no,
                bbox=bbox_list,
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
                bbox=[header_info["bbox"]],
                headings=header_info["headers"],
                caption=None,
                children=header_info["children"],
            )

            parsed_chunks[header_idx] = chunk

        return parsed_chunks
