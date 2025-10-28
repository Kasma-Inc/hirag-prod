from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from hirag_prod._utils import encode_string_by_tiktoken
from hirag_prod.schema.item import Item

ITEM_MERGE_TOKEN_THRESHOLD = 100


@dataclass
class DenseChunk:
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
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = None
    updated_by: str = None
    id: str = None
    extracted_timestamp: Optional[datetime] = None


class UnifiedRecursiveChunker:
    """Recursive chunker for Items obtained after OCR JSON documents (dense aggregation)."""

    def _is_table(self, category: str) -> bool:
        return category == "table"

    def _is_picture(self, category: str) -> bool:
        return category == "picture"

    def _process_page_bboxes(
        self, page_bboxes: List[List[float]], page: int
    ) -> List[float]:
        """Process bboxes for a single page and return aggregated bbox."""
        if not page_bboxes:
            return []

        bbox_len = [len(b) for b in page_bboxes if b]

        # Verify all bboxes have the same length
        if len(set(bbox_len)) > 1:
            raise ValueError(
                f"Inconsistent bbox lengths on page {page}: {set(bbox_len)}"
            )

        bbox_len = bbox_len[0] if bbox_len else 0

        if bbox_len == 4:
            x_0 = min(b[0] for b in page_bboxes)
            y_0 = max(b[1] for b in page_bboxes)
            x_1 = max(b[2] for b in page_bboxes)
            y_1 = min(b[3] for b in page_bboxes)
            return [x_0, y_0, x_1, y_1]
        elif bbox_len == 2:
            # Handle charspan style bbox [start, end]
            min_start = min(b[0] for b in page_bboxes)
            max_end = max(b[1] for b in page_bboxes)
            return [min_start, max_end]
        else:
            # Fallback: use empty bbox
            return []

    def _build_bbox_list_for_pages(
        self, items: List[Item], pages: List[int]
    ) -> Tuple[List[List[float]], List[int]]:
        """Build bbox list for given pages from items.
        Returns (bbox_list, expanded_pages) where expanded_pages duplicates pages for multi-column.
        """
        bbox_list: List[List[float]] = []
        expanded_pages: List[int] = []
        for page in pages:
            page_bboxes = [it.bbox for it in items if it.pageNumber == page]
            lengths = {len(b) for b in page_bboxes if b}

            # Split columns only when all bboxes are 4-length (x0,y0,x1,y1)
            if lengths == {4} and len(page_bboxes) > 1:
                groups: List[List[List[float]]] = []
                current: List[List[float]] = []
                prev_y1: Optional[float] = None

                for b in page_bboxes:
                    y1 = b[3]
                    if prev_y1 is not None and y1 > prev_y1:
                        # next bbox is in the next column; start a new group
                        if current:
                            groups.append(current)
                            current = []
                    current.append(b)
                    prev_y1 = y1

                if current:
                    groups.append(current)
            else:
                groups = [page_bboxes]

            for g in groups:
                agg = self._process_page_bboxes(g, page)
                bbox_list.append(agg)
                expanded_pages.append(page)

        return bbox_list, expanded_pages

    def _create_dense_chunk(
        self,
        chunk_idx: int,
        text: str,
        bbox_list: List[List[float]],
        pages_span: List[int],
        reference_item: Item,
    ) -> DenseChunk:
        """Create a DenseChunk with common fields from a reference item."""
        return DenseChunk(
            chunk_idx=chunk_idx,
            text=text,
            category="text",
            bbox=bbox_list,
            pages_span=pages_span,
            children=None,
            caption=None,
            headings=None,
            page_height=reference_item.pageHeight,
            page_width=reference_item.pageWidth,
            document_id=reference_item.documentId,
            document_type=reference_item.type,
            file_name=reference_item.fileName,
            uri=reference_item.uri,
            private=reference_item.private,
            knowledge_base_id=reference_item.knowledgeBaseId,
            workspace_id=reference_item.workspaceId,
            created_at=reference_item.createdAt,
            updated_at=reference_item.updatedAt,
            created_by=reference_item.createdBy,
            updated_by=reference_item.updatedBy,
            id=reference_item.id,
            extracted_timestamp=reference_item.extractedTimestamp,
        )

    def _build_separate_chunk(
        self, id2item: Dict[str, Item], item: Item, chunk_idx: int, category: str
    ) -> DenseChunk:
        headers = item.headers or []
        cap = item.caption or ""
        header_texts = [id2item[h].text for h in headers if h in id2item]

        if category == "picture":
            merged_caption = "\n".join([*header_texts, cap, item.text]).strip()
        elif category == "table":
            merged_caption = "\n".join([*header_texts, cap]).strip()
        else:
            merged_caption = None

        return DenseChunk(
            chunk_idx=chunk_idx,
            text=item.text,
            category=category,
            bbox=[item.bbox],
            pages_span=[item.pageNumber],
            children=None,
            caption=merged_caption,
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
            created_at=item.createdAt,
            updated_at=item.updatedAt,
            created_by=item.createdBy,
            updated_by=item.updatedBy,
            id=item.id,
            extracted_timestamp=item.extractedTimestamp,
        )

    def chunk(
        self, items: Optional[List[Item]], header_set: Optional[set[str]]
    ) -> List[DenseChunk]:
        if not items:
            return []

        header_set = header_set or set()

        id2item = {item.documentKey: item for item in items}

        chunks: List[DenseChunk] = []
        chunk_idx = 1
        i = 0
        while i < len(items):
            # First Loop: Handle three special cases:
            # 1. table or picture
            # 2. page_footer, page_header, footnote
            # 3. header items
            item = items[i]
            item_type = item.chunkType
            if self._is_table(item_type) or self._is_picture(item_type):
                merged_item = self._build_separate_chunk(
                    id2item=id2item, item=item, chunk_idx=chunk_idx, category=item_type
                )

                chunks.append(merged_item)
                chunk_idx += 1
                i += 1
                continue

            if item_type in ["page_footer", "page_header", "footnote"]:
                i += 1
                continue

            if item.documentKey in header_set:
                i += 1
                continue

            # Second Loop: Handle abundant text items until hitting a header item
            non_header_items = []
            accumulated_text_tokens = 0
            while i < len(items):
                cur_item = items[i]
                # Break if hitting a header item
                if cur_item.documentKey in header_set:
                    break
                # Handle table or picture items (add to chunks directly)
                elif self._is_table(cur_item.chunkType) or self._is_picture(
                    cur_item.chunkType
                ):
                    merged_item = self._build_separate_chunk(
                        id2item=id2item,
                        item=cur_item,
                        chunk_idx=chunk_idx,
                        category=cur_item.chunkType,
                    )
                    chunks.append(merged_item)
                    chunk_idx += 1
                    i += 1
                    continue
                else:
                    if cur_item.chunkType not in [
                        "page_footer",
                        "page_header",
                        "footnote",
                    ]:
                        item_text = cur_item.text or ""
                        non_header_items.append(cur_item)
                        # set a threshold to merge items into a single chunk
                        accumulated_text_tokens += len(
                            encode_string_by_tiktoken(item_text)
                        )
                        if accumulated_text_tokens >= ITEM_MERGE_TOKEN_THRESHOLD:
                            i += 1
                            break
                    i += 1

            if not non_header_items:
                continue

            first_non = non_header_items[0]
            header_ids = first_non.headers

            if not header_ids:  # the first non-header items block
                merged_text = "\n".join(n.text for n in non_header_items)
                non_header_pages = sorted(set(n.pageNumber for n in non_header_items))
                pages_span = [p for p in non_header_pages]
                bbox_list, pages_span = self._build_bbox_list_for_pages(
                    non_header_items, non_header_pages
                )

                merged_item = self._create_dense_chunk(
                    chunk_idx=chunk_idx,
                    text=merged_text,
                    bbox_list=bbox_list,
                    pages_span=pages_span,
                    reference_item=item,
                )

                chunks.append(merged_item)
                chunk_idx += 1
                continue

            header_items = [id2item[hid] for hid in header_ids if hid in id2item]
            non_header_texts = [no_h.text for no_h in non_header_items]
            header_texts = [h.text for h in header_items]
            merged_text = "\n".join(header_texts + non_header_texts)

            non_header_pages = sorted(set(n.pageNumber for n in non_header_items))

            non_header_bbox_list, pages_span = self._build_bbox_list_for_pages(
                non_header_items, non_header_pages
            )

            merged_item = self._create_dense_chunk(
                chunk_idx=chunk_idx,
                text=merged_text,
                bbox_list=non_header_bbox_list,
                pages_span=pages_span,
                reference_item=item,
            )

            chunks.append(merged_item)
            chunk_idx += 1

        return chunks
