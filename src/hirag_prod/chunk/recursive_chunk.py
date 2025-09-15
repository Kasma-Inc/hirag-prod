from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from hirag_prod.schema.item import Item


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
    uploaded_at: datetime = None


class UnifiedRecursiveChunker:
    """Recursive chunker for Items obtained after OCR JSON documents (dense aggregation)."""

    def _is_table(self, category: str) -> bool:
        return category == "table"

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
            min_x = min(b[0] for b in page_bboxes)
            min_y = min(b[1] for b in page_bboxes)
            max_x = max(b[2] for b in page_bboxes)
            max_y = max(b[3] for b in page_bboxes)
            return [min_x, min_y, max_x, max_y]
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
    ) -> List[List[float]]:
        """Build bbox list for given pages from items."""
        bbox_list = []
        for page in pages:
            page_bboxes = [item.bbox for item in items if item.pageNumber == page]
            bbox_list.append(self._process_page_bboxes(page_bboxes, page))
        return bbox_list

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
            uploaded_at=reference_item.uploadedAt,
        )

    def _build_table_chunk(self, item: Item, chunk_idx: int) -> DenseChunk:
        return DenseChunk(
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
    ) -> List[DenseChunk]:
        if not items:
            return []

        header_set = header_set or set()

        id2item = {item.documentKey: item for item in items}

        chunks: List[DenseChunk] = []
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
                bbox_list = self._build_bbox_list_for_pages(
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
            non_header_bbox_list = self._build_bbox_list_for_pages(
                non_header_items, non_header_pages
            )
            bbox_list.extend(non_header_bbox_list)

            merged_item = self._create_dense_chunk(
                chunk_idx=chunk_idx,
                text=merged_text,
                bbox_list=bbox_list,
                pages_span=pages_span,
                reference_item=item,
            )

            chunks.append(merged_item)
            chunk_idx += 1

        return chunks
