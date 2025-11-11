from typing import Dict, List, Optional, Tuple

from hirag_prod._utils import encode_string_by_tiktoken
from hirag_prod.schema.chunk import DenseChunk
from hirag_prod.schema.item import Item

ITEM_MERGE_TOKEN_THRESHOLD = 100


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

    def _create_chunk_from_items(
        self,
        non_header_items: List[Item],
        header_ids: Optional[set[str]],
        id2item: Dict[str, Item],
        chunk_idx: int,
        reference_item: Item,
    ) -> DenseChunk:
        """Create a chunk from a list of items with the same headers."""
        if not non_header_items:
            raise ValueError("Cannot create chunk from empty items list")

        if not header_ids:  # No headers for this group
            merged_text = "\n".join(n.text for n in non_header_items)
            non_header_pages = sorted(set(n.pageNumber for n in non_header_items))
            bbox_list, pages_span = self._build_bbox_list_for_pages(
                non_header_items, non_header_pages
            )

            chunk = self._create_dense_chunk(
                chunk_idx=chunk_idx,
                text=merged_text,
                bbox_list=bbox_list,
                pages_span=pages_span,
                reference_item=reference_item,
            )
            chunk.headings = None
            return chunk

        # Has headers - prepend header texts and convert set to list
        header_items = [id2item[hid] for hid in header_ids if hid in id2item]
        non_header_texts = [no_h.text for no_h in non_header_items]
        header_texts = [h.text for h in header_items]
        merged_text = "\n".join(header_texts + non_header_texts)

        non_header_pages = sorted(set(n.pageNumber for n in non_header_items))
        bbox_list, pages_span = self._build_bbox_list_for_pages(
            non_header_items, non_header_pages
        )

        chunk = self._create_dense_chunk(
            chunk_idx=chunk_idx,
            text=merged_text,
            bbox_list=bbox_list,
            pages_span=pages_span,
            reference_item=reference_item,
        )
        chunk.headings = list(header_ids)
        return chunk

    def _build_separate_chunk(
        self, id2item: Dict[str, Item], item: Item, chunk_idx: int, category: str
    ) -> DenseChunk:
        # item.headers is already a list from Item schema
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
            headings=headers if headers else None,  # Store as list of item IDs
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

            # Second Loop: Handle abundant text items, grouping by same headers
            non_header_items = []
            accumulated_text_tokens = 0
            current_header_ids = set()

            while i < len(items):
                cur_item = items[i]
                # Break if hitting a header item
                if cur_item.documentKey in header_set:
                    break
                # Handle table or picture items (add to chunks directly)
                elif self._is_table(cur_item.chunkType) or self._is_picture(
                    cur_item.chunkType
                ):
                    # Before processing table/picture, flush any accumulated non-header items
                    if non_header_items:
                        chunk = self._create_chunk_from_items(
                            non_header_items,
                            current_header_ids,
                            id2item,
                            chunk_idx,
                            item,
                        )
                        chunks.append(chunk)
                        chunk_idx += 1
                        non_header_items = []
                        accumulated_text_tokens = 0
                        current_header_ids = set()

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
                        # Check if headers are different from current group
                        item_header_ids = (
                            set(cur_item.headers) if cur_item.headers else None
                        )

                        if not current_header_ids:
                            # First item in the group
                            current_header_ids = item_header_ids
                        elif current_header_ids != item_header_ids:
                            # Headers changed - create chunk for previous group
                            if non_header_items:
                                chunk = self._create_chunk_from_items(
                                    non_header_items,
                                    current_header_ids,
                                    id2item,
                                    chunk_idx,
                                    item,
                                )
                                chunks.append(chunk)
                                chunk_idx += 1
                                non_header_items = []
                                accumulated_text_tokens = 0
                            current_header_ids = item_header_ids

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

            # Flush remaining non_header_items
            if non_header_items:
                chunk = self._create_chunk_from_items(
                    non_header_items, current_header_ids, id2item, chunk_idx, item
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks
