import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from .client import PageIndexUtil, ProcessingResult, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class ParseResponse:
    """Response object for document parsing."""
    file_name: str
    status: str # pending, processing, completed, failed
    pages: List[Dict[str, Any]] # page: {index: int, markdown: string, blocks: List[Dict[str, Any]]}
                                # block: {id: string, title: string, text: string, page_index: int, parent_ids: List[str], hierarchy_level: int}
    markdown_document: str
    hierarchy: Dict[str, Any] # hierarchy: {blocks: List[Dict[str, Any]], table_of_contents: string}
                              # block: {id: string, title: string, text: string, page_index: int, parent_ids: List[str], hierarchy_level: int}
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ParseResponse to dictionary."""
        return {
            "file_name": self.file_name,
            "status": self.status,
            "pages": self.pages,
            "markdown_document": self.markdown_document,
            "hierarchy": self.hierarchy,
            "error": self.error
        }

class RemotePageIndex:
    """Remote Page Index operations for document processing and retrieval."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize remote Page Index client.
        
        Args:
            api_key: Page Index API key
        """
        self.util = PageIndexUtil(api_key)
    
    def process_pdf(self, file_name: str) -> ParseResponse:
        """
        Process a PDF file and return document ID.
        
        Args:
            file_name: Path and name to PDF file
            
        Returns:
            ParseResponse if successful, None otherwise
        """
        try:
            # Validate file
            if not Path(file_name).exists():
                logger.error(f"File not found: {file_name}")
                return None

            # Process PDF
            logger.info(f"Processing PDF: {file_name}")
            tree_result = self.util.process_document_complete(file_name, timeout=300)
            md_result = self.util.get_ocr_result(tree_result.doc_id, wait=True)

            if tree_result.status != "completed" or md_result.status != "completed":
                logger.error(f"Tree generation failed: {tree_result.error}")
                return ParseResponse(
                    file_name=file_name,
                    status=tree_result.status,
                    pages=[],
                    markdown_document="",
                    hierarchy={},
                    error=tree_result.error or "Tree generation failed"
                )
            
            # Convert tree result to ParseResponse format
            parse_response = self._convert_to_response(
                file_name, tree_result, md_result
            )

            return parse_response
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None
    
    def _convert_to_response(
        self, file_name: str, tree_result: ProcessingResult, md_result: ProcessingResult
    ) -> ParseResponse:
        """
        Convert processing results to ParseResponse format.

        Args:
            file_name: Name of the PDF file
            tree_result: Result from tree generation
            md_result: Result from OCR markdown generation

        Returns:
            ParseResponse object
        """
        # Extract OCR pages data
        res = ParseResponse(
            file_name=file_name,
            status=md_result.status,
            pages=[],
            markdown_document="",
            hierarchy={},
        )
        
        # Pages from the md result
        doc_md = ""
        if md_result.result:
            for page in md_result.result:
                res.pages.append({
                    "index": page.get("page_index", 0),
                    "markdown": page.get("markdown", ""),
                    "blocks": []
                })
                doc_md += page.get("markdown", "") + "\n\n"

        res.markdown_document = doc_md
        
        # Blocks from tree result - extract hierarchical structure
        blocks = []
        table_of_contents = ""
        
        if tree_result.result:
            blocks = self._extract_blocks_from_tree(tree_result.result)
            table_of_contents = self._generate_table_of_contents(tree_result.result)
            
            # Add blocks to their respective pages
            for block in blocks:
                page_idx = block.get("page_index", 1) - 1  # Convert to 0-based index
                if 0 <= page_idx < len(res.pages):
                    res.pages[page_idx]["blocks"].append(block)

        res.hierarchy = {
            "blocks": blocks,
            "table_of_contents": table_of_contents
        }

        return res
    
    def _extract_blocks_from_tree(self, tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract blocks from tree structure recursively.
        
        Args:
            tree_nodes: List of tree nodes from PageIndex API
            
        Returns:
            List of block dictionaries
        """
        blocks = []
        
        def process_node(node: Dict[str, Any], parent_ids: List[str] = None, level: int = 0):
            if parent_ids is None:
                parent_ids = []
                
            # Create block from current node
            block = {
                "id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "text": node.get("text", ""),
                "page_index": node.get("page_index", 1),
                "parent_ids": parent_ids.copy(),
                "hierarchy_level": level
            }
            blocks.append(block)
            
            # Process child nodes
            child_nodes = node.get("nodes", [])
            new_parent_ids = parent_ids + [node.get("node_id", "")]
            
            for child_node in child_nodes:
                process_node(child_node, new_parent_ids, level + 1)
        
        # Process all top-level nodes
        for node in tree_nodes:
            process_node(node)
            
        return blocks
    
    def _generate_table_of_contents(self, tree_nodes: List[Dict[str, Any]]) -> str:
        """
        Generate a table of contents from tree structure.
        
        Args:
            tree_nodes: List of tree nodes from PageIndex API
            
        Returns:
            Table of contents as markdown string
        """
        toc_lines = []
        
        def process_node_for_toc(node: Dict[str, Any], level: int = 0):
            indent = "  " * level
            title = node.get("title", "")
            page_index = node.get("page_index", 1)
            
            if title:
                toc_lines.append(f"{indent}- {title} (Page {page_index})")
            
            # Process child nodes
            child_nodes = node.get("nodes", [])
            for child_node in child_nodes:
                process_node_for_toc(child_node, level + 1)
        
        # Process all top-level nodes
        for node in tree_nodes:
            process_node_for_toc(node)
        
        return "\n".join(toc_lines)