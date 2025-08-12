import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from pageindex import PageIndexClient
except ImportError:
    PageIndexClient = None


@dataclass
class ProcessingResult:
    """Result from Page Index processing operations."""
    doc_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from Page Index retrieval operations."""
    retrieval_id: str
    doc_id: str
    status: str
    query: str
    retrieved_nodes: Optional[List[Dict]] = None
    error: Optional[str] = None


class PageIndexUtil:
    """Simple utility for Page Index operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Page Index client.
        
        Args:
            api_key: Page Index API key. If None, will try to get from environment
        """
        if PageIndexClient is None:
            raise ImportError(
                "PageIndex SDK not installed. Run: pip install pageindex"
            )
        
        self.api_key = api_key or os.getenv("PAGEINDEX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Page Index API key required. Set PAGEINDEX_API_KEY environment "
                "variable or pass api_key parameter"
            )
        
        self.client = PageIndexClient(api_key=self.api_key)
    
    def submit_document(self, file_path: str) -> str:
        """
        Submit a document for processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document ID for tracking
        """
        result = self.client.submit_document(file_path)
        return result["doc_id"]
    
    def get_ocr_result(self, doc_id: str, wait: bool = False, 
                      timeout: int = 300) -> ProcessingResult:
        """
        Get OCR processing results.
        
        Args:
            doc_id: Document ID
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            ProcessingResult with OCR data
        """
        if wait:
            return self._wait_for_completion(
                doc_id, self.client.get_ocr, timeout
            )
        
        try:
            result = self.client.get_ocr(doc_id)
            return ProcessingResult(
                doc_id=doc_id,
                status=result.get("status", "unknown"),
                result=result.get("result") if result.get("status") == "completed" else None
            )
        except Exception as e:
            return ProcessingResult(
                doc_id=doc_id,
                status="error",
                error=str(e)
            )
    
    def get_tree_result(self, doc_id: str, wait: bool = False, 
                       timeout: int = 300) -> ProcessingResult:
        """
        Get tree generation results.
        
        Args:
            doc_id: Document ID
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            ProcessingResult with tree data
        """
        if wait:
            return self._wait_for_completion(
                doc_id, self.client.get_tree, timeout
            )
        
        try:
            result = self.client.get_tree(doc_id)
            return ProcessingResult(
                doc_id=doc_id,
                status=result.get("status", "unknown"),
                result=result.get("result") if result.get("status") == "completed" else None
            )
        except Exception as e:
            return ProcessingResult(
                doc_id=doc_id,
                status="error",
                error=str(e)
            )
    
    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if document is ready for retrieval.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if ready for retrieval
        """
        try:
            return self.client.is_retrieval_ready(doc_id)
        except Exception:
            return False
    
    def submit_query(self, doc_id: str, query: str, 
                    thinking: bool = False) -> str:
        """
        Submit a retrieval query.
        
        Args:
            doc_id: Document ID
            query: Question or information need
            thinking: Whether to use deeper retrieval
            
        Returns:
            Retrieval ID for tracking
        """
        result = self.client.submit_retrieval_query(
            doc_id=doc_id,
            query=query,
            thinking=thinking
        )
        return result["retrieval_id"]
    
    def get_retrieval_result(self, retrieval_id: str, wait: bool = False,
                           timeout: int = 120) -> RetrievalResult:
        """
        Get retrieval results.
        
        Args:
            retrieval_id: Retrieval ID
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            RetrievalResult with retrieved content
        """
        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result = self.client.get_retrieval_result(retrieval_id)
                    if result.get("status") == "completed":
                        return RetrievalResult(
                            retrieval_id=retrieval_id,
                            doc_id=result.get("doc_id", ""),
                            status="completed",
                            query=result.get("query", ""),
                            retrieved_nodes=result.get("retrieved_nodes")
                        )
                    elif result.get("status") == "failed":
                        return RetrievalResult(
                            retrieval_id=retrieval_id,
                            doc_id=result.get("doc_id", ""),
                            status="failed",
                            query=result.get("query", ""),
                            error="Retrieval failed"
                        )
                    
                    time.sleep(2)
                except Exception as e:
                    return RetrievalResult(
                        retrieval_id=retrieval_id,
                        doc_id="",
                        status="error",
                        query="",
                        error=str(e)
                    )
            
            return RetrievalResult(
                retrieval_id=retrieval_id,
                doc_id="",
                status="timeout",
                query="",
                error=f"Timeout after {timeout} seconds"
            )
        
        try:
            result = self.client.get_retrieval_result(retrieval_id)
            return RetrievalResult(
                retrieval_id=retrieval_id,
                doc_id=result.get("doc_id", ""),
                status=result.get("status", "unknown"),
                query=result.get("query", ""),
                retrieved_nodes=result.get("retrieved_nodes") if result.get("status") == "completed" else None
            )
        except Exception as e:
            return RetrievalResult(
                retrieval_id=retrieval_id,
                doc_id="",
                status="error",
                query="",
                error=str(e)
            )
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_document(doc_id)
            return True
        except Exception:
            return False
    
    def process_document_complete(self, file_path: str, 
                                timeout: int = 300) -> ProcessingResult:
        """
        Process a document completely (OCR + Tree).
        
        Args:
            file_path: Path to PDF file
            timeout: Maximum wait time in seconds
            
        Returns:
            ProcessingResult with tree data
        """
        # Submit document
        doc_id = self.submit_document(file_path)
        
        # Wait for tree completion (tree depends on OCR)
        return self.get_tree_result(doc_id, wait=True, timeout=timeout)
    
    def query_document(self, doc_id: str, query: str, 
                      thinking: bool = False, timeout: int = 120) -> RetrievalResult:
        """
        Query a processed document.
        
        Args:
            doc_id: Document ID
            query: Question or information need
            thinking: Whether to use deeper retrieval
            timeout: Maximum wait time in seconds
            
        Returns:
            RetrievalResult with retrieved content
        """
        if not self.is_retrieval_ready(doc_id):
            return RetrievalResult(
                retrieval_id="",
                doc_id=doc_id,
                status="not_ready",
                query=query,
                error="Document not ready for retrieval"
            )
        
        retrieval_id = self.submit_query(doc_id, query, thinking)
        return self.get_retrieval_result(retrieval_id, wait=True, timeout=timeout)
    
    def _wait_for_completion(self, doc_id: str, get_func, timeout: int) -> ProcessingResult:
        """Wait for processing completion."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = get_func(doc_id)
                status = result.get("status", "unknown")
                
                if status == "completed":
                    return ProcessingResult(
                        doc_id=doc_id,
                        status="completed",
                        result=result.get("result")
                    )
                elif status == "failed":
                    return ProcessingResult(
                        doc_id=doc_id,
                        status="failed",
                        error="Processing failed"
                    )
                
                time.sleep(3)
            except Exception as e:
                return ProcessingResult(
                    doc_id=doc_id,
                    status="error",
                    error=str(e)
                )
        
        return ProcessingResult(
            doc_id=doc_id,
            status="timeout",
            error=f"Timeout after {timeout} seconds"
        )
