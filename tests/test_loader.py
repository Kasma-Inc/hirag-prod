import os
from typing import Any, Dict, Tuple

import pytest
from docling_core.types.doc import DoclingDocument

from hirag_prod.loader import load_document
from hirag_prod.schema import File

# Test data configuration for different document types
TEST_DOCUMENTS = {
    "docx": {
        "filename": "word_sample.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    "csv": {
        "filename": "csv-comma.csv",
        "content_type": "text/csv",
    },
    "xlsx": {
        "filename": "sample_sales_data.xlsm",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "html": {
        "filename": "wiki_labubu.html",
        "content_type": "text/html",
    },
    "pptx": {
        "filename": "Beamer.pptx",
        "content_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    },
    "md": {
        "filename": "fresh_wiki_article.md",
        "content_type": "text/markdown",
    },
    "pdf": {
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "content_type": "application/pdf",
    },
}


class TestDoclingLoader:
    """Test suite for Docling document loader with various file formats"""

    @pytest.fixture
    def test_files_dir(self) -> str:
        """Get the test files directory path"""
        return os.path.join(os.path.dirname(__file__), "test_files")

    def _create_document_meta(
        self, doc_type: str, filename: str, uri: str
    ) -> Dict[str, Any]:
        """
        Create document metadata dictionary

        Args:
            doc_type: Document type (e.g., 'pdf', 'docx')
            filename: Name of the file
            uri: Full path to the document

        Returns:
            Document metadata dictionary
        """
        return {
            "type": doc_type,
            "filename": filename,
            "uri": uri,
            "private": False,
        }

    def _assert_document_loaded(self, docling_doc: Any, doc_md: Any) -> None:
        """
        Assert that document was loaded successfully

        Args:
            docling_doc: Docling document instance
            doc_md: File metadata instance
        """
        assert isinstance(docling_doc, DoclingDocument)
        assert isinstance(doc_md, File)
        assert doc_md.page_content is not None
        assert doc_md.metadata is not None
        assert doc_md.id.startswith("doc-")

    def _load_and_assert_document(
        self, doc_type: str, test_files_dir: str
    ) -> Tuple[Any, Any]:
        """
        Load a document and assert it was loaded correctly

        Args:
            doc_type: Document type to load
            test_files_dir: Directory containing test files

        Returns:
            Tuple of (docling_doc, doc_md)
        """
        config = TEST_DOCUMENTS[doc_type]
        document_path = os.path.join(test_files_dir, config["filename"])

        document_meta = self._create_document_meta(
            doc_type=doc_type, filename=config["filename"], uri=document_path
        )

        # Load the document
        docling_doc, doc_md = load_document(
            document_path=document_path,
            content_type=config["content_type"],
            document_meta=document_meta,
            loader_configs=None,
            loader_type="docling",
        )

        # Assert document loaded successfully
        self._assert_document_loaded(docling_doc, doc_md)

        return docling_doc, doc_md

    # Individual test methods for each document type
    @pytest.mark.parametrize("doc_type", TEST_DOCUMENTS.keys())
    def test_load_document_docling(self, doc_type: str, test_files_dir: str):
        """Test loading various document types with Docling loader"""
        self._load_and_assert_document(doc_type, test_files_dir)


# ================================ OCR ================================
def test_load_pdf_ocr():
    # TODO: s3 operation is not supported yet
    pass
