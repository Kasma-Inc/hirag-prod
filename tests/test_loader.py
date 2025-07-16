import os
from typing import Any, Dict, Tuple

import pytest
from docling_core.types.doc import DoclingDocument

from hirag_prod.loader import load_document
from hirag_prod.schema import File

# Document types supported by Docling loader (excluding txt)
DOCLING_DOCUMENTS = {
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

# Document types supported by Langchain loader (primarily txt)
LANGCHAIN_DOCUMENTS = {
    "txt": {
        "filename": "test.txt",
        "content_type": "text/plain",
    },
}


# ================================ Test Docling Loader ================================
class TestDoclingLoader:
    """Test suite for Docling document loader with various file formats"""

    @pytest.fixture
    def test_files_dir(self) -> str:
        """Get the test files directory path"""
        return os.path.join(os.path.dirname(__file__), "test_files")

    def _create_document_meta(
        self, doc_type: str, filename: str, uri: str
    ) -> Dict[str, Any]:
        return {
            "type": doc_type,
            "filename": filename,
            "uri": uri,
            "private": False,
        }

    def _assert_docling_document_loaded(self, docling_doc: Any, doc_md: Any) -> None:
        """Assert that docling document was loaded successfully"""
        assert isinstance(docling_doc, DoclingDocument)
        assert isinstance(doc_md, File)
        assert doc_md.page_content is not None
        assert doc_md.metadata is not None
        assert doc_md.id.startswith("doc-")

    def _load_and_assert_document(
        self, doc_type: str, test_files_dir: str
    ) -> Tuple[Any, Any]:
        """Load a document with docling loader and assert it was loaded correctly"""
        config = DOCLING_DOCUMENTS[doc_type]
        document_path = os.path.join(test_files_dir, config["filename"])

        document_meta = self._create_document_meta(
            doc_type=doc_type, filename=config["filename"], uri=document_path
        )

        docling_doc, doc_md = load_document(
            document_path=document_path,
            content_type=config["content_type"],
            document_meta=document_meta,
            loader_configs=None,
            loader_type="docling",
        )

        self._assert_docling_document_loaded(docling_doc, doc_md)
        return docling_doc, doc_md

    @pytest.mark.parametrize("doc_type", DOCLING_DOCUMENTS.keys())
    def test_load_document_docling(self, doc_type: str, test_files_dir: str):
        """Test loading various document types with Docling loader"""
        self._load_and_assert_document(doc_type, test_files_dir)


# ================================ Test Langchain Loader ================================
class TestLangchainLoader:
    """Test suite for Langchain document loader with text files"""

    @pytest.fixture
    def test_files_dir(self) -> str:
        """Get the test files directory path"""
        return os.path.join(os.path.dirname(__file__), "test_files")

    def _create_document_meta(
        self, doc_type: str, filename: str, uri: str
    ) -> Dict[str, Any]:
        """Create document metadata dictionary"""
        return {
            "type": doc_type,
            "filename": filename,
            "uri": uri,
            "private": False,
        }

    def _assert_langchain_document_loaded(self, langchain_doc: Any) -> None:
        """Assert that langchain document was loaded successfully"""
        assert isinstance(langchain_doc, File)
        assert langchain_doc.page_content is not None
        assert langchain_doc.id.startswith("doc-")
        assert langchain_doc.metadata is not None
        assert langchain_doc.metadata.type == "txt"
        assert langchain_doc.metadata.filename == "test.txt"

    def _load_and_assert_document(self, doc_type: str, test_files_dir: str) -> Any:
        """Load a document with langchain loader and assert it was loaded correctly"""
        config = LANGCHAIN_DOCUMENTS[doc_type]
        document_path = os.path.join(test_files_dir, config["filename"])

        document_meta = self._create_document_meta(
            doc_type=doc_type, filename=config["filename"], uri=document_path
        )

        langchain_doc = load_document(
            document_path=document_path,
            content_type=config["content_type"],
            document_meta=document_meta,
            loader_configs=None,
            loader_type="langchain",
        )

        self._assert_langchain_document_loaded(langchain_doc)
        return langchain_doc

    @pytest.mark.parametrize("doc_type", LANGCHAIN_DOCUMENTS.keys())
    def test_load_document_langchain(self, doc_type: str, test_files_dir: str):
        """Test loading text files with Langchain loader"""
        self._load_and_assert_document(doc_type, test_files_dir)


# ================================ Test OCR Loader ================================
class TestOCRLoader:
    """Test suite for OCR document loader with PDF files"""

    def test_load_pdf_ocr(self):
        """Test loading PDF with OCR loader"""
        # TODO: s3 operation is not supported yet
