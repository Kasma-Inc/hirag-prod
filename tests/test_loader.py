import os
from typing import Any, Dict, Optional, Tuple

import pytest
from docling_core.types.doc import DoclingDocument

from hirag_prod._utils import upload_file_to_s3
from hirag_prod.loader import check_dots_ocr_health, load_document
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
        return os.path.join("file://", os.path.dirname(__file__), "test_files")

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

        _, langchain_doc = load_document(
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


# ================================ Test Dots OCR Loader ================================
class TestDotsOCRLoader:

    def test_health_check(self):
        """Test Dots OCR health check"""
        assert check_dots_ocr_health()

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

    def _assert_dots_ocr_document_loaded(self, doc: Any, doc_md: Any) -> None:
        """Assert that Dots OCR document was loaded successfully"""
        assert isinstance(doc, list)
        assert isinstance(doc_md, File)
        assert doc_md.page_content is not None
        assert doc_md.metadata is not None
        assert doc_md.id.startswith("doc-")

    def load_document_info(
        self, options: str, dir: Optional[str], file_name: Optional[str]
    ) -> Tuple[str, str]:
        if options == "s3":
            return "s3://monkeyocr/test/input/test_pdf/small.pdf", "small.pdf"

        if options == "local":
            if not dir or not file_name:
                return "", ""

            local_path = os.path.join(dir, file_name)
            # test if local exists
            if not os.path.exists(local_path):
                return "", ""

            s3_path = f"test/input/test_pdf/{file_name}"
            print(f"Uploading {local_path} to {s3_path}")
            upload_file_to_s3(local_path, s3_path)
            s3_path = f"s3://monkeyocr/test/input/test_pdf/{file_name}"
            return s3_path, file_name

    def test_load_pdf_dots_ocr_s3(self):
        """Test loading PDF with Dots OCR loader from S3"""
        s3_path, filename = self.load_document_info(
            "local", "/chatbot/tests/test_files/", "PGhandbook2025.pdf"
        )
        if not s3_path:
            print("Failed to load document from S3")
            return

        print(f"Loading document from: {s3_path}")

        document_meta = self._create_document_meta(
            doc_type="pdf", filename=filename, uri=s3_path
        )

        json_doc, md_doc = load_document(
            document_path=s3_path,
            content_type="application/pdf",
            document_meta=document_meta,
            loader_configs=None,
            loader_type="dots_ocr",
        )

        self._assert_dots_ocr_document_loaded(json_doc, md_doc)

        # save the json and md in the corresponding formats
        import json

        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")

        json_path = f"./tmp/{filename}.json"
        with open(json_path, "w") as json_file:
            json.dump(json_doc, json_file, ensure_ascii=False, indent=4)

        md_path = f"./tmp/{filename}.md"
        with open(md_path, "w") as md_file:
            md_file.write(md_doc.page_content)

        # assert exists
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
