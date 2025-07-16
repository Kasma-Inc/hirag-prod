import os

from hirag_prod.loader import load_document
from hirag_prod.loader.chunk_split import (
    chunk_docling_document,
    chunk_langchain_document,
)


def test_chunk_docling_document():
    """Test chunking a docx document using the docling loader and chunk_docling_document function"""
    # Load a docx document first using the docling loader
    document_path = f"{os.path.dirname(__file__)}/test_files/word_sample.docx"
    content_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    document_meta = {
        "type": "docx",
        "filename": "word_sample.docx",
        "uri": document_path,
        "private": False,
    }

    docling_doc, doc_md = load_document(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        loader_configs=None,
        loader_type="docling",
    )

    chunks = chunk_docling_document(docling_doc, doc_md)

    assert chunks is not None
    assert len(chunks) > 0

    assert hasattr(chunks[0], "id")
    assert hasattr(chunks[0], "page_content")
    assert hasattr(chunks[0], "metadata")

    assert chunks[0].id.startswith("chunk-")
    assert chunks[0].page_content is not None

    assert hasattr(chunks[0].metadata, "chunk_idx")
    assert hasattr(chunks[0].metadata, "document_id")
    assert hasattr(chunks[0].metadata, "chunk_type")

    assert chunks[0].metadata.type == "docx"
    assert chunks[0].metadata.filename == "word_sample.docx"
    assert chunks[0].metadata.uri == document_path
    assert chunks[0].metadata.private == False
    assert chunks[0].metadata.document_id == doc_md.id

    assert chunks[0].metadata.chunk_idx == 0
    assert chunks[0].metadata.chunk_type is not None


def test_chunk_langchain_document():
    """Test chunking a txt document using the langchain loader and chunk_langchain_document function"""
    # Load a txt document first using the langchain loader
    document_path = f"{os.path.dirname(__file__)}/test_files/test.txt"
    content_type = "text/plain"
    document_meta = {
        "type": "txt",
        "filename": "test.txt",
        "uri": document_path,
        "private": False,
    }

    langchain_doc = load_document(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        loader_configs=None,
        loader_type="langchain",
    )

    chunks = chunk_langchain_document(langchain_doc)

    assert chunks is not None
    assert len(chunks) > 0

    assert hasattr(chunks[0], "id")
    assert hasattr(chunks[0], "page_content")
    assert hasattr(chunks[0], "metadata")

    assert chunks[0].id.startswith("chunk-")
    assert chunks[0].page_content is not None

    assert hasattr(chunks[0].metadata, "chunk_idx")
    assert hasattr(chunks[0].metadata, "document_id")
    assert hasattr(chunks[0].metadata, "chunk_type")

    assert chunks[0].metadata.type == "txt"
    assert chunks[0].metadata.filename == "test.txt"
    assert chunks[0].metadata.uri == document_path
    assert chunks[0].metadata.private == False
    assert chunks[0].metadata.document_id == langchain_doc.id

    assert chunks[0].metadata.chunk_idx == 0
    assert chunks[0].metadata.chunk_type is not None
