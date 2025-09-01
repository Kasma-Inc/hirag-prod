import json
import os

from hirag_prod.loader import load_document
from hirag_prod.loader.chunk_split import (
    chunk_docling_document,
    chunk_dots_document,
    chunk_langchain_document,
    get_toc_from_chunks,
)


def _brief(c):
    content = (c.text or "").replace("\n", " ")
    # if len(content) > 80:
    #     content = content[:80] + "..."

    return (
        f"id={c.id} | idx={c.chunkIdx} | type={c.chunkType} | "
        f"page={c.pageNumber} | headers={c.headers} | children={c.children} | caption={c.caption} | "
        f'bbox={c.bbox} | text={content} | size=({c.pageWidth}x{c.pageHeight})"'
    )


def test_chunk_docling_document():
    """Test chunking a pdf document using the docling loader and chunk_docling_document function"""
    # Load a pdf document first using the docling loader
    document_path = (
        f"{os.path.dirname(__file__)}/test_files/Guide-to-U.S.-Healthcare-System.pdf"
    )
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
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

    # print out the chunks
    for c in chunks:
        print("[docling]", _brief(c))

    assert chunks is not None
    assert len(chunks) > 0

    assert hasattr(chunks[0], "documentKey")
    assert hasattr(chunks[0], "text")

    assert chunks[0].documentKey.startswith("chunk-")
    assert chunks[0].text is not None

    assert hasattr(chunks[0], "chunkIdx")
    assert hasattr(chunks[0], "documentId")
    assert hasattr(chunks[0], "chunkType")

    assert chunks[0].type == "pdf"
    assert chunks[0].fileName == "Guide-to-U.S.-Healthcare-System.pdf"
    assert chunks[0].uri == document_path
    assert chunks[0].private is False
    assert chunks[0].documentId == doc_md.id

    assert chunks[0].chunkIdx == 0
    assert chunks[0].chunkType is not None

    print(f"[docling] total chunks: {len(chunks)}")
    # for c in chunks[:3]:
    #     print("[docling]", _brief(c))


def test_chunk_dots_document():
    """Test chunking a pdf document using the dots loader and chunk_dots_document function"""
    # Load a pdf document first using the dots loader
    file_name = "Attention 1706.03762v7.pdf"

    document_path = f"s3://monkeyocr/test/input/test_pdf/{file_name}"

    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": file_name,
        "uri": document_path,
        "private": False,
    }

    json_doc, doc_md = load_document(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        loader_configs=None,
        loader_type="dots_ocr",
    )

    chunks = chunk_dots_document(json_doc, doc_md)

    # print out the chunks
    for c in chunks:
        print("[dots]", _brief(c))

    assert chunks is not None
    assert len(chunks) > 0

    assert hasattr(chunks[0], "documentKey")
    assert hasattr(chunks[0], "text")

    assert chunks[0].documentKey.startswith("chunk-")
    assert chunks[0].text is not None

    assert hasattr(chunks[0], "chunkIdx")
    assert hasattr(chunks[0], "documentId")
    assert hasattr(chunks[0], "chunkType")

    assert chunks[0].chunkType == "pdf"
    assert chunks[0].fileName == file_name
    assert chunks[0].uri == document_path
    assert chunks[0].private is False
    assert chunks[0].documentId == doc_md.id

    assert chunks[0].chunkIdx == 0
    assert chunks[0].chunkType is not None

    print(f"[dots] total chunks: {len(chunks)}")
    toc = get_toc_from_chunks(chunks)
    print(f"[dots] ToC: {json.dumps(toc, indent=2)}")
    # for c in chunks[:3]:
    #     print("[dots]", _brief(c))


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

    _, langchain_doc = load_document(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
        loader_configs=None,
        loader_type="langchain",
    )

    chunks = chunk_langchain_document(langchain_doc)

    assert chunks is not None
    assert len(chunks) > 0

    assert hasattr(chunks[0], "documentKey")
    assert hasattr(chunks[0], "text")

    assert chunks[0].documentKey.startswith("chunk-")
    assert chunks[0].text is not None

    assert hasattr(chunks[0], "chunkIdx")
    assert hasattr(chunks[0], "documentId")
    assert hasattr(chunks[0], "chunkType")

    assert chunks[0].chunkType == "txt"
    assert chunks[0].fileName == "test.txt"
    assert chunks[0].uri == document_path
    assert chunks[0].private is False
    assert chunks[0].documentId == langchain_doc.id

    assert chunks[0].chunkIdx == 0
    assert chunks[0].chunkType is not None

    print(f"[langchain] total chunks: {len(chunks)}")
    for c in chunks[:2]:
        print("[langchain]", _brief(c))


if __name__ == "__main__":
    # test_chunk_docling_document()
    # test_chunk_langchain_document()
    test_chunk_dots_document()
