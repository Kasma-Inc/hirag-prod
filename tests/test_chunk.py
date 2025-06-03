import os

from hirag_prod.chunk import FixTokenChunk
from hirag_prod.loader import load_document


def test_chunk_documents():
    # Load a document first using the same approach as in test_loader.py
    document_path = f"{os.path.dirname(__file__)}/Guide-to-U.S.-Healthcare-System.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    documents = load_document(
        document_path, content_type, document_meta, loader_type="langchain"
    )

    # Test chunking the loaded documents
    chunker = FixTokenChunk(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    for document in documents:
        chunks = chunker.chunk(document)
        chunked_docs.extend(chunks)

    # Verify the chunking results
    assert chunked_docs is not None
    assert len(chunked_docs) > 0

    # Check that each chunk has the expected metadata
    for chunk_id, chunk in enumerate(chunked_docs):
        assert "chunk_idx" in chunk.metadata.__dict__
        assert chunk.page_content is not None
        assert chunk.metadata.type == "pdf"
        assert chunk.metadata.filename == "Guide-to-U.S.-Healthcare-System.pdf"
