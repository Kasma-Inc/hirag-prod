import os

from hirag_prod.loader import load_document


def test_load_pdf_langchain():
    document_path = f"{os.path.dirname(__file__)}/Guide-to-U.S.-Healthcare-System.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    loader_configs = None
    documents = load_document(
        document_path,
        content_type,
        document_meta,
        loader_configs,
        loader_type="langchain",
    )
    assert len(documents) > 0
    assert documents[0].page_content is not None
    assert documents[0].metadata is not None
    assert documents[0].id.startswith("doc-")


def test_load_pdf_doc2x():
    document_path = f"{os.path.dirname(__file__)}/agent.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "agent.pdf",
        "uri": document_path,
        "private": False,
    }
    documents = load_document(
        document_path, content_type, document_meta, loader_type="doc2x"
    )
    assert len(documents) > 0
    assert documents[0].page_content is not None
    assert documents[0].metadata is not None
    assert documents[0].id.startswith("doc-")


def test_parse_pptx():
    document_path = os.path.join(os.path.dirname(__file__), "Beamer.pptx")
    content_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    document_meta = {
        "type": "pptx",
        "filename": "Beamer.pptx",
        "uri": document_path,
        "private": False,
    }
    loader_configs = None
    documents = load_document(
        document_path,
        content_type,
        document_meta,
        loader_configs,
        loader_type="pptagent",
    )
    assert len(documents) >= 1
    # work_dir is auto-detected as .../ppt_templates/Beamer
    abs_doc_path = os.path.abspath(document_path)
    doc_dir = os.path.dirname(abs_doc_path)
    doc_name = os.path.splitext(os.path.basename(abs_doc_path))[0]
    work_dir = os.path.join(doc_dir, "ppt_templates", doc_name)
    assert documents[0].metadata.uri == work_dir
    # Check that work_dir exists and contains expected files
    assert os.path.isdir(work_dir)
    assert os.path.isfile(os.path.join(work_dir, "image_stats.json"))
    assert os.path.isfile(os.path.join(work_dir, "slide_induction.json"))
    assert os.path.isdir(os.path.join(work_dir, "slide_images"))
    assert os.path.isdir(os.path.join(work_dir, "images"))
    assert os.path.isdir(os.path.join(work_dir, "template_images"))
