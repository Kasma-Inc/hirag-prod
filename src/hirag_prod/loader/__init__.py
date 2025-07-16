from typing import Any, Literal, Optional, Tuple

from hirag_prod.loader.csv_loader import CSVLoader
from hirag_prod.loader.excel_loader import ExcelLoader
from hirag_prod.loader.html_loader import HTMLLoader
from hirag_prod.loader.md_loader import MdLoader
from hirag_prod.loader.pdf_loader import PDFLoader
from hirag_prod.loader.ppt_loader import PowerPointLoader
from hirag_prod.loader.txt_loader import TxtLoader
from hirag_prod.loader.word_loader import WordLoader
from hirag_prod.schema import File

DEFAULT_LOADER_CONFIGS = {
    "application/pdf": {
        "loader": PDFLoader,
        "args": {},
    },
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
        "loader": WordLoader,
        "args": {},
    },
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
        "loader": PowerPointLoader,
        "args": {},
    },
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
        "loader": ExcelLoader,
        "args": {},
    },
    "text/html": {
        "loader": HTMLLoader,
        "args": {},
    },
    "text/csv": {
        "loader": CSVLoader,
        "args": {},
    },
    "text/markdown": {
        "loader": MdLoader,
        "args": {},
    },
    "text/plain": {
        "loader": TxtLoader,
        "args": {},
    },
}


def load_document(
    document_path: str,
    content_type: str,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
    loader_type: Literal["docling", "OCR", "langchain"] = "docling",
) -> Tuple[Any, File]:
    """Load a document from the given path and content type

    Args:
        document_path (str): The path to the document.
        content_type (str): The content type of the document.
        document_meta (Optional[dict]): The metadata of the document.
        loader_configs (Optional[dict]): If unspecified, use DEFAULT_LOADER_CONFIGS.

    Raises:
        ValueError: If the content type is not supported.

    Returns:
        Tuple[Any, File]: The loaded document.
    """
    # TODO: Optimize loader selection logic - consolidate loader types and reduce branching
    # TODO: Add async support for concurrent document loading
    if content_type == "text/plain":
        loader_type = "langchain"
    # TODO: OCR NOT SUPPORTED YET
    # elif content_type == "application/pdf":
    #     loader_type = "OCR"
    else:
        loader_type = "docling"

    if loader_configs is None:
        loader_configs = DEFAULT_LOADER_CONFIGS

    if content_type not in loader_configs:
        raise ValueError(f"Unsupported document type: {content_type}")
    loader_conf = loader_configs[content_type]
    loader = loader_conf["loader"]()

    if loader_type == "docling":
        docling_doc, doc_md = loader.load_docling(document_path, document_meta)
        return docling_doc, doc_md
    elif loader_type == "langchain":
        langchain_doc = loader.load_langchain(document_path, document_meta)
        return langchain_doc
    elif loader_type == "OCR":
        if loader is not isinstance(PDFLoader):
            raise ValueError("OCR loader only supports PDF documents")
        ocr_doc, doc_md = loader.load_ocr(document_path, document_meta)
        return ocr_doc, doc_md


__all__ = [
    "PowerPointLoader",
    "PDFLoader",
    "WordLoader",
    "ExcelLoader",
    "load_document",
    "HTMLLoader",
    "CSVLoader",
    "TxtLoader",
    "MdLoader",
]
