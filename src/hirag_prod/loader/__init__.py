import logging
import os
from typing import Any, Literal, Optional, Tuple

import requests

from hirag_prod._utils import route_file_path, validate_document_path
from hirag_prod.loader.csv_loader import CSVLoader
from hirag_prod.loader.excel_loader import ExcelLoader
from hirag_prod.loader.html_loader import HTMLLoader
from hirag_prod.loader.image_loader import ImageLoader
from hirag_prod.loader.md_loader import MdLoader
from hirag_prod.loader.pdf_loader import PDFLoader
from hirag_prod.loader.ppt_loader import PowerPointLoader
from hirag_prod.loader.txt_loader import TxtLoader
from hirag_prod.loader.word_loader import WordLoader
from hirag_prod.schema import File

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HiRAG")

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
    "multimodal/image": {
        "loader": ImageLoader,
        "args": {},
    },
}


def check_docling_cloud_health() -> bool:
    """docling cloud service health check"""
    try:
        base_url = os.path.join(os.getenv("DOCLING_CLOUD_BASE_URL"), "/health")
        token = os.getenv("DOCLING_CLOUD_AUTH_TOKEN")
        model = os.getenv("DOCLING_CLOUD_MODEL_NAME", "docling")
        if not base_url or not token:
            return False
        headers = {
            "Model-Name": model,
            "Authorization": f"Bearer {token}",
        }
        resp = requests.post(base_url, headers=headers)
        try:
            data = resp.json()
        except Exception:
            return True

        return data.get("success", True)
    except Exception:
        return False


def load_document(
    document_path: str,
    content_type: str,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
    loader_type: Literal["docling", "docling_cloud", "langchain"] = "docling_cloud",
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
    else:
        if check_docling_cloud_health():
            loader_type = "docling_cloud"
        else:
            loader_type = "docling"

    if loader_configs is None:
        loader_configs = DEFAULT_LOADER_CONFIGS

    if content_type not in loader_configs:
        raise ValueError(f"Unsupported document type: {content_type}")
    loader_conf = loader_configs[content_type]
    loader = loader_conf["loader"]()

    try:
        document_path = route_file_path(loader_type, document_path)
    except Exception as e:
        logger.warning(f"Unexpected error in route_file_path, using original path: {e}")

    if loader_type == "docling_cloud":
        docling_doc, doc_md = loader.load_docling_cloud(document_path, document_meta)
        return docling_doc, doc_md
    elif loader_type == "docling":
        validate_document_path(document_path)
        docling_doc, doc_md = loader.load_docling(document_path, document_meta)
        return docling_doc, doc_md
    elif loader_type == "langchain":
        validate_document_path(document_path)
        langchain_doc = loader.load_langchain(document_path, document_meta)
        return None, langchain_doc


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
