import logging
import threading
from typing import Any, Literal, Optional, Tuple

import requests

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_document_converter_config
from hirag_prod.loader.csv_loader import CSVLoader
from hirag_prod.loader.html_loader import HTMLLoader
from hirag_prod.loader.image_loader import ImageLoader
from hirag_prod.loader.md_loader import MdLoader
from hirag_prod.loader.pdf_loader import PDFLoader
from hirag_prod.loader.ppt_loader import PowerPointLoader
from hirag_prod.loader.txt_loader import TxtLoader
from hirag_prod.loader.utils import route_file_path, validate_document_path
from hirag_prod.loader.word_loader import WordLoader
from hirag_prod.schema import File, LoaderType
from hirag_prod.tracing import traced

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


def check_cloud_health(
    document_converter_type: Literal["dots_ocr"],
) -> bool:
    """Check the health of the cloud service"""
    try:
        health_url = f"{get_document_converter_config(document_converter_type).base_url.rstrip('/')}/health"

        headers = {
            "Content-Type": "application/json",
            "Model-Name": get_document_converter_config(
                document_converter_type
            ).model_name,
            "Authorization": f"Bearer {get_document_converter_config(document_converter_type).api_key}",
        }

        resp = requests.get(health_url, headers=headers)

        # Check if response is empty/null (success case)
        if resp.status_code == 200 and not resp.text.strip():
            return True

        # Try to parse JSON response
        try:
            data = resp.json()
            # Return True if the JSON response indicates success
            if data.get("success") == "true":
                return True
            # Otherwise False
            return False
        except Exception as e:
            # If we can't parse JSON but got a response, treat as failure
            log_error_info(logging.ERROR, "Failed to parse JSON response", e)
            return False

    except Exception as e:
        log_error_info(logging.ERROR, "Failed to check cloud health", e)
        return False


docling_lock: threading.Lock = threading.Lock()


@traced()
def load_document(
    document_path: str,
    content_type: str,
    loader_type: LoaderType,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
) -> Tuple[Any, File]:
    """Load a document from the given path and content type

    Args:
        document_path (str): The path to the document.
        content_type (str): The content type of the document.
        loader_type (LoaderType): The loader type to use.
        document_meta (Optional[dict]): The metadata of the document.
        loader_configs (Optional[dict]): If unspecified, use DEFAULT_LOADER_CONFIGS.

    Raises:
        ValueError: If the content type is not supported.

    Returns:
        Tuple[Any, File]: The loaded document.
    """

    if loader_configs is None:
        loader_configs = DEFAULT_LOADER_CONFIGS

    if content_type not in loader_configs:
        raise ValueError(f"Unsupported document type: {content_type}")
    loader_conf = loader_configs[content_type]
    loader = loader_conf["loader"]()

    # Dots OCR doesn't require routing, so handle it separately
    if loader_type == "dots_ocr":
        try:
            cloud_check = check_cloud_health("dots_ocr")
            if not cloud_check:
                raise RuntimeError(f"Cloud health check failed for dots_ocr.")
            json_doc, doc_md = loader.load_dots_ocr(document_path, document_meta)
            return json_doc, doc_md
        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Error loading document with Dots OCR, falling back to docling",
                e,
            )
            loader_type = "docling"

    # Route for local loaders
    try:
        document_path = route_file_path(document_path)
    except Exception as e:
        log_error_info(
            logging.WARNING,
            f"Unexpected error in route_file_path, using original path",
            e,
        )

    validate_document_path(document_path)

    if loader_type == "docling":
        with docling_lock:
            docling_doc, doc_md = loader.load_docling(document_path, document_meta)
        return docling_doc, doc_md

    if loader_type == "langchain":
        langchain_doc = loader.load_langchain(document_path, document_meta)
        return None, langchain_doc

    raise ValueError(f"Unsupported loader type: {loader_type}")


__all__ = [
    "PowerPointLoader",
    "PDFLoader",
    "WordLoader",
    "load_document",
    "HTMLLoader",
    "CSVLoader",
    "TxtLoader",
    "MdLoader",
]
