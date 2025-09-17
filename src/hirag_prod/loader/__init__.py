import logging
from typing import Any, Literal, Optional, Tuple

import requests

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_document_converter_config
from hirag_prod.loader.csv_loader import CSVLoader
from hirag_prod.loader.excel_loader import ExcelLoader
from hirag_prod.loader.html_loader import HTMLLoader
from hirag_prod.loader.image_loader import ImageLoader
from hirag_prod.loader.md_loader import MdLoader
from hirag_prod.loader.pdf_loader import PDFLoader
from hirag_prod.loader.ppt_loader import PowerPointLoader
from hirag_prod.loader.txt_loader import TxtLoader
from hirag_prod.loader.utils import route_file_path, validate_document_path
from hirag_prod.loader.word_loader import WordLoader
from hirag_prod.schema import File, LoaderType

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


def check_cloud_health(
    document_converter_type: Literal["dots_ocr", "docling_cloud"],
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
            # Return True if the JSON response indicates success (which is not possible now)
            if data.get("success") == "true":
                return True
            # Otherwise False
            return False
        except Exception as e:
            # If we can't parse JSON but got a response, treat as failure
            log_error_info(logging.ERROR, "Failed to parsing JSON response", e)
            return False

    except Exception as e:
        log_error_info(logging.ERROR, "Failed to check cloud health", e)
        return False


def load_document(
    document_path: str,
    content_type: str,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
    loader_type: LoaderType = "dots_ocr",
) -> Tuple[Any, File]:
    """Load a document from the given path and content type

    Args:
        document_path (str): The path to the document.
        content_type (str): The content type of the document.
        document_meta (Optional[dict]): The metadata of the document.
        loader_configs (Optional[dict]): If unspecified, use DEFAULT_LOADER_CONFIGS.
        loader_type (LoaderType): The loader type to use.

    Raises:
        ValueError: If the content type is not supported.

    Returns:
        Tuple[Any, File]: The loaded document.
    """
    # TODO: Optimize loader selection logic - consolidate loader types and reduce branching
    # TODO: Add async support for concurrent document loading
    if content_type == "text/plain":
        loader_type = "langchain"
    elif content_type == "text/markdown":  # Prefer local modified docling for markdown
        loader_type = "docling"

    if loader_type in ["docling_cloud", "dots_ocr"]:
        cloud_check = False
        if loader_type == "docling_cloud":
            cloud_check = check_cloud_health("docling_cloud")
        elif loader_type == "dots_ocr":
            cloud_check = check_cloud_health("dots_ocr")

        if not cloud_check:
            # Show warning in log
            logger.warning(
                f"Cloud health check failed for {loader_type}, falling back to docling."
            )
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
        log_error_info(
            logging.WARNING,
            f"Unexpected error in route_file_path, using original path",
            e,
        )

    if loader_type == "docling_cloud":
        docling_doc, doc_md = loader.load_docling_cloud(document_path, document_meta)
        return docling_doc, doc_md
    elif loader_type == "dots_ocr":
        json_doc, doc_md = loader.load_dots_ocr(document_path, document_meta)
        return json_doc, doc_md
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
