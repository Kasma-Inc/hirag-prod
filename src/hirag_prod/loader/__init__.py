import os
from typing import List, Literal, Optional

from hirag_prod.schema import File

from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .ppt_loader import PowerPointLoader
from .ppt_parser import PPTParser
from .txt_loader import TxtLoader
from .word_loader import WordLoader

DEFAULT_LOADER_CONFIGS = {
    "application/pdf": {
        "loader": PDFLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
        "init_args": {
            "max_output_docs": 5,
        },
    },
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
        "loader": WordLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
    },
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
        "loader": PowerPointLoader,
        "args": {"mode": "single", "strategy": "fast"},
    },
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
        "loader": ExcelLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
    },
    "text/html": {
        "loader": HTMLLoader,
        "args": {},
    },
    "text/csv": {
        "loader": CSVLoader,
        "args": {},
    },
    "text/plain": {
        "loader": TxtLoader,
        "args": {},
    },
}


# TODO: Remove deprecated PPTAGENT_LOADER_CONFIGS - no longer used in production
# This configuration was abandoned and should be cleaned up from codebase
PPTAGENT_LOADER_CONFIGS = {
    "pptx": {"loader": PPTParser, "init_args": {"work_dir": "temp"}},
}


def load_document(
    document_path: str,
    content_type: str,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
    loader_type: Literal["doc2x", "langchain", "pptagent"] = "langchain",
) -> List[File]:
    """Load a document from the given path and content type

    Args:
        document_path (str): The path to the document.
        content_type (str): The content type of the document.
        document_meta (Optional[dict]): The metadata of the document.
        loader_configs (Optional[dict]): If unspecified, use DEFAULT_LOADER_CONFIGS.

    Raises:
        ValueError: If the content type is not supported.

    Returns:
        List[File]: The loaded documents.
    """
    # TODO: Optimize loader selection logic - consolidate loader types and reduce branching
    # TODO: Add async support for concurrent document loading
    if loader_configs is None:
        loader_configs = DEFAULT_LOADER_CONFIGS
    elif loader_configs is None and loader_type == "pptagent":
        loader_configs = PPTAGENT_LOADER_CONFIGS

    if content_type not in loader_configs:
        raise ValueError(f"Unsupported document type: {content_type}")
    loader_conf = loader_configs[content_type]
    if "init_args" in loader_conf:
        loader = loader_conf["loader"](**loader_conf["init_args"])
    else:
        loader = loader_conf["loader"]()

    if loader_type == "langchain":
        if "args" in loader_conf:
            raw_docs = loader.load(document_path, document_meta, **loader_conf["args"])
        else:
            raw_docs = loader.load(document_path, document_meta)
    elif loader_type == "doc2x":
        raw_docs = loader.load_doc2x(document_path, document_meta)
    elif loader_type == "pptagent":
        # work_dir: document_path=.../xxx.pptx → .../ppt_templates/xxx
        abs_doc_path = os.path.abspath(document_path)
        doc_dir = os.path.dirname(abs_doc_path)
        doc_name = os.path.splitext(os.path.basename(abs_doc_path))[0]
        # work_dir = <doc_dir>/ppt_templates/<doc_name>
        work_dir = os.path.join(doc_dir, "ppt_templates", doc_name)
        loader = PPTParser(work_dir=work_dir)
        raw_docs = loader.parse_pptx(document_path)
    else:
        raise ValueError(
            f"Unsupported loader type: {loader_type}, should be one of ['doc2x', 'langchain', 'pptagent']"
        )
    return raw_docs


__all__ = [
    "PowerPointLoader",
    "PDFLoader",
    "WordLoader",
    "ExcelLoader",
    "load_document",
    "HTMLLoader",
    "CSVLoader",
    "PPTParser",
    "TxtLoader",
]
