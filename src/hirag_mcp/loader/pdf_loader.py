import warnings
from typing import List

from langchain_community import document_loaders
from ..loader.base_loader import BaseLoader

from .._utils import compute_mdhash_id
from ..schema import File

# Suppress PyPDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


class PDFLoader(BaseLoader):
    """Loads PDF documents"""

    def __init__(self, max_output_docs: int = 5):
        self.loader_type = document_loaders.PyPDFLoader

    def _load(self, document_path: str, **loader_args) -> List[File]:
        raw_docs = self.loader_type(document_path, **loader_args).load()
        return raw_docs

    def _set_doc_metadata(self, files: List[File], document_meta: dict) -> List[File]:
        for doc in files:
            document_meta[self.page_number_key] = doc.metadata["page"]
            doc.metadata = document_meta
            doc.id = compute_mdhash_id(doc.page_content.strip(), prefix="doc-")
