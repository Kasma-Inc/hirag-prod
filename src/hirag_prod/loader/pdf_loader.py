import warnings

from langchain_community import document_loaders

from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.loader.markify_loader import markify_client

# Suppress PyPDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


class PDFLoader(BaseLoader):
    """Loads PDF documents"""

    def __init__(self, max_output_docs: int = 5):
        self.loader_type = document_loaders.PyPDFLoader
        self.loader_markify = markify_client
        self.max_output_docs = max_output_docs
