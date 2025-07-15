from docling.document_converter import DocumentConverter

from hirag_prod.loader.base_loader import BaseLoader


class MdLoader(BaseLoader):
    def __init__(self):
        self.loader_docling = DocumentConverter()
