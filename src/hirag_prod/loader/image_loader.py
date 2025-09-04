from docling.document_converter import DocumentConverter

from hirag_prod.loader.base_loader import BaseLoader


class ImageLoader(BaseLoader):
    """Loads images"""

    def __init__(self):
        self.loader_docling = DocumentConverter()
