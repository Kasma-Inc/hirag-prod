from docling.document_converter import DocumentConverter

from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.loader.docling_cloud import DoclingCloudClient


class MdLoader(BaseLoader):
    def __init__(self):
        self.loader_docling = DocumentConverter()
        self.loader_docling_cloud = DoclingCloudClient()
