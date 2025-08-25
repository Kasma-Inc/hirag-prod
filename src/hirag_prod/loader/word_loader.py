from docling.document_converter import DocumentConverter

from .base_loader import BaseLoader
from .docling_cloud import DoclingCloudClient


class WordLoader(BaseLoader):
    def __init__(self):
        self.loader_docling = DocumentConverter()
        self.loader_docling_cloud = DoclingCloudClient()
