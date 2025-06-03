from langchain_community import document_loaders

from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.loader.markify_loader import markify_client


class ExcelLoader(BaseLoader):
    def __init__(self):
        self.loader_type = document_loaders.UnstructuredExcelLoader
        self.loader_markify = markify_client
