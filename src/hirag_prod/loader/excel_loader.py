from langchain_community import document_loaders

from hirag_prod.loader.base_loader import BaseLoader


class ExcelLoader(BaseLoader):
    def __init__(self):
        self.loader_type = document_loaders.UnstructuredExcelLoader
