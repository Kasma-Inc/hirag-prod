from langchain_community import document_loaders

from hirag_prod.loader.base_loader import BaseLoader


class CSVLoader(BaseLoader):
    def __init__(self):
        self.loader_type = document_loaders.CSVLoader
