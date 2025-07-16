from langchain_community import document_loaders

from hirag_prod.loader.base_loader import BaseLoader


class TxtLoader(BaseLoader):
    """Specialized loader for txt documents, using langchain's TextLoader"""

    def __init__(self):
        self.loader_langchain = document_loaders.TextLoader
