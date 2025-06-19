from typing import List, Optional

from langchain_community import document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.schema import File, FileMetadata

SEPARATOR = "=+=+=+=+=+=+=+=+="


class TxtLoader(BaseLoader):
    """Specialized loader for txt documents, using langchain's TextLoader and RecursiveCharacterTextSplitter"""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize TxtLoader

        Args:
            chunk_size (int): maximum token number of each chunk, default is 1000
            chunk_overlap (int): overlap token number between chunks, default is 200
            separators (Optional[List[str]]): separators for splitting, default is SEPARATOR
        """
        self.loader_type = document_loaders.TextLoader

        # set default separators
        if separators is None:
            separators = [SEPARATOR]

        # create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def _load(self, document_path: str, **loader_args) -> List[File]:
        """
        override the _load method, use langchain's TextLoader and RecursiveCharacterTextSplitter

        Args:
            **loader_args: loader arguments

        Returns:
            List[File]: split documents
        """
        # use TextLoader to load the document
        loader = self.loader_type(document_path, **loader_args)

        # use text_splitter to split the document
        raw_docs = loader.load_and_split(text_splitter=self.text_splitter)

        # convert to File object
        docs = []
        for i, doc in enumerate(raw_docs, start=1):
            file_doc = File(
                id=compute_mdhash_id(doc.page_content, prefix="doc-"),
                page_content=doc.page_content,
                metadata=FileMetadata(page_number=i),
            )
            docs.append(file_doc)

        return docs
