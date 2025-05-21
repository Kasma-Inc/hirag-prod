#! /usr/bin/env python3

from abc import ABC
from typing import List, Optional, Type

from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader

from .._utils import compute_mdhash_id
from ..schema import File


class BaseLoader(ABC):
    """Base class for all loaders"""

    loader_type: Type[LangchainBaseLoader]

    # additional metadata to add to the loaded raw documents
    page_number_key: str = "page_number"

    def _load(self, document_path: str, **loader_args) -> List[File]:
        raw_docs = self.loader_type(document_path, **loader_args).load()
        return raw_docs

    def load(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> list[File]:
        """Load document and set the metadata of the output

        Args:
            document_path (str): The document path for langchain loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the langchain loader.

        Returns:
            list[File]: Raw files.
        """
        if document_meta is None:
            document_meta = {}
        raw_docs = self._load(document_path, **loader_args)
        self._set_doc_metadata(raw_docs, document_meta)
        return raw_docs

    def _set_doc_metadata(self, files: List[File], document_meta: dict) -> List[File]:
        for file in files:
            document_meta[self.page_number_key] = None
            file.metadata = document_meta
            file.id = compute_mdhash_id(file.page_content.strip(), prefix="doc-")
