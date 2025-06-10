#! /usr/bin/env python3
import os
from abc import ABC
from typing import List, Optional, Type

from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader
from pptagent.document import Document
from pptagent.llms import LLM

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.schema import File, FileMetadata

from .doc2x_loader import Doc2XClient


class BaseLoader(ABC):
    """Base class for all loaders"""

    loader_type: Type[LangchainBaseLoader]
    loader_doc2x: Type[Doc2XClient]
    # additional metadata to add to the loaded raw documents
    page_number_key: str = "page_number"

    def _load(self, document_path: str, **loader_args) -> List[File]:
        raw_docs = self.loader_type(document_path, **loader_args).load()
        docs = []
        for i, doc in enumerate(raw_docs, start=1):
            # Only set page number and doc hash here
            doc = File(
                id=compute_mdhash_id(doc.page_content, prefix="doc-"),
                page_content=doc.page_content,
                metadata=FileMetadata(page_number=i),
            )
            docs.append(doc)
        return docs

    def _load_doc2x(self, document_path: str) -> List[File]:
        docs = self.loader_doc2x.convert_and_split_pdf(file_path=document_path)

        return docs

    def load(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> list[File]:
        """Load document and set the metadata of the output

        Args:
            document_path (str): The document path for langchain loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the langchain loader.

        Returns:
            list[File]: Raw documents.
        """
        if document_meta is None:
            document_meta = {}
        raw_docs = self._load(document_path, **loader_args)
        self._set_doc_metadata(raw_docs, document_meta)
        return raw_docs

    def load_doc2x(
        self, document_path: str, document_meta: Optional[dict] = None
    ) -> List[File]:
        """Load document with Doc2X and set the metadata of the output

        Args:
            document_path (str): The document path for Doc2X loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.

        Returns:
            list[File]: Raw documents.
        """
        if document_meta is None:
            document_meta = {}
        raw_docs = self._load_doc2x(document_path)
        self._set_doc_metadata(raw_docs, document_meta)
        return raw_docs

    def _set_doc_metadata(self, docs: List[File], document_meta: dict) -> List[File]:
        for doc in docs:
            # Keep the original page number
            page_number = doc.metadata.page_number

            assert document_meta.get("private") is not None, "private is required"
            # Create metadata with all required fields
            metadata = FileMetadata(
                page_number=page_number,
                type=document_meta.get("type", "pdf"),  # Default to pdf
                filename=document_meta.get("filename", ""),
                uri=document_meta.get("uri", ""),
                private=document_meta.get("private"),
            )
            doc.metadata = metadata
        return docs

    def to_file_content(self, docs: List[File], image_dir: str) -> dict:
        """
        Refine the document structure with language model assistance

        Args:
            docs (List[File]): The extracted text contents from input file
            image_dir (str): The image directory of the input file

        Returns:
            dict: The refined document structure as a dictionary
        """
        text_content = "".join(
            doc.page_content for doc in docs
        )  # concate a list of markdown texts

        api_base = os.environ.get("API_BASE", None)
        language_model_name = os.environ.get("LANGUAGE_MODEL", "gpt-4.1")
        vision_model_name = os.environ.get("VISION_MODEL", "gpt-4.1")

        language_model = LLM(language_model_name, api_base)
        vision_model = LLM(vision_model_name, api_base)

        file_content = Document.from_markdown(
            text_content, language_model, vision_model, image_dir
        )

        return file_content.to_dict()
