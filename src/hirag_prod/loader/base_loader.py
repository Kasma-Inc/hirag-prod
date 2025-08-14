#! /usr/bin/env python3
from abc import ABC
from typing import Optional, Tuple, Type

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.loader.docling_cloud import DoclingCloudClient
from hirag_prod.schema import File, FileMetadata


class BaseLoader(ABC):
    """Base class for all loaders"""

    loader_docling: Type[DocumentConverter]
    loader_docling_cloud: Type[DoclingCloudClient]
    loader_langchain: Type[LangchainBaseLoader]

    def load_docling_cloud(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> Tuple[DoclingDocument, File]:
        """Load document and set the metadata of the split chunks

        Args:
            document_path (str): The document path for loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the loader.

        Returns:
            File: the loaded document
        """
        assert document_meta.get("private") is not None, "private is required"
        docling_doc: DoclingDocument = self.loader_docling_cloud.convert(document_path)
        md_str: str = docling_doc.export_to_markdown()
        doc_md = File(
            id=compute_mdhash_id(md_str, prefix="doc-"),
            page_content=md_str,
            metadata=FileMetadata(
                type=document_meta.get("type", "pdf"),  # Default to pdf
                filename=document_meta.get("filename", ""),
                uri=document_meta.get("uri", ""),
                private=document_meta.get("private"),
                uploaded_at=document_meta.get("uploaded_at"),
                knowledge_base_id=document_meta.get("knowledge_base_id", ""),
                workspace_id=document_meta.get("workspace_id", ""),
            ),
        )
        return docling_doc, doc_md

    def load_docling(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> Tuple[DoclingDocument, File]:
        """Load document and set the metadata of the split chunks

        Args:
            document_path (str): The document path for loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the loader.

        Returns:
            File: the loaded document
        """
        assert document_meta.get("private") is not None, "private is required"
        docling_doc: DoclingDocument = self.loader_docling.convert(
            document_path
        ).document
        md_str: str = docling_doc.export_to_markdown()
        doc_md = File(
            id=compute_mdhash_id(md_str, prefix="doc-"),
            page_content=md_str,
            metadata=FileMetadata(
                type=document_meta.get("type", "pdf"),  # Default to pdf
                filename=document_meta.get("filename", ""),
                uri=document_meta.get("uri", ""),
                private=document_meta.get("private"),
                uploaded_at=document_meta.get("uploaded_at"),
                knowledge_base_id=document_meta.get("knowledge_base_id", ""),
                workspace_id=document_meta.get("workspace_id", ""),
            ),
        )
        return docling_doc, doc_md

    def load_langchain(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> File:
        """Load document and set the metadata of the split chunks

        Args:
            document_path (str): The document path for loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the loader.

        Returns:
            File: the loaded document
        """
        assert document_meta.get("private") is not None, "private is required"
        langchain_docs = self.loader_langchain(document_path, **loader_args).load()
        doc_langchain = File(
            id=compute_mdhash_id(langchain_docs[0].page_content, prefix="doc-"),
            page_content=langchain_docs[0].page_content,
            metadata=FileMetadata(
                type=document_meta.get("type", "pdf"),  # Default to pdf
                filename=document_meta.get("filename", ""),
                uri=document_meta.get("uri", ""),
                private=document_meta.get("private"),
                uploaded_at=document_meta.get("uploaded_at"),
                knowledge_base_id=document_meta.get("knowledge_base_id", ""),
                workspace_id=document_meta.get("workspace_id", ""),
            ),
        )
        return doc_langchain
