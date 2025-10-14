#! /usr/bin/env python3
from abc import ABC
from typing import Optional, Tuple, Type

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.loader import document_converter
from hirag_prod.schema import File, create_file


class BaseLoader(ABC):
    """Base class for all loaders"""

    loader_docling: Type[DocumentConverter]
    loader_langchain: Type[LangchainBaseLoader]

    def load_dots_ocr(
        self, document_path: str, document_meta: Optional[dict] = None
    ) -> Tuple[File, File]:
        """Load document and set the metadata of the split chunks

        Args:
            document_path (str): The document path for loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the loader.

        Returns:
            Tuple[File, File]: the loaded document
        """
        assert document_meta.get("private") is not None, "private is required"
        assert document_path.startswith("s3://") or document_path.startswith("oss://")
        workspace_id = document_meta.get("workspaceId", None)
        knowledge_base_id = document_meta.get("knowledgeBaseId", None)
        processed_doc = document_converter.convert(
            "dots_ocr",
            document_path,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )

        assert processed_doc is not None, "Failed to receive parsed document."

        json_doc = processed_doc.get("json", None)
        md_doc_raw = processed_doc.get("md", None)
        # md_nohf_doc_raw = processed_doc.get("md_nohf", None)

        # Convert md to File
        md_doc = create_file(
            metadata=document_meta,
            documentKey=document_meta.get(
                "documentKey", compute_mdhash_id(md_doc_raw, prefix="doc-")
            ),
            text=md_doc_raw,
            pageNumber=len(json_doc),
        )
        return json_doc, md_doc

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
        file_type = document_meta.get("type", None)
        if file_type == "md" or file_type == "markdown":
            # For markdown files, use export_to_text to better match original pattern
            md_str: str = docling_doc.export_to_text()
        else:
            md_str: str = docling_doc.export_to_markdown()
        doc_md = create_file(
            metadata=document_meta,
            documentKey=document_meta.get(
                "documentKey", compute_mdhash_id(md_str, prefix="doc-")
            ),
            text=md_str,
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
        doc_langchain = create_file(
            # Langchain Doc text stored in page_content
            metadata=document_meta,
            documentKey=document_meta.get(
                "documentKey",
                compute_mdhash_id(langchain_docs[0].page_content, prefix="doc-"),
            ),
            text=langchain_docs[0].page_content,
        )
        return doc_langchain
