import json
import warnings

from langchain_community import document_loaders
from pptagent.document import Document
from pptagent.model_utils import ModelManager
from pptagent.utils import pjoin

from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.loader.markify_loader import markify_client

# Suppress PyPDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


class PDFLoader(BaseLoader):
    """Loads PDF documents"""

    def __init__(self, work_dir: str, max_output_docs: int = 5):
        self.loader_type = document_loaders.PyPDFLoader
        self.loader_markify = markify_client
        self.models = ModelManager()
        self.work_dir = work_dir
        self.max_output_docs = max_output_docs

    def refine_markdown(self, markdown_text: str) -> Document:
        """
        Refine the markdown text structure with language model assistance

        Args:
            markdown_text (str): The extracted text content from the PDF

        Returns:
            Document: The refined mardown text
        """
        pdf_dir = pjoin(self.work_dir, "pdf")
        refined_md = Document.from_markdown(
            markdown_text,
            self.models.language_model,
            self.models.vision_model,
            pdf_dir,
        )

        # Save the refined document
        json.dump(
            refined_md.to_dict(),
            open(pjoin(pdf_dir, "refined_doc.json"), "w"),
            ensure_ascii=False,
            indent=4,
        )

        return refined_md
