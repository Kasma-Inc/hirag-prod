from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from hirag_prod.loader.base_loader import BaseLoader
from hirag_prod.loader.docling_cloud import DoclingCloudClient
from hirag_prod.loader.dots_ocr import DotsOCRClient


class PDFLoader(BaseLoader):
    """Loads PDF documents"""

    def __init__(self):
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
        )  # use more accurate TableFormer model
        self.loader_docling = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.loader_docling_cloud = DoclingCloudClient()
        self.loader_dots_ocr = DotsOCRClient()
