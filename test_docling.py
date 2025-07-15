# supported input formats: PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, HTML, XHTML, CSV
#                    Images format: PNG, JPEG, TIFF, BMP, WEBP
# supported output formats: HTML(Both image embedding  and referencing are supported), Markdown, JSON (Lossless serialization of Docling Document), Text, Doctags

# I. Content Items (all belong to DocItem)
## texts, tables, pictures, key-value items

# II. Content Structure
## body, furniture, groups

import json
import os
import time
from enum import Enum

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc import DocItemLabel

os.makedirs("tests/output", exist_ok=True)

from enum import Enum


class ChunkType(Enum):
    """Enumeration of chunk types based on DocItem labels."""

    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    CODE = "code"
    CAPTION = "caption"
    FORMULA = "formula"
    PICTURE = "picture"
    CHART = "chart"
    FOOTNOTE = "footnote"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    DOCUMENT_INDEX = "document_index"
    CHECKBOX = "checkbox"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"
    GRADING_SCALE = "grading_scale"
    HANDWRITTEN_TEXT = "handwritten_text"
    EMPTY_VALUE = "empty_value"
    REFERENCE = "reference"
    MIXED = "mixed"
    UNKNOWN = "unknown"


LABEL_TO_CHUNK_TYPE = {
    DocItemLabel.TEXT: ChunkType.TEXT,
    DocItemLabel.TABLE: ChunkType.TABLE,
    DocItemLabel.LIST_ITEM: ChunkType.LIST,
    DocItemLabel.TITLE: ChunkType.TITLE,
    DocItemLabel.SECTION_HEADER: ChunkType.SECTION_HEADER,
    DocItemLabel.PARAGRAPH: ChunkType.PARAGRAPH,
    DocItemLabel.CODE: ChunkType.CODE,
    DocItemLabel.CAPTION: ChunkType.CAPTION,
    DocItemLabel.FORMULA: ChunkType.FORMULA,
    DocItemLabel.PICTURE: ChunkType.PICTURE,
    DocItemLabel.CHART: ChunkType.CHART,
    DocItemLabel.FOOTNOTE: ChunkType.FOOTNOTE,
    DocItemLabel.PAGE_FOOTER: ChunkType.PAGE_FOOTER,
    DocItemLabel.PAGE_HEADER: ChunkType.PAGE_HEADER,
    DocItemLabel.DOCUMENT_INDEX: ChunkType.DOCUMENT_INDEX,
    DocItemLabel.CHECKBOX_SELECTED: ChunkType.CHECKBOX,
    DocItemLabel.CHECKBOX_UNSELECTED: ChunkType.CHECKBOX,
    DocItemLabel.FORM: ChunkType.FORM,
    DocItemLabel.KEY_VALUE_REGION: ChunkType.KEY_VALUE_REGION,
    DocItemLabel.GRADING_SCALE: ChunkType.GRADING_SCALE,
    DocItemLabel.HANDWRITTEN_TEXT: ChunkType.HANDWRITTEN_TEXT,
    DocItemLabel.EMPTY_VALUE: ChunkType.EMPTY_VALUE,
    DocItemLabel.REFERENCE: ChunkType.REFERENCE,
}


def determine_chunk_type(chunk) -> ChunkType:
    """
    Determine the type of a chunk based on its doc_items.

    Args:
        chunk: A DocChunk object containing doc_items with labels

    Returns:
        ChunkType: The determined type of the chunk
    """
    if not chunk.meta.doc_items:
        return ChunkType.UNKNOWN

    # Get all unique labels from doc_items (using DocItemLabel directly)
    labels = {item.label for item in chunk.meta.doc_items}

    # If only one type of item, return that type
    if len(labels) == 1:
        label = labels.pop()
        return LABEL_TO_CHUNK_TYPE.get(label, ChunkType.UNKNOWN)

    # If multiple types, return MIXED
    return ChunkType.MIXED


def format_chunk_with_type(chunk) -> str:
    """
    Format a chunk with its type prefix.

    Args:
        chunk: A DocChunk object

    Returns:
        str: Formatted chunk text with type prefix
    """
    chunk_type = determine_chunk_type(chunk)
    return f"[{chunk_type.value.upper()}] {chunk.text}"


doc_converter = DocumentConverter()
chunker = HierarchicalChunker()

# ================================ PDF ================================
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = (
    TableFormerMode.ACCURATE
)  # use more accurate TableFormer model

source = "tests/test_files/Guide-to-U.S.-Healthcare-System.pdf"  # document per local path or URL
pdf_doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

start_time = time.time()
print(f"Start to convert: {source}")

doc = pdf_doc_converter.convert(source).document

end_time = time.time()
elapsed_time = end_time - start_time

print(f"PDF Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_pdf.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_pdf.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_pdf_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ DOCX ================================
source = "tests/test_files/word_sample.docx"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")

doc = doc_converter.convert(source).document

end_time = time.time()
elapsed_time = end_time - start_time

print(f"DOCX Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_docx.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_docx.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_docx_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ XLSX ================================
source = "tests/test_files/sample_sales_data.xlsm"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")
doc = doc_converter.convert(source).document
end_time = time.time()
elapsed_time = end_time - start_time
print(f"XLSX Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_xlsx.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_xlsx.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_xlsx_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ PPTX ================================
source = "tests/test_files/Beamer.pptx"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")
doc = doc_converter.convert(source).document
end_time = time.time()
elapsed_time = end_time - start_time
print(f"PPTX Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_pptx.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_pptx.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_pptx_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ Markdown ================================
source = "tests/test_files/fresh_wiki_article.md"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")
doc = doc_converter.convert(source).document
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Markdown Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_markdown.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_markdown.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_markdown_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ HTML ================================
source = "tests/test_files/wiki_labubu.html"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")
doc = doc_converter.convert(source).document
end_time = time.time()
elapsed_time = end_time - start_time
print(f"HTML Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_html.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_html.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_html_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))


# ================================ CSV ================================
source = "tests/test_files/csv-comma.csv"  # document per local path or URL

start_time = time.time()
print(f"Start to convert: {source}")
doc = doc_converter.convert(source).document
end_time = time.time()
elapsed_time = end_time - start_time
print(f"CSV Conversion completed! Time taken: {elapsed_time:.2f} seconds")
with open("tests/output/test_csv.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
with open("tests/output/test_csv.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(doc.export_to_dict()))
chunked_texts = [format_chunk_with_type(chunk) for chunk in chunker.chunk(doc)]
with open("tests/output/test_csv_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunked_texts))
