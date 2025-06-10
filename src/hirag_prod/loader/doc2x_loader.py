import os
import zipfile
from pathlib import Path
from typing import List

import tiktoken
from dotenv import load_dotenv
from pdfdeal import Doc2X
from pdfdeal.file_tools import auto_split_mds

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.schema import File, FileMetadata

load_dotenv()

MD_SEPARATOR = "=+=+=+=+=+=+=+=+="


class Doc2XClient:
    def __init__(
        self, output_dir: str = "converted_docs", debug: bool = True, timeout: int = 30
    ):
        api_key = os.getenv("DOC2X_API_KEY")
        if not api_key:
            raise ValueError("DOC2X_API_KEY environment variable is not set")
        self.client = Doc2X(apikey=api_key, debug=debug)
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def split_text_by_tokens(
        self, text: str, max_tokens: int = 8192, separator: str = MD_SEPARATOR
    ) -> List[str]:
        """Split by separator first, then split by token limit if exceeding"""
        encoding = tiktoken.get_encoding("cl100k_base")

        # Split by separator
        sections = text.split(separator)
        chunks = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            tokens = encoding.encode(section)

            # If section is not exceeding the limit, add it directly
            if len(tokens) <= max_tokens:
                chunks.append(section)
            else:
                # Split by tokens if exceeding the limit
                for i in range(0, len(tokens), max_tokens):
                    chunk_tokens = tokens[i : i + max_tokens]
                    chunks.append(encoding.decode(chunk_tokens))

        return chunks

    def convert_and_split_pdf(
        self, file_path: str, max_tokens: int = 8000
    ) -> List[File]:
        """Convert PDF to markdown and split into chunks while preserving images."""
        file_path = Path(file_path)
        base_name = file_path.stem

        # Create document-specific output directory
        doc_output_dir = self.output_dir / base_name
        doc_output_dir.mkdir(exist_ok=True)

        # Convert PDF
        zip_paths, failed_info, flag = self.client.pdf2file(
            pdf_file=str(file_path),
            output_path=str(doc_output_dir),
            output_names=[f"{base_name}.zip"],
            output_format="md",
        )

        if flag:
            raise RuntimeError(f"Doc2X conversion failed: {failed_info}")

        # Extract zip contents
        with zipfile.ZipFile(zip_paths[0], "r") as zip_ref:
            zip_ref.extractall(doc_output_dir)

        # Remove zip file after extraction
        os.remove(zip_paths[0])

        # Find and rename output.md
        output_md = doc_output_dir / "output.md"
        target_md = doc_output_dir / f"{base_name}.md"

        if output_md.exists():
            output_md.rename(target_md)
        else:
            raise RuntimeError("No output.md found in converted files")

        auto_split_mds(mdpath=doc_output_dir, out_type="replace")

        # Read markdown content
        with open(target_md, "r", encoding="utf-8") as f:
            md_content = f.read().strip()

        if not md_content:
            raise RuntimeError("Markdown file is empty")

        chunks = self.split_text_by_tokens(md_content, max_tokens=max_tokens)
        files = []

        for i, chunk in enumerate(chunks, start=1):
            file = File(
                id=compute_mdhash_id(chunk.strip(), prefix="doc-"),
                page_content=chunk,
                metadata=FileMetadata(page_number=i),
            )
            files.append(file)

        return files


# Initialize client
doc2x_client = Doc2XClient(output_dir="loaded_docs", debug=True)
