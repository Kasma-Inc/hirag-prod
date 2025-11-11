"""
Dots OCR Service
"""

import logging
import os
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urlparse

from docling_core.types import DoclingDocument
from resources.ocr_client import OCR
from utils.logging_utils import log_error_info
from utils.sync_function_utils import run_sync_function_using_thread
from pydantic import BaseModel

from hirag_prod.loader.utils import download_load_file, exists_cloud_file
from hirag_prod.tracing import traced
from hirag_prod.usage import ModelIdentifier, ModelUsage, UsageRecorder

logger: logging.Logger = logging.getLogger(__name__)

# TODO: Fix dots_ocr/ dir DNE problem, now using docling's as temp solution
OUTPUT_DIR_PREFIX = "docling_cloud/output"


@traced()
async def convert(
    converter_type: Literal["dots_ocr"],
    input_file_path: str,
    workspace_id: Optional[str] = None,
    knowledge_base_id: Optional[str] = None,
) -> Optional[Union[Dict[str, Any], DoclingDocument]]:
    """
    Convert a document using Dots OCR Service and return Parsed Document.

    Supports both synchronous and asynchronous processing:
    - Synchronous: Direct response with processed document
    - Asynchronous: Job submission with polling for completion

    Args:
        input_file_path: File path to the input document file
        converter_type: Type of converter to use.
        knowledge_base_id: Knowledge Base ID for the document (required for /parse/file endpoint)
        workspace_id: Workspace ID for the document (required for /parse/file endpoint)

    Returns:
        ParsedDocument: The processed document

    Raises:
        requests.exceptions.RequestException: If the API request fails
        ValueError: If the input parameters are invalid
        FileNotFoundError: If the output JSON file is not found

        ParsedDocument: [{page_no: int, full_layout_info: [{bbox:[int, int, int, int], category: str, text: str}, ...boxes]}, ...pages ]
        Possible types: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
    """

    parsed_url = urlparse(input_file_path)
    bucket_name = parsed_url.netloc
    file_path = parsed_url.path.lstrip("/")
    file_name = os.path.basename(file_path)

    file_name_without_ext = os.path.splitext(file_name)[0]
    output_relative_path = f"{OUTPUT_DIR_PREFIX}/{file_name_without_ext}"
    output_path = f"{parsed_url.scheme}://{bucket_name}/{OUTPUT_DIR_PREFIX}/{file_name_without_ext}"

    try:
        logger.info(f"Sending document conversion request for {input_file_path}")

        # verify that input s3 path exists
        if parsed_url.scheme in ["s3", "oss"]:
            if not exists_cloud_file(parsed_url.scheme, bucket_name, file_path):
                log_error_info(
                    logging.ERROR,
                    f"Input {parsed_url.scheme.upper()} path does not exist: {input_file_path}",
                    None,
                )
                return None
        else:
            raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")

        await OCR().convert(
            input_file_path, output_path, workspace_id, knowledge_base_id
        )

        logger.info(
            f"Document conversion request successful. Output saved to {output_path}"
        )

        json_file_path = f"{output_relative_path}/{file_name_without_ext}.json"
        if converter_type == "dots_ocr":
            md_file_path = f"{output_relative_path}/{file_name_without_ext}.md"
            md_nohf_file_path = (
                f"{output_relative_path}/{file_name_without_ext}_nohf.md"
            )

            return {
                "json": await run_sync_function_using_thread(
                    download_load_file,
                    "json",
                    "dict",
                    parsed_url,
                    bucket_name,
                    json_file_path,
                ),
                "md": await run_sync_function_using_thread(
                    download_load_file,
                    "md",
                    "dict",
                    parsed_url,
                    bucket_name,
                    md_file_path,
                ),
                "md_nohf": await run_sync_function_using_thread(
                    download_load_file,
                    "md",
                    "dict",
                    parsed_url,
                    bucket_name,
                    md_nohf_file_path,
                ),
            }
        else:
            return await run_sync_function_using_thread(
                download_load_file,
                "json",
                "docling_document",
                parsed_url,
                bucket_name,
                json_file_path,
            )

    except Exception as e:
        log_error_info(
            logging.ERROR, f"Failed to process document", e, raise_error=True
        )
