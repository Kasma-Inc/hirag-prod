"""
Dots OCR Service
"""

import logging
import os
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urlparse

import requests
from docling_core.types import DoclingDocument

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_document_converter_config
from hirag_prod.loader.utils import download_load_file, exists_cloud_file

logger: logging.Logger = logging.getLogger(__name__)

# TODO: Fix dots_ocr/ dir DNE problem, now using docling's as temp solution
OUTPUT_DIR_PREFIX = "docling_cloud/output"


def convert(
    converter_type: Literal["dots_ocr"],
    input_file_path: str,
    workspace_id: Optional[str] = None,
    knowledge_base_id: Optional[str] = None,
) -> Optional[Union[Dict[str, Any], DoclingDocument]]:
    """
    Convert a document using Dots OCR Service and return Parsed Document.

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

    entry_point = get_document_converter_config(converter_type).entry_point

    headers = {
        "Model-Name": get_document_converter_config(converter_type).model_name,
        "Entry-Point": entry_point,
        "Authorization": f"Bearer {get_document_converter_config(converter_type).api_key}",
    }

    if entry_point == "/parse/file":
        if not workspace_id or not knowledge_base_id:
            raise ValueError(
                "workspace_id and knowledge_base_id are required for /parse/file endpoint"
            )
        files = {
            "input_s3_path": (None, input_file_path),
            "output_s3_path": (None, output_path),
            "workspaceId": (None, workspace_id) if workspace_id else (None, ""),
            "knowledgebaseId": (
                (None, knowledge_base_id) if knowledge_base_id else (None, "")
            ),
        }
    else:
        files = {
            "input_s3_path": (None, input_file_path),
            "output_s3_path": (None, output_path),
        }

    try:
        logger.info(f"Sending document conversion request for {input_file_path}")

        # verify that input s3 path exists
        if parsed_url.scheme in ["s3", "oss"]:
            if not exists_cloud_file(parsed_url.scheme, bucket_name, file_path):
                logger.error(
                    f"Input {parsed_url.scheme.upper()} path does not exist: {input_file_path}"
                )
                return None
        else:
            raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")

        response = requests.post(
            get_document_converter_config(converter_type).base_url,
            headers=headers,
            files=files,
            timeout=get_document_converter_config(converter_type).timeout,
        )

        response.raise_for_status()

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
                "json": download_load_file(
                    "json", "dict", parsed_url, bucket_name, json_file_path
                ),
                "md": download_load_file(
                    "md", "dict", parsed_url, bucket_name, md_file_path
                ),
                "md_nohf": download_load_file(
                    "md", "dict", parsed_url, bucket_name, md_nohf_file_path
                ),
            }
        else:
            return download_load_file(
                "json", "docling_document", parsed_url, bucket_name, json_file_path
            )

    except requests.exceptions.Timeout as e:
        log_error_info(
            logging.ERROR,
            f"Request timeout after {get_document_converter_config(converter_type).timeout} seconds",
            e,
            raise_error=True,
        )
    except requests.exceptions.RequestException as e:
        log_error_info(
            logging.ERROR,
            f"Document conversion API request failed",
            e,
            raise_error=True,
        )
    except Exception as e:
        log_error_info(
            logging.ERROR, f"Failed to process document", e, raise_error=True
        )
