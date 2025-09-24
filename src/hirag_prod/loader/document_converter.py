"""
Dots OCR Service
"""

import logging
import os
import time
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


def _poll_dots_job_status(
    job_id: str,
    timeout: int = 300,
    retries: int = 3,
    polling_interval: Optional[int] = None,
) -> bool:
    """
    Poll job status until completion or timeout with retry logic.

    Args:
        job_id: Job ID to poll
        timeout: Maximum time to wait in seconds
        retries: Maximum number of consecutive failures before giving up
        polling_interval: Time between polls in seconds (uses config default if None)

    Returns:
        bool: True if job completed successfully, False otherwise
    """
    config = get_document_converter_config("dots_ocr")
    base_url = config.base_url
    api_key = config.api_key

    if polling_interval is None:
        polling_interval = config.polling_interval

    status_url = f"{base_url.rstrip('/')}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Model-Name": config.model_name,
        "Entry-Point": "/status",
    }
    data = {
        "OCRJobId": job_id,
    }

    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = retries

    while time.time() - start_time < timeout:
        try:
            response = requests.post(status_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()

            # Reset failure counter on successful request
            consecutive_failures = 0

            status_data = response.json()
            status = status_data.get("status", "").lower()

            logger.info(f"Job {job_id} status: {status}")

            # "pending", "retrying", "processing", "completed", "failed", "canceled"
            if status == "completed":
                return True
            elif status in ["failed", "error", "cancelled"]:
                logger.error(f"Job {job_id} failed with status: {status}")
                return False

            # Job still running, wait before next poll
            time.sleep(polling_interval)

        # To avoid network issues causing immediate failure, but not a perfect solution
        except requests.exceptions.RequestException as e:
            consecutive_failures += 1
            logger.warning(
                f"Failed to check job status (attempt {consecutive_failures}/{max_consecutive_failures}): {e}"
            )

            if consecutive_failures >= max_consecutive_failures:
                log_error_info(
                    logging.ERROR,
                    f"Max consecutive failures ({max_consecutive_failures}) reached, stopping polling",
                    e,
                    raise_error=True,
                )
                return False

            time.sleep(polling_interval)
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                f"Unexpected error checking job status (attempt {consecutive_failures}/{max_consecutive_failures}): {e}"
            )

            if consecutive_failures >= max_consecutive_failures:
                log_error_info(
                    logging.ERROR,
                    f"Max consecutive failures ({max_consecutive_failures}) reached, stopping polling",
                    e,
                    raise_error=True,
                )
                return False

            time.sleep(polling_interval)

    logger.error(f"Job {job_id} polling timed out after {timeout} seconds")
    return False


def convert(
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

    entry_point = get_document_converter_config(converter_type).entry_point

    headers = {
        "Model-Name": get_document_converter_config(converter_type).model_name,
        "Entry-Point": entry_point,
        "Authorization": f"Bearer {get_document_converter_config(converter_type).api_key}",
    }

    if entry_point == "/parse/file" or entry_point == "parse/file":
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

        if entry_point == "/parse/file" or entry_point == "parse/file":
            response_data = response.json()
            job_id = response_data.get("OCRJobId", None)

            if job_id:
                # Asynchronous processing - poll for completion
                logger.info(f"Document conversion job submitted with ID: {job_id}")

                # Poll job status with the configured timeout
                if not _poll_dots_job_status(
                    job_id,
                    timeout=get_document_converter_config(converter_type).timeout,
                    retries=get_document_converter_config(
                        converter_type
                    ).polling_retries,
                ):
                    logger.error(f"Job {job_id} did not complete successfully")
                    return None

                logger.info(f"Job {job_id} completed successfully")

            else:
                raise ValueError("No job ID found in the response for async processing")

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
