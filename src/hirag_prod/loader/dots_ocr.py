"""
Dots OCR Service
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Literal
from urllib.parse import urlparse

import requests

from hirag_prod._utils import download_oss_file, download_s3_file, exists_s3_file

# TODO: Fix dots_ocr/ dir DNE problem, now using docling's as temp solution
OUTPUT_DIR_PREFIX = "docling_cloud/output"


@dataclass
class DotsOCRClient:
    """
    Client for Dots OCR Service.

    This class provides methods to perform document processing using
    the Dots OCR service with configurable input and output S3/OSS paths.
    """

    api_url: str = os.getenv("DOTS_OCR_BASE_URL", None)
    auth_token: str = os.getenv("DOTS_OCR_AUTH_TOKEN", None)
    model_name: str = os.getenv("DOTS_OCR_MODEL_NAME", "DotsOCR")
    entry_point: str = os.getenv("DOTS_OCR_ENTRY_POINT", "parse/file")
    timeout: int = int(os.getenv("DOTS_OCR_TIMEOUT", 300))
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        """Validate required environment variables."""
        if not self.api_url:
            raise ValueError("DOTS_OCR_BASE_URL must be set")
        if not self.auth_token:
            raise ValueError("DOTS_OCR_AUTH_TOKEN must be set")

    def _download_load_file(
        self,
        parsed_url: urlparse,
        bucket_name: str,
        file_path: str,
        file_type: Literal["json", "md"],
    ) -> dict:

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=f".{file_type}", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            if parsed_url.scheme == "s3":
                flag = download_s3_file(bucket_name, file_path, tmp_path)
                if not flag:
                    raise ValueError(
                        f"Failed to download {file_path} from {bucket_name}"
                    )
            elif parsed_url.scheme == "oss":
                flag = download_oss_file(bucket_name, file_path, tmp_path)
                if not flag:
                    raise ValueError(
                        f"Failed to download {file_path} from {bucket_name}"
                    )
            else:
                raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")

            with open(tmp_path, "r") as f:
                if file_type == "json":
                    parsed_doc = json.load(f)
                elif file_type == "md":
                    parsed_doc = f.read()
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")

            self.logger.info(f"Successfully loaded document from {file_path}")
            return parsed_doc

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def convert(self, input_file_path: str) -> Dict[str, Any]:
        """
        Convert a document using Dots OCR Service and return Parsed Document.

        Args:
            input_file_path: File path to the input document file

        Returns:
            ParsedDocument: The processed document

        Raises:
            requests.exceptions.RequestException: If the API request fails
            ValueError: If the input parameters are invalid
            FileNotFoundError: If the output JSON file is not found

            ParsedDocument: [{page_no: int, full_layout_info: [{bbox:[int, int, int, int], category: str, text: str}, ...boxes]}, ...pages ]
            Possible types: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
        """

        if not input_file_path:
            raise ValueError("input_file_path is required")

        parsed_url = urlparse(input_file_path)
        bucket_name = parsed_url.netloc
        file_path = parsed_url.path.lstrip("/")
        file_name = os.path.basename(file_path)

        file_name_without_ext = os.path.splitext(file_name)[0]
        output_relative_path = f"{OUTPUT_DIR_PREFIX}/{file_name_without_ext}"
        output_path = f"{parsed_url.scheme}://{bucket_name}/{OUTPUT_DIR_PREFIX}/{file_name_without_ext}"

        headers = {
            "Model-Name": self.model_name,
            "Entry-Point": self.entry_point,
            "Authorization": f"Bearer {self.auth_token}",
        }

        files = {
            "input_s3_path": (None, input_file_path),
            "output_s3_path": (None, output_path),
        }

        try:
            self.logger.info(f"Sending Dots OCR request for {input_file_path}")
            # Testing Only
            self.logger.info(f"Request headers: {headers}")
            self.logger.info(f"Request files: {files}")

            # verify that input s3 path exists
            if not exists_s3_file(file_path):
                self.logger.error(f"Input S3 path does not exist: {input_file_path}")
                return None

            response = requests.post(
                self.api_url, headers=headers, files=files, timeout=self.timeout
            )

            response.raise_for_status()

            self.logger.info(
                f"Dots OCR request successful. Output saved to {output_path}"
            )

            # json: <output_relative_path>/<file_name_without_ext>.json
            # md: <output_relative_path>/<file_name_without_ext>.md
            # md_nohf: <output_relative_path>/<file_name_without_ext>.md
            json_file_path = f"{output_relative_path}/{file_name_without_ext}.json"
            md_file_path = f"{output_relative_path}/{file_name_without_ext}.md"
            md_nohf_file_path = (
                f"{output_relative_path}/{file_name_without_ext}_nohf.md"
            )

            return_files = {}

            return_files["json"] = self._download_load_file(
                parsed_url, bucket_name, json_file_path, "json"
            )
            return_files["md"] = self._download_load_file(
                parsed_url, bucket_name, md_file_path, "md"
            )
            return_files["md_nohf"] = self._download_load_file(
                parsed_url, bucket_name, md_nohf_file_path, "md"
            )

            return return_files

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Dots OCR API request failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to process document: {str(e)}")
            raise
