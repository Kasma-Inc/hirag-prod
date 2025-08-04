"""
Docling Cloud Service
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from docling_core.types.doc import DoclingDocument

from hirag_prod._utils import download_oss_file, download_s3_file

OUTPUT_DIR_PREFIX = "docling_cloud/output"


@dataclass
class DoclingCloudClient:
    """
    Client for Docling Cloud Service.

    This class provides methods to perform document processing using
    the Docling Cloud service with configurable input and output S3/OSS paths.
    """

    api_url: str = os.getenv("DOCLING_CLOUD_BASE_URL", None)
    auth_token: str = os.getenv("DOCLING_CLOUD_AUTH_TOKEN", None)
    model_name: str = os.getenv("DOCLING_CLOUD_MODEL_NAME", "docling")
    entry_point: str = os.getenv("DOCLING_CLOUD_ENTRY_POINT", "/ocr")
    timeout: int = int(os.getenv("DOCLING_CLOUD_TIMEOUT", 300))
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        """Validate required environment variables."""
        if not self.api_url:
            raise ValueError("DOCLING_CLOUD_BASE_URL must be set")
        if not self.auth_token:
            raise ValueError("DOCLING_CLOUD_AUTH_TOKEN must be set")

    def convert(self, input_file_path: str) -> DoclingDocument:
        """
        Convert a document using Docling Cloud Service and return DoclingDocument.

        Args:
            input_file_path: File path to the input document file

        Returns:
            DoclingDocument: The processed document

        Raises:
            requests.exceptions.RequestException: If the API request fails
            ValueError: If the input parameters are invalid
            FileNotFoundError: If the output JSON file is not found
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
            self.logger.info(f"Sending Docling Cloud request for {input_file_path}")

            response = requests.post(
                self.api_url, headers=headers, files=files, timeout=self.timeout
            )

            response.raise_for_status()

            self.logger.info(
                f"Docling Cloud request successful. Output saved to {output_path}"
            )

            json_file_path = f"{output_relative_path}/{file_name_without_ext}.json"

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as tmp_file:
                tmp_json_path = tmp_file.name

            try:
                if parsed_url.scheme == "s3":
                    flag = download_s3_file(bucket_name, json_file_path, tmp_json_path)
                    if not flag:
                        raise ValueError(
                            f"Failed to download {json_file_path} from {bucket_name}"
                        )
                elif parsed_url.scheme == "oss":
                    flag = download_oss_file(bucket_name, json_file_path, tmp_json_path)
                    if not flag:
                        raise ValueError(
                            f"Failed to download {json_file_path} from {bucket_name}"
                        )
                else:
                    raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")

                docling_doc = DoclingDocument.load_from_json(tmp_json_path)

                self.logger.info(
                    f"Successfully loaded DoclingDocument from {json_file_path}"
                )
                return docling_doc

            finally:
                if os.path.exists(tmp_json_path):
                    os.unlink(tmp_json_path)

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Docling Cloud API request failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to process document: {str(e)}")
            raise
