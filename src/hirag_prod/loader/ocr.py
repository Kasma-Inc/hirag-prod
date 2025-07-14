"""
OCR API Integration
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class OCRClient:
    """
    Client for OCR API integration.

    This class provides methods to perform OCR operations on images
    using the OCR service with configurable input and output paths.
    """

    api_url: str = os.getenv("OCR_BASE_URL", None)
    model_name: str = os.getenv("OCR_MODEL_NAME", None)
    auth_token: str = os.getenv("OCR_AUTH_TOKEN", None)
    entry_point: str = os.getenv("OCR_ENTRY_POINT", "/parse")
    timeout: int = int(os.getenv("OCR_TIMEOUT", 120))
    logger: logging.Logger = logging.getLogger(__name__)

    def parse(self, input_s3_path: str, output_s3_path: str) -> Dict[str, Any]:
        """
        Parse a pdf or image file using OCR API.

        Args:
            input_s3_path: S3 path to the input pdf or image file
            output_s3_path: S3 path where the output should be stored

        Returns:
            Dict containing the API response

        Raises:
            requests.exceptions.RequestException: If the API request fails
            ValueError: If the input parameters are invalid
        """
        # Validate input parameters
        if not input_s3_path or not input_s3_path.startswith("s3://"):
            raise ValueError(
                "input_s3_path must be a valid S3 path starting with 's3://'"
            )

        if not output_s3_path or not output_s3_path.startswith("s3://"):
            raise ValueError(
                "output_s3_path must be a valid S3 path starting with 's3://'"
            )

        # Prepare headers
        headers = {
            "Model-Name": self.model_name,
            "Entry-Point": self.entry_point,
            "Authorization": f"Bearer {self.auth_token}",
        }

        # Prepare data payload
        data = {"input_s3_path": input_s3_path, "output_s3_path": output_s3_path}

        try:
            self.logger.info(f"Sending OCR request for {input_s3_path}")

            # Make the API request
            response = requests.post(
                self.api_url, headers=headers, data=data, timeout=self.timeout
            )

            # Raise an exception for bad status codes
            response.raise_for_status()

            self.logger.info(
                f"OCR request successful. Output saved to {output_s3_path}"
            )

            # Try to parse JSON response, fallback to text if not JSON
            try:
                return response.json()
            except ValueError:
                return {"response": response.text, "status_code": response.status_code}

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OCR API request failed: {str(e)}")
            raise


def __post_init__(self):
    if not self.api_url:
        raise ValueError("OCR_BASE_URL must be set")
    if not self.model_name:
        raise ValueError("OCR_MODEL_NAME must be set")
    if not self.auth_token:
        raise ValueError("OCR_AUTH_TOKEN must be set")
