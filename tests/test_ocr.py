import os

import pytest

from hirag_prod.loader.ocr import OCRClient


# Here we use the MonkeyOCR client to test the OCR client.
class TestMonkeyOCRClient:
    """MonkeyOCR client tests."""

    @pytest.fixture
    def client(self):
        """Create MonkeyOCR client."""
        return OCRClient()

    def test_parse_success(self, client):
        """Test successful parsing."""
        result = client.parse(
            "s3://monkeyocr/test/input/example.png",
            "s3://monkeyocr/test/output/example",
        )

        assert result["success"] is True
        assert "files" in result
        assert len(result["files"]) == 6

    def test_invalid_input_path(self, client):
        """Test invalid input path."""
        with pytest.raises(ValueError):
            client.parse("invalid_path", "s3://monkeyocr/test/output/example")

    def test_invalid_output_path(self, client):
        """Test invalid output path."""
        with pytest.raises(ValueError):
            client.parse("s3://monkeyocr/test/input/example.png", "invalid_path")


def test_client_initialization():
    """Test client initialization."""
    client = OCRClient()

    assert client.api_url == os.getenv("OCR_BASE_URL")
    assert client.model_name == os.getenv("OCR_MODEL_NAME")
    assert client.auth_token == os.getenv("OCR_AUTH_TOKEN")
