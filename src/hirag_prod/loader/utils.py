import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Literal, Union
from urllib.parse import ParseResult, urlparse

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError
from configs.functions import get_cloud_storage_config, initialize_config_manager
from docling_core.types import DoclingDocument
from utils.logging_utils import log_error_info

from hirag_prod.tracing import traced

S3_DOWNLOAD_DIR = "./files/s3"
OSS_DOWNLOAD_DIR = "./files/oss"

logger: logging.Logger = logging.getLogger(__name__)


@traced()
def download_load_file(
    file_type: Literal["json", "md"],
    return_type: Literal["dict", "docling_document"],
    parsed_url: ParseResult,
    bucket_name: str,
    file_path: str,
) -> Union[Dict, DoclingDocument]:

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=f".{file_type}", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        if parsed_url.scheme in ["s3", "oss"]:
            flag = download_cloud_file(
                parsed_url.scheme, bucket_name, file_path, tmp_path
            )
            if not flag:
                raise ValueError(f"Failed to download {file_path} from {bucket_name}")
        else:
            raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")

        with open(tmp_path, "r") as f:
            if file_type == "json":
                if return_type == "dict":
                    parsed_doc = json.load(f)
                else:
                    parsed_doc = DoclingDocument.from_json(f.read())
            else:
                if return_type == "docling_document":
                    raise ValueError(
                        "File type 'md' dose not support return type 'docling_document'"
                    )
                parsed_doc = f.read()

        logger.info(f"Successfully loaded document from {file_path}")
        return parsed_doc

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================================================================
# Cloud storage utils
# ========================================================================


def create_s3_client(storage_type: Literal["s3", "oss"]) -> BaseClient:
    cloud_storage_config = get_cloud_storage_config()
    return boto3.client(
        "s3",
        aws_access_key_id=cloud_storage_config.access_key_id,
        aws_secret_access_key=cloud_storage_config.secret_access_key.get_secret_value(),
        region_name=cloud_storage_config.region,
        endpoint_url=cloud_storage_config.endpoint,
        config=Config(
            s3={"addressing_style": "virtual"},
            signature_version="v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def exists_cloud_file(
    storage_type: Literal["s3", "oss"], bucket_name: str, cloud_file_path: str
) -> bool:
    s3_client: BaseClient = create_s3_client(storage_type)
    try:
        s3_client.head_object(Bucket=bucket_name, Key=cloud_file_path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        log_error_info(logging.ERROR, f"Failed to check file existence", e)
        return False


# List files in s3
def list_cloud_files(storage_type: Literal["s3", "oss"], prefix: str = None) -> bool:
    """
    List files on the cloud.

    Args:
        storage_type (Literal["s3", "oss"]): The type of cloud storage.
        prefix (str): The prefix of the files to list.

    Returns:
        bool: True if the file list was successfully printed, False otherwise.
    """
    s3_client: BaseClient = create_s3_client(storage_type)
    try:
        if prefix is None:
            response = s3_client.list_objects_v2(
                Bucket=get_cloud_storage_config().bucket_name
            )
        else:
            response = s3_client.list_objects_v2(
                Bucket=get_cloud_storage_config().bucket_name, Prefix=prefix
            )

        if "Contents" in response:
            print(f"========== Cloud File List ({prefix}) ==========")
            for idx, item in enumerate(response["Contents"]):
                print(f"{idx+1}. {item['Key']}")
            print(f"========== End of Cloud File List ({prefix}) ==========")
            return True
        else:
            print(f"No files found in {prefix}")
            return False
    except ClientError as e:
        log_error_info(logging.ERROR, f"Failed to list cloud files", e)
        return False


# Download files from s3
def download_cloud_file(
    storage_type: Literal["s3", "oss"],
    bucket_name: str,
    cloud_file_path: str,
    download_file_path: str,
) -> bool:
    """
    Download a file from a cloud bucket to a local path.

    Args:
        storage_type (Literal["s3", "oss"]): The type of cloud storage.
        bucket_name (str): The name of the cloud bucket.
        cloud_file_path (str): The path of the file on the cloud bucket.
        download_file_path (str): The local path to save the downloaded file.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.
    """
    s3_client: BaseClient = create_s3_client(storage_type)
    try:
        s3_client.download_file(bucket_name, cloud_file_path, download_file_path)
        logger.info(
            f"✅ Successfully downloaded {cloud_file_path} to {download_file_path}"
        )
        return True
    except ClientError as e:
        log_error_info(logging.ERROR, f"Failed to download cloud file", e)
        return False


# ========================================================================
# File path router
# ========================================================================


def route_file_path(url_path: str) -> str:
    """
    Parse a url path to a located file path
    """
    parsed_url = urlparse(url_path)
    if parsed_url.scheme == "file":
        local_file_path = parsed_url.path
        return local_file_path

    bucket_name = parsed_url.netloc
    file_path = parsed_url.path.lstrip("/")
    file_name = os.path.basename(file_path)

    if parsed_url.scheme == "s3":
        local_file_path = os.path.join(S3_DOWNLOAD_DIR, file_name)
        os.makedirs(S3_DOWNLOAD_DIR, exist_ok=True)
        flag = download_cloud_file("s3", bucket_name, file_path, local_file_path)
        if not flag:
            raise ValueError(f"Failed to download {file_path} from {bucket_name}")
        return local_file_path
    elif parsed_url.scheme == "oss":
        local_file_path = os.path.join(OSS_DOWNLOAD_DIR, file_name)
        os.makedirs(OSS_DOWNLOAD_DIR, exist_ok=True)
        flag = download_cloud_file("oss", bucket_name, file_path, local_file_path)
        if not flag:
            raise ValueError(f"Failed to download {file_path} from {bucket_name}")
        return local_file_path
    else:
        raise ValueError(f"Unsupported scheme: '{parsed_url.scheme}'")


def validate_document_path(document_path: str) -> None:
    """Validate input parameters"""
    if not document_path or not isinstance(document_path, str):
        raise ValueError("document_path must be a non-empty string")

    if not Path(document_path).exists():
        raise FileNotFoundError(f"Document not found: {document_path}")


if __name__ == "__main__":
    initialize_config_manager()
    print("========== LIST S3 FILES ==========")
    list_cloud_files("s3")
    print("===================================")
    print("========== LIST OSS FILES ==========")
    list_cloud_files("oss")
