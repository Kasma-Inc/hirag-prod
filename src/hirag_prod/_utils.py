import asyncio
import html
import json
import logging
import numbers
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Callable, Coroutine, Iterable, List, Optional, TypeVar

import boto3
import numpy as np
import tiktoken
from botocore.exceptions import ClientError
from dotenv import load_dotenv

logger = logging.getLogger("HiRAG")
ENCODER = None
load_dotenv("/chatbot/.env", override=True)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}..."
                        )
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None


def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if "." in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist


def extract_values_from_json(
    json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False
):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group("key").strip('"')  # Strip quotes from key
        value = match.group("value").strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")

    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")

    return prediction_json


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    # remove potential double quotes
    result = result.strip('"')
    return result


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"[Retrieval Time: {elapsed_time:.6f} seconds]")


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=chunk_key,
    )


T = TypeVar("T")


async def _limited_gather_with_factory(
    coro_factories: Iterable[Callable[[], Coroutine[Any, Any, T]]],
    limit: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> List[Optional[T]]:
    """Execute coroutine factories with concurrency limit and proper retry support.

    This is the recommended version for retry functionality.

    Args:
        coro_factories: Iterable of functions that create fresh coroutines
        limit: Maximum number of concurrent executions
        max_retries: Maximum number of retry attempts per task
        retry_delay: Base delay between retries (with exponential backoff)

    Returns:
        List of results, with None for permanently failed tasks
    """
    # TODO: Add adaptive concurrency based on system resources and task complexity
    sem = asyncio.Semaphore(limit)

    async def _worker(
        coro_factory: Callable[[], Coroutine[Any, Any, T]], task_id: int
    ) -> Optional[T]:
        """Execute a coroutine factory with retry logic."""
        async with sem:
            for attempt in range(max_retries):
                try:
                    # Create a fresh coroutine for each attempt
                    coro = coro_factory()
                    result = await coro
                    return result
                except Exception as e:
                    if attempt <= max_retries - 1:
                        delay = retry_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Task {task_id} failed (attempt {attempt + 1}/{max_retries}): "
                            f"{type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Task {task_id} failed permanently after {max_retries} attempts: "
                            f"{type(e).__name__}: {str(e)}"
                        )
            return None

    # Convert to list for indexing
    factory_list = list(coro_factories)

    # Create tasks for all factories
    tasks = [
        asyncio.create_task(_worker(factory, i))
        for i, factory in enumerate(factory_list)
    ]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ========================================================================
# Amazon S3 utils
# ========================================================================


# Upload files to s3
async def upload_file_to_s3(
    file_obj: str, upload_obj_type: str, upload_obj: str = None
) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")
    # If S3 object_name was not specified, use file_name
    if upload_obj is None:
        upload_obj = os.path.join(
            os.getenv("AWS_UPLOAD_PATH", None),
            upload_obj_type,
            os.path.basename(file_obj),
        )
    try:
        s3_client.upload_file(file_obj, aws_bucket_name, upload_obj)
        print(f"✅ Successfully uploaded {file_obj} to {upload_obj}")
    except ClientError as e:
        logger.error(e)
        return False
    return True


# List files in s3
async def list_s3_files(prefix: str = None) -> bool:
    """
    List files in an Amazon S3 bucket.

    Args:
        prefix (str): The prefix of the files to list.

    Returns:
        bool: True if the file list was successfully printed, False otherwise.
    """
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        if prefix is None:
            response = s3_client.list_objects_v2(Bucket=aws_bucket_name)
        else:
            response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=prefix)

        if "Contents" in response:
            print(f"========== S3 File List ({prefix}) ==========")
            for idx, item in enumerate(response["Contents"]):
                print(f"{idx+1}. {item['Key']}")
            print(f"========== End of S3 File List ({prefix}) ==========")
            return True
        else:
            print(f"No files found in {prefix}")
            return False
    except ClientError as e:
        logger.error(e)
        return False


# Download files from s3
async def download_s3_file(file_name: str, file_type: str, download_path: str) -> bool:
    """
    Download a file from an Amazon S3 bucket to a local path.

    Args:
        obj_name (str): The name of the object to download.
        local_path (str): The path to save the downloaded file.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.
    """
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    aws_obj_name = os.path.join(
        os.getenv("AWS_UPLOAD_PATH", None), file_type, file_name
    )
    local_obj_name = os.path.join(download_path, file_name)
    try:
        s3_client.download_file(aws_bucket_name, aws_obj_name, local_obj_name)
        print(f"✅ Successfully downloaded {aws_obj_name} to {local_obj_name}")
        return True
    except ClientError as e:
        logger.error(e)
        return False


# Delete files from s3
async def delete_s3_file(file_name: str, file_type: str) -> bool:
    """
    Delete a file from an Amazon S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The key (path) of the file to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    aws_obj_name = os.path.join(
        os.getenv("AWS_UPLOAD_PATH", None), file_type, file_name
    )

    try:
        s3_client.delete_object(Bucket=aws_bucket_name, Key=aws_obj_name)
        print(f"✅ Successfully deleted {aws_obj_name} from bucket {aws_bucket_name}")
        return True
    except ClientError as e:
        logger.error(f"Error deleting {aws_obj_name}: {e}")
        return False


if __name__ == "__main__":
    print("LIST S3 FILES:")
    if os.getenv("AWS_UPLOAD_PATH"):
        aws_upload_path = os.getenv("AWS_UPLOAD_PATH")
        asyncio.run(list_s3_files(aws_upload_path))

    if os.getenv("AWS_DOWNLOAD_PATH"):
        aws_download_path = os.getenv("AWS_DOWNLOAD_PATH")
        asyncio.run(list_s3_files(aws_download_path))
