import asyncio
import html
import json
import logging
import numbers
import os
import re
from functools import wraps
from hashlib import md5
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    List,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
)

import numpy as np
import tiktoken
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from hirag_prod.configs.functions import get_config_manager, get_hi_rag_config

logger = logging.getLogger("HiRAG")
ENCODER = None
load_dotenv("/chatbot/.env")

T = TypeVar("T")


def log_error_info(
    level: int,
    message: str,
    error: BaseException,
    debug_only: bool = False,
    exc_info: Optional[bool] = None,
    raise_error: bool = False,
    new_error_class: Type[T] = None,
):
    if (not debug_only) or get_config_manager().debug:
        logger.log(
            level,
            f"{message}: {error}",
            exc_info=get_config_manager().debug if exc_info is None else exc_info,
        )
    if raise_error:
        raise new_error_class(message) if new_error_class else error


def retry_async(
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
):
    """Async retry decorator"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _max_retries = (
                max_retries
                if max_retries is not None
                else get_hi_rag_config().max_retries
            )
            _delay = delay if delay is not None else get_hi_rag_config().retry_delay

            last_exception = None
            for attempt in range(_max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == _max_retries - 1:
                        break
                    log_error_info(
                        logging.WARNING,
                        f"Attempt {attempt + 1} failed: {e}, retrying in {_delay}s",
                        e,
                    )
                    await asyncio.sleep(_delay)
            raise last_exception

        return wrapper

    return decorator


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        # If in a sub-thread, create a new event loop.
        log_error_info(logging.INFO, "Creating a new event loop in a sub-thread.", e)
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
                stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        log_error_info(
                            logging.ERROR,
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...",
                            e,
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
        except ValueError as e:
            # If conversion fails, return the value as-is (likely a string)
            log_error_info(
                logging.INFO, f"Failed to convert string: {value}", e, debug_only=True
            )
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
AsyncEmbeddingFunction: TypeAlias = Callable[[list[str]], Awaitable[np.ndarray]]


T = TypeVar("T")


async def _limited_gather_with_factory(
    coro_factories: Iterable[Callable[[], Coroutine[Any, Any, T]]],
    limit: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    desc: Optional[str] = None,
    show_progress: bool = False,  # Default to False, only show when explicitly requested
) -> List[Optional[T]]:
    """Execute coroutine factories with concurrency limit and proper retry support.

    This is the recommended version for retry functionality.

    Args:
        coro_factories: Iterable of functions that create fresh coroutines
        limit: Maximum number of concurrent executions
        max_retries: Maximum number of retry attempts per task
        retry_delay: Base delay between retries (with exponential backoff)
        desc: Description for the progress bar
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        List of results, with None for permanently failed tasks
    """
    # TODO: Add adaptive concurrency based on system resources and task complexity
    sem = asyncio.Semaphore(limit)

    factory_list = list(coro_factories)
    total_tasks = len(factory_list)

    progress_bar = None
    if show_progress and total_tasks > 0:
        progress_bar = tqdm(
            total=total_tasks,
            desc=desc or "Processing",
            unit="chunk",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    async def _worker(
        coro_factory: Callable[[], Coroutine[Any, Any, T]], task_id: int
    ) -> Optional[T]:
        """Execute a coroutine factory with retry logic."""
        async with sem:
            for attempt in range(max_retries):
                try:
                    coro = coro_factory()
                    result = await coro
                    if progress_bar:
                        progress_bar.update(1)
                    return result
                except Exception as e:
                    if attempt <= max_retries - 1:
                        delay = retry_delay * (2**attempt)  # Exponential backoff
                        log_error_info(
                            logging.WARNING,
                            f"Task {task_id} failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s...",
                            e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        log_error_info(
                            logging.WARNING,
                            f"Task {task_id} failed permanently after {max_retries} attempts: {type(e).__name__}: {str(e)}",
                            e,
                        )
                        if progress_bar:
                            progress_bar.update(1)
            return None

    tasks = [
        asyncio.create_task(_worker(factory, i))
        for i, factory in enumerate(factory_list)
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=False)
    finally:
        if progress_bar:
            progress_bar.close()

    return results


# extract <ref>index</ref> tags in-order
def extract_ref_indices(text: str) -> list[int]:
    """Extract <ref>index</ref> tags in-order and return their integer indices."""
    if not text:
        return []
    return [int(m) for m in re.findall(r"<ref>\s*(\d+)\s*</ref>", text)]
