import asyncio
import logging
import re
from chunk import Chunk
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
    Tuple,
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


def log_error_info(
    level: int,
    message: str,
    error: BaseException,
    debug_only: bool = False,
    exc_info: Optional[bool] = None,
    raise_error: bool = False,
    new_error_class: Optional[Type[Any]] = None,
):
    if (not debug_only) or get_config_manager().debug:
        logger.log(
            level,
            f"{message}: {error}",
            exc_info=get_config_manager().debug if exc_info is None else exc_info,
        )
    if raise_error:
        raise new_error_class(message) if new_error_class is not None else error


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


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


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
def extract_ref_indices(text: str) -> List[list[int]]:
    """Extract <ref>index</ref> tags in-order and return their integer indices."""
    if not text:
        return [[]]
    return [[int(m)] for m in re.findall(r"<ref>\s*(\d+)\s*</ref>", text)]


async def extract_ref_indices_from_markdown(
    text: str,
    chunk_list: List[Chunk],
    threshold: float = 0.8,
    concurrency: int = 4,
) -> Tuple[List[List[int]], List[str]]:
    """Extract text segments immediately preceding each [slot](x) inside fenced ```markdown blocks,
    then, for each segment, rerank against the given chunk_list and return indices
    of chunks passing threshold (>= threshold). If none pass, return the single best.

    Returns:
        List[List[int]]: citations per extracted segment, each a list of indices into chunk_list.
        List[str]: texts to cite per extracted segment.
    """
    if not text:
        return [], []

    markdown_blocks = re.findall(
        r"```markdown\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not markdown_blocks:
        return [], []

    markdown_content = "\n".join(markdown_blocks)
    texts_to_cite: list[str] = []

    last_slot_end: int = 0
    for m in re.finditer(r"\[slot\]\((\d+)\)", markdown_content):
        start = m.start()
        i = start - 1
        while i >= 0 and markdown_content[i].isspace():
            i -= 1

        end_term = i

        prev_term = -1
        j = end_term - 1
        while j >= 0:
            ch = markdown_content[j]
            if ch == "\n":
                prev_term = j
                break
            j -= 1

        prev_term = max(prev_term, last_slot_end)
        snippet = markdown_content[prev_term + 1 : end_term + 1].strip()

        snippet = re.sub(r"^(?:>\s*)?(?:[-*+]\s+|\d+\.\s+|#{1,6}\s+)\s*", "", snippet)

        texts_to_cite.append(snippet)
        last_slot_end = m.end()

    if not texts_to_cite or not chunk_list:
        return [[] for _ in texts_to_cite], texts_to_cite

    from hirag_prod.resources.functions import get_reranker

    reranker = get_reranker()

    items = [{"text": c["text"], "list_index": i} for i, c in enumerate(chunk_list)]

    async def _rank_one(q: str):
        try:
            reranked = await reranker.rerank(q, items, key="text")
            return reranked or []
        except Exception as e:
            log_error_info(logging.ERROR, "Reranking failed", e)
            return []

    factories = [lambda q=q: _rank_one(q) for q in texts_to_cite]
    reranked_lists = await _limited_gather_with_factory(
        factories, limit=concurrency, max_retries=3, retry_delay=1.0
    )

    results: List[List[int]] = []
    for reranked in reranked_lists:
        if not reranked:
            results.append([])
            continue
        indices = [
            element.get("list_index")
            for element in reranked
            if element.get("relevance_score", 0.0) >= threshold
        ]
        if indices:
            seen = set()
            selected = []
            for idx in indices:
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    selected.append(idx)
            results.append(selected)
        else:
            best = reranked[0].get("list_index")
            results.append([best] if best is not None else [])

    return results, texts_to_cite
