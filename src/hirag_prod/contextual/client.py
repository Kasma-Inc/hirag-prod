import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextual import AsyncContextualAI
from contextual.types import ParseCreateResponse, ParseJobStatusResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_envs
from hirag_prod.storage.pg_utils import queryContextResult, saveContextResult


class ContextualClient:
    """
    A client for parsing documents using Contextual AI's API.

    This client provides an easy-to-use interface for parsing documents (PDFs, Word docs, PowerPoint)
    into structured markdown and JSON formats.

    Example:
        client = ContextualClient()
        result = await client.parse_document("document.pdf")
        print(result['markdown_document'])
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Contextual AI client.

        Args:
            api_key: Your Contextual AI API key. If not provided, will look for CONTEXTUAL_API_KEY environment variable.
        """
        self.api_key = api_key or get_envs().CONTEXTUAL_API_KEY
        if not self.api_key:
            raise ValueError(
                "API key must be provided either as an argument or through the CONTEXTUAL_API_KEY environment variable."
            )
        self.client = AsyncContextualAI(api_key=self.api_key)

    async def get_parse_status(self, job_id: str) -> Optional[ParseJobStatusResponse]:
        """
        Get the status of a parse job.

        Args:
            job_id: The job ID returned from parse_document
        """
        try:
            response = await self.client.parse.job_status(job_id=job_id)
            return response
        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Error getting parse status for job {job_id}",
                e,
                raise_error=True,
            )

    async def get_parse_results(
        self, job_id: str, output_types: List[str] = None, session: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Get the results of a completed parse job.
        First checks database cache, then fetches from API if not found and saves to DB.

        Args:
            job_id: The job ID returned from parse_document
            output_types: List of output types to retrieve (e.g., ["markdown-document", "blocks-per-page"])
            session: Optional AsyncSession for database operations

        * Now using ["markdown-document"] as default value, refer to: https://docs.contextual.ai/api-reference/parse/parse-result
        """
        try:
            # First check if results are cached in database (if session is provided)
            if session:
                cached_result = await queryContextResult(session, job_id)
                if cached_result:
                    return cached_result

            # If not in cache, fetch from API
            if output_types is None:
                output_types = ["markdown-document"]

            response = await self.client.parse.job_results(
                job_id=job_id, output_types=output_types
            )

            # Add job_id to response for saving
            if hasattr(response, "model_dump"):
                result_dict = response.model_dump()
            else:
                result_dict = response

            result_dict["job_id"] = job_id

            # Save to database if session is provided
            if session:
                saved_dict = await saveContextResult(session, result_dict)
                return saved_dict
            else:
                return result_dict

        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Error getting parse results for job {job_id}",
                e,
                raise_error=True,
            )

    async def wait_for_parse_completion(
        self,
        job_id: str,
        poll_interval: int = 5,
        max_wait_time: int = 300,
        session: AsyncSession = None,
    ) -> Dict[str, Any]:
        """
        Wait for a parse job to complete and return the result.

        Args:
            job_id: The job ID returned from parse_document
            poll_interval: Time in seconds between status checks
            max_wait_time: Maximum time in seconds to wait for completion
            session: Optional AsyncSession for database operations

        * I didn't see official handles for this, used AI generated polling for waiting the results
        """
        import time

        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_response = await self.get_parse_status(job_id)

            if hasattr(status_response, "status"):
                status = status_response.status
            else:
                raise Exception(f"Unexpected status response format for job {job_id}")

            if status == "completed":
                # Get the actual results
                return await self.get_parse_results(job_id, session=session)
            elif status == "failed":
                raise Exception(f"Parse job {job_id} failed")

            await asyncio.sleep(poll_interval)

        raise Exception(
            f"Parse job {job_id} did not complete within {max_wait_time} seconds"
        )

    async def parse_document_sync(
        self,
        file_path: str,
        parse_mode: str = "standard",
        enable_document_hierarchy: bool = True,
        enable_split_tables: bool = False,
        max_split_table_cells: Optional[int] = None,
        figure_caption_mode: str = "concise",
        page_range: Optional[str] = None,
        poll_interval: int = 5,
        max_wait_time: int = 300,
        session: AsyncSession = None,
    ) -> Dict[str, Any]:
        """
        Parse a document and wait for completion in a single call.

        This is a convenience method that combines parse_document and wait_for_parse_completion.

        Args:
            file_path: Path to the document file
            parse_mode: "basic" for simple documents, "standard" for complex documents
            enable_document_hierarchy: Add table of contents with document structure
            enable_split_tables: Split large tables into multiple smaller tables
            max_split_table_cells: Threshold for splitting tables (if enable_split_tables is True)
            figure_caption_mode: "concise" or "detailed" captions for figures
            page_range: Optional page range to parse (e.g., "0,1,2" or "0-2,5,6")
            poll_interval: Time in seconds between status checks
            max_wait_time: Maximum time in seconds to wait for completion
            session: Optional AsyncSession for database operations

        Returns:
            The completed parse results
        """
        # Start the parse job
        parse_response = await self.parse_document(
            file_path=file_path,
            parse_mode=parse_mode,
            enable_document_hierarchy=enable_document_hierarchy,
            enable_split_tables=enable_split_tables,
            max_split_table_cells=max_split_table_cells,
            figure_caption_mode=figure_caption_mode,
            page_range=page_range,
        )

        # Extract job ID
        if hasattr(parse_response, "job_id"):
            job_id = parse_response.job_id
        else:
            raise Exception("Could not extract job_id from parse response")

        # Wait for completion and return results
        parse_result = await self.wait_for_parse_completion(
            job_id, poll_interval, max_wait_time, session=session
        )

        # Add job_id to parse result
        parse_result["job_id"] = job_id
        return parse_result

    # This is a helper function and also for non-waiting parsing
    async def parse_document(
        self,
        file_path: str,
        parse_mode: str = "standard",
        enable_document_hierarchy: bool = True,
        enable_split_tables: bool = False,
        max_split_table_cells: Optional[int] = None,
        figure_caption_mode: str = "concise",
        page_range: Optional[str] = None,
    ) -> Optional[ParseCreateResponse]:
        """
        This function is for posting the job to the Contextual API, but would not wait for results.

        To wait for results, call parse_document_sync or do polling and use get_parse_status.

        Refer to: https://docs.contextual.ai/api-reference/parse/parse-file

        Args:
            file_path: Path to the document file
            parse_mode: "basic" for simple documents, "standard" for complex documents
            enable_document_hierarchy: Add table of contents with document structure
            enable_split_tables: Split large tables into multiple smaller tables
            max_split_table_cells: Threshold for splitting tables (if enable_split_tables is True)
            figure_caption_mode: "concise" or "detailed" captions for figures
            page_range: Optional page range to parse (e.g., "0,1,2" or "0-2,5,6")

        Returns:
            A dictionary containing job_id for successful posting
            Details of error on unsuccessful cases
        """

        try:
            # Convert string path to Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Use the parse API endpoint
            response = await self.client.parse.create(
                raw_file=file_path_obj,
                parse_mode=parse_mode,
                enable_document_hierarchy=enable_document_hierarchy,
                enable_split_tables=enable_split_tables,
                max_split_table_cells=max_split_table_cells,
                figure_caption_mode=figure_caption_mode,
                page_range=page_range,
            )

            return response
        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Error parsing document {file_path}",
                e,
                raise_error=True,
            )
