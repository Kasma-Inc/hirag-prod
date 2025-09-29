import logging
from datetime import datetime
from enum import Enum

from hirag_prod._utils import log_error_info
from hirag_prod.resources.functions import get_db_session_maker
from hirag_prod.storage.pg_utils import update_job_status

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusTracker:
    """
    Minimal job tracker without any caching.

    - No Redis usage
    - Only persists job status transitions to Postgres
    """

    def __init__(
        self,
        auto_cleanup: bool = True,
    ):
        """Initialize the resume tracker"""
        self.auto_cleanup = auto_cleanup

    # ============================== Job tracking =============================
    async def save_job_status_to_postgres(self, job_id: str, status: str) -> None:
        """Persist job status to PostgreSQL."""
        try:
            normalized_status = status or ""
            if normalized_status.lower() == "progress":
                normalized_status = JobStatus.PROCESSING.value
            async with get_db_session_maker()() as session:
                await update_job_status(
                    session, job_id, normalized_status, updated_at=datetime.now()
                )
        except Exception as e:
            log_error_info(logging.ERROR, f"Failed to save job status to Postgres", e)

    async def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
    ) -> None:
        await self.save_job_status_to_postgres(job_id, status.value)
