import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict

from opentelemetry.metrics import (  # Observation,
    Meter,
    get_meter_provider,
    set_meter_provider,
)
from opentelemetry.metrics._internal import NoOpMeter

from hirag_prod._utils import log_error_info

logger = logging.getLogger("HiRAG")

# environment variable for Prometheus multiprocess mode
PROMETHEUS_MULTIPROC_DIR = "PROMETHEUS_MULTIPROC_DIR"

_meter: Meter = NoOpMeter("")


def setup_metrics(enable_metrics: bool) -> Meter:
    """Setup OpenTelemetry metrics with OTLP exporter."""

    global _meter

    if enable_metrics:

        # Exporter to export metrics to Prometheus
        if PROMETHEUS_MULTIPROC_DIR in os.environ:
            # If running in multiprocess mode, use the multiprocess provider
            from opentelemetry.sdk.extension.prometheus_multiprocess import (
                PrometheusMeterProvider,
            )

            provider = PrometheusMeterProvider()
        else:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from opentelemetry.sdk.metrics import MeterProvider

            reader = PrometheusMetricReader(False)
            provider = MeterProvider(metric_readers=[reader])

        set_meter_provider(provider)
        _meter = get_meter_provider().get_meter(__name__)

    return _meter


def get_meter() -> Meter:
    """Get the global meter instance."""
    return _meter


def prepare_prometheus_multiproc_dir():
    if PROMETHEUS_MULTIPROC_DIR not in os.environ:
        logger.info(
            f"{PROMETHEUS_MULTIPROC_DIR} not set, will start in single process mode."
        )
        return

    dir = os.environ[PROMETHEUS_MULTIPROC_DIR]

    logger.info(f"Using Prometheus multiproc dir: {dir}")

    if os.path.exists(dir):
        shutil.rmtree(dir)

    try:
        os.makedirs(dir, exist_ok=False)
    except Exception as e:
        logger.error(f"Failed to create Prometheus multiproc dir: {e}")
        raise


def cleanup_prometheus_multiproc_dir():
    dir = os.environ.get(PROMETHEUS_MULTIPROC_DIR, None)
    if dir and os.path.exists(dir):
        try:
            os.removedirs(dir)
            logger.info(f"Cleaned up Prometheus multiproc dir: {dir}")
        except Exception as e:
            logger.error(f"Failed to clean up Prometheus multiproc dir: {e}")


@dataclass
class ProcessingMetrics:
    """Processing metrics"""

    total_chunks: int = 0
    processed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    processing_time: float = 0.0
    error_count: int = 0
    file_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "processing_time": self.processing_time,
            "error_count": self.error_count,
            "file_id": self.file_id,
        }


class MetricsCollector:
    """Metrics collector"""

    def __init__(self):
        self.metrics = ProcessingMetrics()
        self.operation_times: Dict[str, float] = {}

    @asynccontextmanager
    async def track_operation(self, operation: str):
        """Track operation execution time"""
        start = time.perf_counter()
        try:
            logger.info(f"🚀 Starting {operation}")
            yield
            duration = time.perf_counter() - start
            self.operation_times[operation] = duration
            logger.info(f"✅ Completed {operation} in {duration:.3f}s")
        except Exception as e:
            self.metrics.error_count += 1
            duration = time.perf_counter() - start
            log_error_info(
                logging.ERROR,
                f"❌ Failed {operation} after {duration:.3f}s",
                e,
                raise_error=True,
            )
