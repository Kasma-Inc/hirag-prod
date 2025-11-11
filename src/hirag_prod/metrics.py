import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

# from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import (  # Observation,
    Meter,
    get_meter_provider,
    set_meter_provider,
)
from opentelemetry.metrics._internal import NoOpMeterProvider

from utils.logging_utils import log_error_info

# from opentelemetry.sdk.metrics import MeterProvider
# from opentelemetry.sdk.metrics._internal.export import ConsoleMetricExporter
# from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


logger = logging.getLogger("HiRAG")

_meter: Optional[Meter] = None


def setup_metrics():
    """Setup OpenTelemetry metrics with OTLP exporter."""

    global _meter
    if _meter is not None:
        return  # Already initialized
    # TODO(yukkit): Enable OTLP exporter
    # exporter = OTLPMetricExporter(insecure=True)
    # exporter = ConsoleMetricExporter()
    # reader = PeriodicExportingMetricReader(exporter)
    # provider = MeterProvider(metric_readers=[reader])
    provider = NoOpMeterProvider()
    set_meter_provider(provider)
    _meter = get_meter_provider().get_meter(__name__)


def get_meter() -> Meter:
    """Get the global meter instance."""
    global _meter
    if _meter is None:
        raise RuntimeError("Meter is not initialized. Call setup_metrics first.")
    return _meter


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
