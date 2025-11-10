import os

from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.httpx import (
    HTTPXClientInstrumentor,
    RequestInfo,
    ResponseInfo,
)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.metrics import Meter

_http_client_requests_total = None  # lazy created


def setup_instrumentors(enable_metrics: bool, meter: Meter) -> None:
    global _http_client_requests_total
    if enable_metrics:
        _http_client_requests_total = meter.create_counter("http_client_requests_total")

        # see https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = (
            "http,database,gen_ai_latest_experimental"
        )

    SQLAlchemyInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    AsyncPGInstrumentor().instrument()
    ThreadingInstrumentor().instrument()
    _setup_httpx()


def _maybe_count_http(method: str, url: str, status: str):
    if _http_client_requests_total is None:
        return
    _http_client_requests_total.add(1, {"method": method, "url": url, "status": status})


def _setup_httpx():
    def httpx_response_hook(_, request: RequestInfo, response: ResponseInfo):
        _maybe_count_http(
            request.method.decode(), str(request.url), str(response.status_code)
        )

    async def httpx_async_response_hook(
        span, request: RequestInfo, response: ResponseInfo
    ):
        _maybe_count_http(
            request.method.decode(), str(request.url), str(response.status_code)
        )

    HTTPXClientInstrumentor().instrument(
        response_hook=httpx_response_hook,
        async_response_hook=httpx_async_response_hook,
    )
