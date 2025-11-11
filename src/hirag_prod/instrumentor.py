from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import (
    HTTPXClientInstrumentor,
    RequestInfo,
    ResponseInfo,
)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.metrics import Meter
from requests import PreparedRequest, Response

_http_client_requests_total = None  # lazy created


def setup_fastapi(app, enable_metrics: bool) -> None:
    if enable_metrics:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
            inprogress_name="inprogress",
            inprogress_labels=True,
        ).instrument(app).expose(app)

    FastAPIInstrumentor.instrument_app(
        app=app,
        excluded_urls="/status,/metrics,/health",
        exclude_spans=["receive", "send"],
    )


def setup_instrumentors(enable_metrics: bool, meter: Meter) -> None:
    global _http_client_requests_total
    if enable_metrics:
        _http_client_requests_total = meter.create_counter("http_client_requests_total")

    SQLAlchemyInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    AsyncPGInstrumentor().instrument()
    ThreadingInstrumentor().instrument()
    _setup_requests()
    _setup_httpx()


def _maybe_count_http(method: str, url: str, status: str):
    if _http_client_requests_total is None:
        return
    _http_client_requests_total.add(1, {"method": method, "url": url, "status": status})


def _setup_requests():
    def requests_response_hook(_, request: PreparedRequest, response: Response):
        _maybe_count_http(
            request.method or "unknown",
            request.url or "unknown",
            str(response.status_code),
        )

    RequestsInstrumentor().instrument(
        excluded_urls="/health,/status",
        response_hook=requests_response_hook,
    )


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
