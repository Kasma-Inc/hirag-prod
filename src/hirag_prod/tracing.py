"""
Private Tracing setup and utility functions for OpenTelemetry.
Exposes setup_tracing, get_tracer, trace_span, trace_span_async and traced decorator.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from inspect import signature
from typing import AsyncGenerator, Optional

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer: trace.Tracer = trace.NoOpTracer()


def _build_tracer_provider(
    service_name: str = "dotsocr",
    otel_exporter_otlp_traces_endpoint: Optional[str] = None,
    otel_exporter_otlp_traces_timeout: Optional[int] = None,
) -> trace.TracerProvider:
    if otel_exporter_otlp_traces_endpoint is None:
        return trace.NoOpTracerProvider()

    trace_provider = trace_sdk.TracerProvider(
        resource=Resource(attributes={"service.name": service_name})
    )
    otlp_exporter = OTLPSpanExporter(
        endpoint=otel_exporter_otlp_traces_endpoint,
        timeout=otel_exporter_otlp_traces_timeout,
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace_provider.add_span_processor(span_processor)
    return trace_provider


def setup_tracing(
    service_name: str = "rag",
    otel_exporter_otlp_traces_endpoint: Optional[str] = None,
    otel_exporter_otlp_traces_timeout: Optional[int] = None,
) -> trace.Tracer:
    """
    Initialize the tracer with OTLP exporter and FastAPI instrumentation.
    """
    global _tracer

    trace_provider = _build_tracer_provider(
        service_name,
        otel_exporter_otlp_traces_endpoint,
        otel_exporter_otlp_traces_timeout,
    )

    trace.set_tracer_provider(trace_provider)

    _tracer = trace.get_tracer(__name__)

    return _tracer


def get_tracer() -> trace.Tracer:
    """
    Get the initialized tracer instance.
    """
    return _tracer


def get_global_textmap() -> propagate.textmap.TextMapPropagator:
    """
    Get the global textmap propagator instance.
    """
    return propagate.get_global_textmap()


def start_child_span(name: str, parent_span: Optional[trace.Span] = None) -> trace.Span:
    """
    Start and return a child span of the given parent span.
    If parent_span is None, starts a new span with current context.
    """
    tracer = get_tracer()
    if parent_span is not None and parent_span.get_span_context().is_valid:
        return tracer.start_span(name, context=trace.set_span_in_context(parent_span))
    else:
        return tracer.start_span(name)


def serialize_current_span_context() -> dict:
    """
    Serialize the current span context to a dictionary.
    """
    parent_span_context = {}
    get_global_textmap().inject(parent_span_context)
    return parent_span_context


@contextmanager
def trace_span(name: str, **attributes):
    """
    Sync context manager for tracing a span.

    - Records CPU time automatically as a span attribute.
    - Records any raised exception into the span.
    """

    tracer = trace.get_tracer(__name__)
    cpu_start = time.process_time()

    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        for k, v in attributes.items():
            span.set_attribute(k, v)

        try:
            yield span
        except Exception as e:
            # Record the exception and mark span as errored
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Always record CPU time
            span.set_attribute(
                "cpu_time_ns", (time.process_time() - cpu_start) * 1_000_000_000
            )


@asynccontextmanager
async def trace_span_async(name: str, **attributes):
    """
    Async context manager for tracing a span.

    - Records CPU time automatically as a span attribute.
    - Records any raised exception into the span.
    """

    tracer = trace.get_tracer(__name__)
    cpu_start = time.process_time()

    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        for k, v in attributes.items():
            span.set_attribute(k, v)

        try:
            yield span
        except Exception as e:
            # Record the exception and mark span as errored
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Always record CPU time
            span.set_attribute(
                "cpu_time_ns", (time.process_time() - cpu_start) * 1_000_000_000
            )


def traced(
    name: Optional[str] = None,
    record_args: Optional[list[str]] = None,
    record_return: bool = False,
):
    """
    Decorator to wrap a function (sync or async) in a tracing span.
    Optionally record only specified function arguments.

    :param name: span name. Defaults to function name if None.
    :param record_args: list of argument names to record. If None, record all.
    :param record_return: whether to record the return value. Defaults to False.
    """

    def decorator(func):
        span_name = name or func.__qualname__
        sig = signature(func)

        def _build_attributes(args, kwargs, exclude: list[str] = ["self", "cls"]):
            """
            Build individual attributes for each selected function argument.
            """
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            attributes = {}
            for arg_name, value in bound.arguments.items():
                if arg_name in exclude:
                    continue
                if record_args is None or arg_name in record_args:
                    attributes[f"param.{arg_name}"] = json.dumps(value, default=str)
            return attributes

        def _build_json_attributes(args, kwargs, exclude: list[str] = ["self", "cls"]):
            """
            Build a single JSON string attribute with selected function arguments.
            """
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            data = {}
            for k, v in bound.arguments.items():
                if k in exclude:
                    continue
                if record_args is None or k in record_args:
                    try:
                        data[k] = v
                    except Exception:
                        data[k] = repr(v)
            return {"params": json.dumps(data, default=str)}

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attrs = _build_attributes(args, kwargs)
                async with trace_span_async(span_name, **attrs) as span:
                    result = await func(*args, **kwargs)
                    if record_return:
                        try:
                            span.set_attribute(
                                "return", json.dumps(result, default=str)
                            )
                        except Exception:
                            span.set_attribute("return", repr(result))
                    return result

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attrs = _build_attributes(args, kwargs)
                with trace_span(span_name, **attrs) as span:
                    result = func(*args, **kwargs)
                    if record_return:
                        try:
                            span.set_attribute(
                                "return", json.dumps(result, default=str)
                            )
                        except Exception:
                            span.set_attribute("return", repr(result))
                    return result

            return sync_wrapper

    return decorator


def traced_async_gen(
    name: Optional[str] = None,
    record_args: Optional[list[str]] = None,
    record_return: bool = False,
    per_yield_span: bool = True,
    **span_attrs,
):
    """
    Decorator for tracing async generator functions with OpenTelemetry.

    This decorator traces the lifecycle of an async generator.
    Optionally, each `yield` can be wrapped in a child span to capture timing
    and per-chunk behavior.

    Args:
        name (Optional[str]): Span name. Defaults to the function name.
        record_args (Optional[List[str]]): List of argument names to record as attributes.
            If None, record all arguments.
        record_return (bool): Whether to record yielded values. Defaults to False.
        per_yield_span (bool): If True, creates a child span for each yield.
        **span_attrs: Additional static attributes to attach to the root span.

    Example:
        @traced_async_gen(record_args=["user_id"], record_return=True, per_yield_span=True)
        async def stream_output(user_id: str):
            for token in ["hi", "there"]:
                yield token
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator:
            span_name = name or func.__qualname__

            # Build attributes from args
            attributes = dict(span_attrs)
            from inspect import signature

            sig = signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            arg_names_to_record = record_args or bound.arguments.keys()
            for arg_name in arg_names_to_record:
                if arg_name in bound.arguments:
                    attributes[f"arg.{arg_name}"] = bound.arguments[arg_name]

            with _tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    async for result in func(*args, **kwargs):
                        if per_yield_span:
                            # Each yield gets its own span for detailed latency tracing
                            with _tracer.start_as_current_span(
                                f"{span_name}.yield",
                                attributes={
                                    "value": (
                                        json.dumps(result, default=str)
                                        if record_return
                                        else "<hidden>"
                                    )
                                },
                            ):
                                yield result
                        else:
                            if record_return:
                                span.add_event(
                                    "yield", {"value": json.dumps(result, default=str)}
                                )
                            yield result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator
