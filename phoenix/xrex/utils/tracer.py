# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import os
from pathlib import Path
from typing import TypeAlias, Union

from opentelemetry import context, propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.util.types import Attributes, AttributeValue
from serde import serde
from xai_configlib import Config, configclass

from xrex.utils import cluster

logger = logging.getLogger("tracer")
rank_logger = logging.getLogger("rank")

_DEFAULT_TIMEOUT_SECONDS: int = 10

NestedAttributes: TypeAlias = dict[str, Union[AttributeValue, "NestedAttributes"]]


class OsUrandomIdGenerator(IdGenerator):
    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), "big")

    def generate_trace_id(self) -> int:
        return int.from_bytes(os.urandom(16), "big")


class MockTracer:
    def start_as_current_span(self, *args, **kwargs):
        class MockContextManager:
            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        return MockContextManager()


@serde
@configclass
class TracerConfig(Config):
    enable: bool = True

    disable_otel: bool = False
    disable_perfetto: bool = True

    enable_driver_trace: bool = False

    otel_collector_addr: str | None = None

    perfetto_collector_addr: str | None = None

    local_trace_dir: Path | str | None = None

    enable_debug_log: bool = False

    sampling_rate: float = 1.0

    tracer_flush_interval: int = 600


def add_perfetto_span_processors(
    provider: TracerProvider, config: TracerConfig, service_name: str
) -> None:
    return


def setup_tracer(
    service_name: str,
    config: TracerConfig,
    resource_attributes: NestedAttributes | None = None,
) -> TracerProvider | None:
    if not config.enable:
        rank_logger.debug("Tracer disabled")
        return None

    debug_to_console = config.enable_debug_log
    sampling_rate = config.sampling_rate

    if not (
        config.otel_collector_addr
        or config.perfetto_collector_addr
        or config.local_trace_dir
        or debug_to_console
    ):
        rank_logger.info(
            "Tracer disabled: otel_collector_addr, perfetto_collector_addr, and debug_to_console are disabled"
        )
        return None

    logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.WARNING)

    sampler = TraceIdRatioBased(rate=sampling_rate)
    id_gen = OsUrandomIdGenerator()
    resource = Resource(attributes=compose_resource_attributes(service_name, resource_attributes))

    provider = TracerProvider(
        id_generator=id_gen,
        sampler=sampler,
        resource=resource,
    )

    if debug_to_console:
        exporter = ConsoleSpanExporter(service_name=service_name)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    if config.otel_collector_addr and not config.disable_otel:
        exporter = OTLPSpanExporter(config.otel_collector_addr, timeout=_DEFAULT_TIMEOUT_SECONDS)
        old_shutdown = exporter.shutdown

        def shutdown_patched():
            old_shutdown()
            exporter._client = None

        exporter.shutdown = shutdown_patched
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    if not config.disable_perfetto:
        add_perfetto_span_processors(provider, config, service_name)

    trace.set_tracer_provider(provider)

    return provider


def get_otel_collector_addr(tracer_config: TracerConfig) -> str | None:
    if not tracer_config.enable:
        return None
    if tracer_config.disable_otel:
        return None
    return tracer_config.otel_collector_addr


def flatten_attributes(attributes: NestedAttributes) -> Attributes:
    if not attributes:
        return attributes
    flattened = {}
    for attr_key, attr_value in attributes.items():
        if not attr_value:
            continue
        elif isinstance(attr_value, dict):
            for inner_key, inner_value in flatten_attributes(attr_value).items():
                flattened[f"{attr_key}.{inner_key}"] = inner_value
        else:
            flattened[attr_key] = attr_value
    return flattened


def compose_resource_attributes(
    service_name,
    extra_resource_attributes: NestedAttributes | None = None,
) -> Attributes:
    basic = flatten_attributes(
        {
            "service.name": service_name,
            "host.name": cluster.get_hostname() or "UNKNOWN",
            "k8s.cluster.name": cluster.get_cluster() or "UNKNOWN",
            "k8s.node.name": cluster.get_node_name() or "UNKNOWN",
            "k8s.pod.name": cluster.get_pod_name() or "UNKNOWN",
            "user": cluster.get_user() or "UNKNOWN",
            "app": {
                "job": cluster.get_job_name() or "UNKNOWN",
                "host": cluster.get_hostname() or "UNKNOWN",
            },
        }
    )
    extra = flatten_attributes(extra_resource_attributes) or {}
    return basic | extra


def inject_from_span_ctx(span_context: trace.SpanContext) -> dict[str, str]:
    carrier = {}
    propagate.inject(
        carrier,
        context=trace.set_span_in_context(trace.NonRecordingSpan(span_context)),
    )
    return carrier


def extract_to_span_ctx(carrier: dict[str, str]) -> trace.SpanContext:
    ctx = propagate.extract(carrier)
    span = trace.get_current_span(ctx)
    return span.get_span_context()


def current_trace_ctx_carrier() -> dict[str, str]:
    carrier = {}
    propagate.inject(carrier)
    return carrier


def current_traceparent() -> str | None:
    return current_trace_ctx_carrier().get("traceparent")


def current_trace_ctx() -> trace.Context:
    return trace.set_span_in_context(trace.get_current_span())


def trace_ctx_from_carrier(carrier: dict[str, str]) -> trace.Context:
    return propagate.extract(carrier)


def trace_ctx_from_traceparent(traceparent: str) -> trace.Context:
    return propagate.extract({"traceparent": traceparent})


def attach_span(span: trace.Span) -> object:
    ctx = trace.set_span_in_context(span)
    token = context.attach(ctx)
    return token


def start_and_attach_span(tracer: trace.Tracer, *args, **kwargs) -> tuple[trace.Span, object]:
    span = tracer.start_span(*args, **kwargs)
    token = attach_span(span)
    return span, token


def detach_and_end_span(span: trace.Span, token: object):
    context.detach(token)
    span.end()


def start_linked_root_span(
    tracer: trace.Tracer,
    name: str,
    attributes: dict[str, AttributeValue] | None = None,
) -> trace.Span:
    attributes = attributes or {}

    parent_span = trace.get_current_span()
    parent_ctx = parent_span.get_span_context()

    links: list[trace.Link] = []
    if parent_ctx.is_valid:
        links.append(trace.Link(parent_ctx))
        attributes["parent_trace_id"] = format(parent_ctx.trace_id, "032x")

    empty_context = trace.set_span_in_context(trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT))

    return tracer.start_span(
        name,
        context=empty_context,
        links=links,
        attributes=attributes,
    )
