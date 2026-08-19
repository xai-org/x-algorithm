# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import asyncio
import logging
import os
import queue
import threading
import time
import typing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import grpc
import pyarrow as pa
from xai_configlib import configclass
from xai_proto.recsys_kafka_pb2 import (
    CheckStateRequest,
    FetchMessageRequest,
    FetchMessageResponse,
    InitClientRequest,
    PartitionOffset,
    SeekOffsets,
    SeekRequest,
)
from xai_proto.recsys_kafka_pb2_grpc import KafkaDispatcherStub

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.streaming.kafkaconsumer import deserialize_arrow_ipc
from xrex.data.streaming.kafkaloader import PhoenixKafkaDataset

log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
rank_logger = logging.getLogger("rank")

OPTIONS = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    (
        "grpc.default_compression_algorithm",
        grpc.Compression.NoCompression,
    ),
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.http2.max_pings_without_data", 0),
]


def get_grpc_channel(grpc_host: str, grpc_port) -> grpc.aio.Channel:
    target = f"{grpc_host}:{grpc_port}"
    return grpc.aio.insecure_channel(target, options=OPTIONS)


async def get_service_dimensions(grpc_host_template: str, grpc_port: int) -> tuple[int, int]:
    master_node_host = grpc_host_template.format(shard_index="0")
    async with get_grpc_channel(master_node_host, grpc_port) as channel:
        stub = KafkaDispatcherStub(channel)
        response = await stub.CheckState(CheckStateRequest())
        num_servers = response.num_shards
        num_processors = response.num_processors
        if num_servers is None or num_processors is None:
            raise ValueError(f"Invalid {num_servers=} or {num_processors=} in CheckStateResponse")
        return num_servers, num_processors


async def get_server_assignments(
    grpc_host_template: str,
    grpc_port: int,
    num_clients: int,
    client_ix: int,
) -> tuple[int, int, list[int]]:
    num_servers, num_processors = await get_service_dimensions(grpc_host_template, grpc_port)
    if num_servers is None or num_processors is None:
        raise ValueError("Failed to get num_servers or num_processors from gRPC service")

    assert num_clients % (num_servers * num_processors) == 0, (
        f"Num dataloaders {num_clients} must be a multiple of number of servers {num_servers=} * {num_processors=} = {num_servers * num_processors}"
    )

    global_num_processors = num_servers * num_processors
    num_clients_per_processor = num_clients // global_num_processors
    global_processor_ix = client_ix // num_clients_per_processor
    server_index = global_processor_ix // num_processors
    processor_index = global_processor_ix % num_processors
    clients_assigned_to_processor = [
        a for a in range(num_clients) if a // num_clients_per_processor == global_processor_ix
    ]
    assert len(clients_assigned_to_processor) == num_clients_per_processor, (
        f"Number of clients assigned to server {global_processor_ix} is not equal to num_clients_per_processor {num_clients_per_processor}"
    )
    return server_index, processor_index, clients_assigned_to_processor


async def initialize_clients(
    stub: KafkaDispatcherStub,
    clients_to_register: list[int],
    topic_name: str,
    seek_to_offset: dict[int, int] | None = None,
    seek_to_timestamp_ms: int | None = None,
    reset_to_latest: bool = False,
):
    logging.info(f"Registering client {clients_to_register} with gRPC service")
    _seek_to_offset = (
        [
            PartitionOffset(topic=topic_name, partition=partition, offset=offset)
            for partition, offset in seek_to_offset.items()
        ]
        if seek_to_offset
        else None
    )
    seek_request = SeekRequest()
    if _seek_to_offset is not None:
        seek_request.offsets.CopyFrom(SeekOffsets(offsets=_seek_to_offset))
    elif seek_to_timestamp_ms is not None:
        seek_request.timestampMs = seek_to_timestamp_ms
    elif reset_to_latest:
        seek_request.latest = reset_to_latest
    register_response = await stub.InitClientMessage(
        InitClientRequest(register_client_ids=clients_to_register, seek_request=seek_request)
    )
    rank_logger.info(f"Register response: {register_response}")
    if not register_response.success:
        rank_logger.error(f"Failed to register clients: {register_response.error_message}")
        raise Exception(f"Client registration failed: {register_response.error_message}")


async def consume_messages(
    grpc_host_template: str,
    grpc_port: int,
    num_shards: int,
    shard_index: int,
    topic_name: str,
    seek_to_offset: dict[int, int] | None,
    seek_to_timestamp_ms: int | None,
    reset_to_latest: bool,
    fetch_batch_size: int,
    batch_size: int,
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
    _stop_event: threading.Event | None = None,
    grpc_timeout: float = 30.0,
):
    (
        server_index,
        processor_index,
        clients_assigned_to_processor,
    ) = await get_server_assignments(
        grpc_host_template,
        grpc_port,
        num_shards,
        shard_index,
    )

    grpc_host = grpc_host_template.format(shard_index=str(server_index))
    async with get_grpc_channel(grpc_host, processor_index + grpc_port) as channel:
        stub = KafkaDispatcherStub(channel)

        if int(shard_index) == sorted(clients_assigned_to_processor)[0]:
            rank_logger.info(
                f"Client {shard_index}: Registering to current server {server_index}: {clients_assigned_to_processor} with gRPC service {grpc_host}"
            )
            await initialize_clients(
                stub,
                clients_assigned_to_processor,
                topic_name,
                seek_to_offset,
                seek_to_timestamp_ms,
                reset_to_latest,
            )
        rank_logger.info("Successfully connected to Kafka dispatcher gRPC service")

        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_running_loop()

            raw_message_queue = asyncio.Queue(maxsize=100)
            rank_logger.info(
                f"Consuming from Kafka dispatcher gRPC service, fetch_batch_size: {fetch_batch_size}"
            )

            async def consume_buffered():
                count = 0
                batch: list[pa.RecordBatch] = []
                current_offsets: dict[int, int] = {}
                processing_time_sum = 0

                while not (_stop_event and _stop_event.is_set()):
                    try:
                        response = await raw_message_queue.get()
                        if count % 100 == 0:
                            rank_logger.info(
                                f"len of raw_message_queue: {raw_message_queue.qsize()}"
                            )

                        valid_msgs = [msg for msg in response.messages if msg.value]
                        if not valid_msgs:
                            continue

                        deserialize_tasks = [
                            loop.run_in_executor(executor, deserialize_arrow_ipc, msg.value)
                            for msg in valid_msgs
                        ]
                        record_batches = await asyncio.gather(
                            *deserialize_tasks, return_exceptions=True
                        )

                        for kafka_msg, record_batch in zip(valid_msgs, record_batches):
                            if isinstance(record_batch, Exception):
                                continue

                            count += 1
                            if kafka_msg.offset > current_offsets.get(kafka_msg.partition, -1):
                                current_offsets[kafka_msg.partition] = kafka_msg.offset

                            batch.append(record_batch)
                            if len(batch) >= batch_size:
                                t = time.time()
                                batch_processed = post_process_fn(batch)
                                example_queue.put((batch_processed, current_offsets.copy()))
                                if count % 100 == 0:
                                    rank_logger.info(f"example_queue.size: {example_queue.qsize()}")
                                elapsed = time.time() - t
                                processing_time_sum += elapsed
                                batch = []

                    except Exception as e:
                        rank_logger.error(f"Buffered consumer error: {e}")

            consumer_task = asyncio.create_task(consume_buffered())

            while not (_stop_event and _stop_event.is_set()):
                try:
                    request = FetchMessageRequest(
                        event_count_to_fetch=fetch_batch_size, client_id=shard_index
                    )
                    response = await stub.FetchMessage(request, timeout=grpc_timeout)
                except Exception as e:
                    rank_logger.error(f"Error in FetchMessage: {e}")
                    raise

                if response.error_code < 0 or not response.messages:
                    rank_logger.debug(
                        f"FetchMessage failed: {response.error_message}, length of messages: {len(response.messages) if response.messages else 0}"
                    )
                    await asyncio.sleep(0.01)
                    continue
                response = typing.cast(FetchMessageResponse, response)
                await raw_message_queue.put(response)

            await consumer_task


@configclass
class PhoenixKafkaDispatcherDataset(PhoenixKafkaDataset):
    grpc_host_template: str = ""
    grpc_port: int = 50051
    grpc_timeout: float = 60.0
    poll_interval: float = 1.0

    fetch_batch_size: int = 384

    def _async_kafka_consumer_arrow(
        self,
        example_queue: queue.Queue,
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        stop_event: threading.Event | None = None,
    ):
        del run_server
        assert self.seek_to_timestamp_ms is None or self.seek_to_offset is None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        rank_logger.info(f"[{os.getpid()}] Starting gRPC protobuf service consumer")

        async def consumption_loop(stop_event):
            post_process_fn = partial(
                self.kafka_to_training_batch,
                batch_size=batch_size,
            )

            try:
                await consume_messages(
                    self.grpc_host_template,
                    self.grpc_port,
                    num_shards,
                    shard_index,
                    self.topic_name,
                    self.seek_to_offset,
                    self.seek_to_timestamp_ms,
                    self.reset_to_latest,
                    self.fetch_batch_size,
                    batch_size,
                    post_process_fn,
                    example_queue,
                    stop_event,
                    self.grpc_timeout,
                )
            except Exception as e:
                rank_logger.error(f"Failed to connect to gRPC protobuf service: {e}")
                raise

        try:
            loop.run_until_complete(consumption_loop(stop_event))
        except Exception as e:
            rank_logger.error(f"Consumer [{os.getpid()}] failed: {e}", exc_info=True)
        finally:
            rank_logger.info(f"Closing gRPC protobuf service consumer [{os.getpid()}]")
            loop.close()
