# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import os

KAFKA_CLUSTER: str = os.environ.get("XREX_KAFKA_CLUSTER", "phoenix")

KAFKA_RUST_BOOTSTRAP: str = os.environ.get("XREX_KAFKA_RUST_BOOTSTRAP", "phoenix")

KAFKA_DISPATCHER_HOST_TEMPLATE: str = os.environ.get("XREX_KAFKA_DISPATCHER_HOST_TEMPLATE", "")

GRPC_DATALOADER_ADDRESS: str = os.environ.get("XREX_GRPC_DATALOADER_ADDRESS", "")

OTEL_ENDPOINT: str = os.environ.get("XREX_OTEL_ENDPOINT", "")

SID_TOPIC: str = os.environ.get("XREX_SID_TOPIC", "recsys_sid_training")
SID_KAFKA_CLUSTER: str = os.environ.get("XREX_SID_KAFKA_CLUSTER", "phoenix")

RANKING_KAFKA_TOPIC: str = os.environ.get("XREX_RANKING_KAFKA_TOPIC", "recsys_ranking_training")
RANKING_RUST_KAFKA_TOPIC: str = os.environ.get(
    "XREX_RANKING_RUST_KAFKA_TOPIC", "recsys_ranking_training"
)
RANKING_DISPATCHER_KAFKA_TOPIC: str = os.environ.get(
    "XREX_RANKING_DISPATCHER_KAFKA_TOPIC", "recsys_ranking_training"
)

RETRIEVAL_KAFKA_TOPIC: str = os.environ.get(
    "XREX_RETRIEVAL_KAFKA_TOPIC", "recsys_retrieval_training"
)
RETRIEVAL_RUST_KAFKA_TOPIC: str = os.environ.get(
    "XREX_RETRIEVAL_RUST_KAFKA_TOPIC", "recsys_retrieval_training"
)

GEN_RECS_TOPIC: str = os.environ.get("XREX_GEN_RECS_TOPIC", "recsys_gen_recs_training")

KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
KAFKA_SASL_USERNAME: str = os.environ.get("KAFKA_SASL_USERNAME", "")


RANKING_DUMP_PATH: str = os.environ.get("XREX_RANKING_DUMP_PATH", "/path/to/offline_kafka_dump")
RETRIEVAL_DUMP_PATH: str = os.environ.get("XREX_RETRIEVAL_DUMP_PATH", "/path/to/offline_kafka_dump")

SID_GLOBAL_IDS_SNAPSHOT: str = os.environ.get("XREX_SID_GLOBAL_IDS_SNAPSHOT", "")

GEN_RECS_GLOBAL_IDS: str = os.environ.get(
    "XREX_GEN_RECS_GLOBAL_IDS", "/path/to/global_negative_pool.parquet"
)

GEN_RECS_OFFLINE_DATA_PATH: str = os.environ.get("OFFLINE_DATA_PATH", "synth_dump_gen_recs")
GEN_RECS_OFFLINE_ARTIFACTS_DIR: str = os.environ.get(
    "OFFLINE_ARTIFACTS_DIR", "synth_index/gen_recs_artifacts"
)

GEN_RECS_EVAL_POST_EMB_PATH: str = os.environ.get(
    "XREX_GEN_RECS_EVAL_POST_EMB_PATH",
    "synth_index/gen_recs_artifacts/inference_posts.parquet",
)


OBJECT_STORE_ENDPOINT: str = os.environ.get(
    "XREX_OBJECT_STORE_ENDPOINT", "http://object-store.example.invalid"
)
OBJECT_STORE_CREDENTIALS_DIR: str = os.environ.get(
    "XREX_OBJECT_STORE_CREDENTIALS_DIR", ".credentials/object_store"
)
GCS_CREDENTIALS_DIR: str = os.environ.get("XREX_GCS_CREDENTIALS_DIR", ".credentials/gcs")

GCS_MIRROR_BUCKET: str = os.environ.get("XREX_GCS_MIRROR_BUCKET", "")
GCS_MIRROR_PREFIX: str = os.environ.get("GCS_PREFIX", "")

ADS_S3_ENV_PREFIX: str = os.environ.get("XREX_ADS_S3_ENV_PREFIX", "")

PHOENIX_INDEX_BASE: str = os.environ.get("PHOENIX_INDEX_BASE", "phoenix_index")

ADS_INDEX_URI: str = os.environ.get("ADS_INDEX_URI", "")
DPA_INDEX_URI: str = os.environ.get("DPA_INDEX_URI", "")


ADS_CONVERSION_TRAIN_DATA: str = os.environ.get("XREX_ADS_CONVERSION_TRAIN_DATA", "")
ADS_CONVERSION_EVAL_DATA: str = os.environ.get("XREX_ADS_CONVERSION_EVAL_DATA", "")
ADS_CONVERSION_CHECKPOINT_DIR: str = os.environ.get("XREX_ADS_CONVERSION_CHECKPOINT_DIR", "")


AOT_CACHE_DIR: str = os.environ.get("XREX_AOT_CACHE_DIR", "")


REPL_SOCKET_PATH: str = os.environ.get("XREX_REPL_SOCKET", "/tmp/driver.sock")


CLICKHOUSE_HOST: str = os.environ.get("XREX_CLICKHOUSE_HOST", "")

CLICKHOUSE_USER: str = os.environ.get("RL2_CLICKHOUSE_USERNAME", "")

CLICKHOUSE_RUN_URL: str = os.environ.get("XREX_CLICKHOUSE_RUN_URL", "")


RUN_TRACKER_URL: str = os.environ.get("XAI_RUN_TRACKER_URL", "")
CHECKPOINT_PATH_PREFIX: str = os.environ.get("XAI_CHECKPOINT_PATH_PREFIX", "")


COMMIT_HASH_FILE: str = os.environ.get("XREX_COMMIT_HASH_FILE", "")


ROCE_FABRIC_CLOUD: str = os.environ.get("XREX_ROCE_FABRIC_CLOUD", "")


RETRIEVAL_DATASET_PATHS: dict[str, tuple[str, str | None]] = {}
