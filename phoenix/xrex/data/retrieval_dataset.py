# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import io
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from xrex import settings
from xrex.data.cold_pool_filter import (
    cold_pool_mask,
    default_cold_pool_metadata_path,
    fresh_post_mask,
)
from xrex.data.recsys.constants import VIDEO_CREATIVE_BIT

logger = logging.getLogger(__name__)

_O2_SCHEME = "o2://"

O2_DEFAULT_ENDPOINT = settings.OBJECT_STORE_ENDPOINT

_O2_CREDENTIALS_DIR = Path(settings.OBJECT_STORE_CREDENTIALS_DIR)
_GCS_CREDENTIALS_DIR = Path(settings.GCS_CREDENTIALS_DIR)

_GCS_DEFAULT_BUCKET = settings.GCS_MIRROR_BUCKET

_GCS_MIRRORED = {"ACTIVE_ADS", "DPA_PRODUCTS"}

PHOENIX_INDEX_BASE = Path(settings.PHOENIX_INDEX_BASE)


def _bridge_o2_env() -> None:
    if not settings.O2_ENV_PREFIX:
        return
    for suffix in ("ENDPOINT", "ACCESS_KEY", "SECRET_KEY"):
        prefixed = f"{settings.O2_ENV_PREFIX}O2_PROD_{suffix}"
        generic = f"O2_PROD_{suffix}"
        if os.environ.get(generic):
            continue
        val = os.environ.get(prefixed)
        if not val:
            secret = _O2_CREDENTIALS_DIR / prefixed
            if secret.is_file():
                val = secret.read_text().strip()
        if val:
            os.environ[generic] = val


_bridge_o2_env()


_SID_SNAPSHOTS = "post_sid_v8_256x6_snapshots"


def _sid_window(filename: str) -> tuple[str, str]:
    return (
        str(PHOENIX_INDEX_BASE / _SID_SNAPSHOTS / filename),
        str(PHOENIX_INDEX_BASE / f"{_SID_SNAPSHOTS}_backup" / filename),
    )


def _idx(sub: str) -> str:
    return str(PHOENIX_INDEX_BASE / sub)


_O2_MAX_STALENESS_SECONDS = 3 * 3600


def _read_credential(env_var: str, search_dirs: list[Path] | None = None) -> str:
    val = os.environ.get(env_var)
    if val:
        return val
    for d in search_dirs or [_O2_CREDENTIALS_DIR]:
        path = d / env_var
        if path.is_file():
            return path.read_text().strip()
    searched = ", ".join(str(d / env_var) for d in (search_dirs or [_O2_CREDENTIALS_DIR]))
    raise RuntimeError(
        f"{env_var} not found in environment or at {searched}. "
        f"Set the env var or mount the credential file in the credentials directory."
    )


def _read_file_credential(filename: str, search_dirs: list[Path]) -> str | None:
    val = os.environ.get(filename)
    if val:
        return val
    for d in search_dirs:
        path = d / filename
        if path.is_file():
            return str(path)
    return None


def _parse_snapshot_timestamp(key: str) -> int | None:
    basename = key.rsplit("/", 1)[-1]
    part = basename.split("_", 1)[0]
    try:
        return int(part)
    except ValueError:
        return None


def _download_from_o2(name: str, bucket: str, prefix: str) -> tuple[bytes, int, str]:
    import boto3

    endpoint = os.environ.get("O2_PROD_ENDPOINT", O2_DEFAULT_ENDPOINT)
    access_key = _read_credential("O2_PROD_ACCESS_KEY")
    secret_key = _read_credential("O2_PROD_SECRET_KEY")
    logger.info("%s: O2 endpoint: %s", name, endpoint)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    objects = [
        (obj["Key"], obj["LastModified"].timestamp())
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    if not objects:
        raise FileNotFoundError(f"{name}: no .parquet files under s3://{bucket}/{prefix}")

    objects.sort(key=lambda x: x[0], reverse=True)
    key, _ = objects[0]

    ts = _parse_snapshot_timestamp(key) or int(time.time())
    age = time.time() - ts
    logger.info("%s: O2 latest file: %s (age: %.0fs)", name, key, age)

    import random

    jitter = random.uniform(0, 30)
    logger.info("%s: downloading s3://%s/%s (jitter=%.1fs)", name, bucket, key, jitter)
    time.sleep(jitter)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    logger.info("%s: downloaded %s bytes from O2", name, f"{len(body):,}")
    return body, ts, f"s3://{bucket}/{key}"


def _download_from_gcs(name: str, prefix: str) -> tuple[bytes, int, str]:
    from google.cloud import storage
    from google.oauth2 import service_account

    sa_key_path = _read_file_credential("shadow.json", [_GCS_CREDENTIALS_DIR, _O2_CREDENTIALS_DIR])
    bucket_name = os.environ.get("GCS_BUCKET", _GCS_DEFAULT_BUCKET)
    prefix = settings.GCS_MIRROR_PREFIX or prefix

    if sa_key_path and Path(sa_key_path).is_file():
        credentials = service_account.Credentials.from_service_account_file(sa_key_path)
        client = storage.Client(credentials=credentials, project=credentials.project_id)
    else:
        client = storage.Client()

    bucket = client.bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".parquet")]
    if not blobs:
        raise FileNotFoundError(f"{name}: no snapshot parquets under gs://{bucket_name}/{prefix}")

    blobs.sort(key=lambda b: b.name, reverse=True)
    latest = blobs[0]

    ts = _parse_snapshot_timestamp(latest.name) or int(time.time())
    logger.info("%s: GCS latest file: %s (age: %.0fs)", name, latest.name, time.time() - ts)

    body = latest.download_as_bytes()
    logger.info("%s: downloaded %s bytes from GCS", name, f"{len(body):,}")
    return body, ts, f"gs://{bucket_name}/{latest.name}"


def _gcs_available() -> bool:
    sa_path = _read_file_credential("shadow.json", [_GCS_CREDENTIALS_DIR, _O2_CREDENTIALS_DIR])
    return sa_path is not None and Path(sa_path).is_file()


OBJECTIVE_COLUMN = "objective"
CARD_TYPE_COLUMN = "unified_card_type"
MEDIA_ARMS_COLUMN = "tweet_media_info_arms"
VIDEO_CARD_TYPES = frozenset({2, 4, 6, 9, 10, 11, 13, 15, 19, 21, 22, 23, 24, 25, 27})
VIDEO_MEDIA_ARM = 3


def _row_bits_from_table(table: pa.Table) -> np.ndarray:
    objectives = table.column(OBJECTIVE_COLUMN).combine_chunks().fill_null(0)
    bits = np.left_shift(1, objectives.to_numpy(zero_copy_only=False).astype(np.int32))
    is_video = np.zeros(table.num_rows, dtype=bool)
    if CARD_TYPE_COLUMN in table.column_names:
        cards = table.column(CARD_TYPE_COLUMN).combine_chunks().fill_null(0)
        is_video |= np.isin(cards.to_numpy(zero_copy_only=False), list(VIDEO_CARD_TYPES))
    if MEDIA_ARMS_COLUMN in table.column_names:
        arms = table.column(MEDIA_ARMS_COLUMN).combine_chunks()
        values = pc.list_flatten(arms).to_numpy(zero_copy_only=False)
        parents = pc.list_parent_indices(arms).to_numpy(zero_copy_only=False)
        is_video[parents[values == VIDEO_MEDIA_ARM]] = True
    return (bits | np.where(is_video, VIDEO_CREATIVE_BIT, 0)).astype(np.int32)


def _parse_parquet_ids(
    body: bytes, *, read_objective: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    columns = ["post_id", "author_id"]
    if read_objective:
        names = pq.ParquetFile(io.BytesIO(body)).schema_arrow.names
        columns += [
            c for c in (OBJECTIVE_COLUMN, CARD_TYPE_COLUMN, MEDIA_ARMS_COLUMN) if c in names
        ]
    table = pq.read_table(io.BytesIO(body), columns=columns)
    pids = table.column("post_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    aids = (
        table.column("author_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    )
    row_bits = _row_bits_from_table(table) if read_objective else None
    return pids, aids, row_bits


def _load_from_o2(
    name: str, uri: str, *, gcs_mirror: bool = False, read_objective: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    bucket, prefix = uri[len(_O2_SCHEME) :].split("/", 1)
    has_gcs = gcs_mirror and _gcs_available()

    try:
        body, ts, source = _download_from_o2(name, bucket, prefix)
        age = time.time() - ts

        if age <= _O2_MAX_STALENESS_SECONDS or not has_gcs:
            if age > _O2_MAX_STALENESS_SECONDS:
                logger.warning(
                    "%s: O2 snapshot is stale (%.0fs) but no GCS fallback configured", name, age
                )
            pids, aids, row_bits = _parse_parquet_ids(body, read_objective=read_objective)
            logger.info("%s: loaded %s rows from %s", name, f"{len(pids):,}", source)
            return pids, aids, row_bits

        logger.info(
            "%s: O2 snapshot stale (%.0fs > %ds), trying GCS fallback",
            name,
            age,
            _O2_MAX_STALENESS_SECONDS,
        )
        try:
            gcs_body, gcs_ts, gcs_source = _download_from_gcs(name, prefix)
            if gcs_ts > ts:
                pids, aids, row_bits = _parse_parquet_ids(gcs_body, read_objective=read_objective)
                logger.info(
                    "%s: loaded %s rows from %s (newer than O2)", name, f"{len(pids):,}", gcs_source
                )
                return pids, aids, row_bits
            else:
                logger.info("%s: GCS snapshot not newer than O2, using O2", name)
                pids, aids, row_bits = _parse_parquet_ids(body, read_objective=read_objective)
                logger.info("%s: loaded %s rows from %s", name, f"{len(pids):,}", source)
                return pids, aids, row_bits
        except Exception as gcs_err:
            logger.warning("%s: GCS fallback failed (%s), using stale O2 snapshot", name, gcs_err)
            pids, aids, row_bits = _parse_parquet_ids(body, read_objective=read_objective)
            logger.info("%s: loaded %s rows from %s", name, f"{len(pids):,}", source)
            return pids, aids, row_bits

    except Exception as o2_err:
        if not has_gcs:
            raise
        logger.warning("%s: O2 download failed (%s), trying GCS fallback", name, o2_err)
        gcs_body, _, gcs_source = _download_from_gcs(name, prefix)
        pids, aids, row_bits = _parse_parquet_ids(gcs_body, read_objective=read_objective)
        logger.info("%s: loaded %s rows from %s (O2 fallback)", name, f"{len(pids):,}", gcs_source)
        return pids, aids, row_bits


class RetrievalCorpus(NamedTuple):
    post_ids: np.ndarray
    author_ids: np.ndarray
    dataset_types: np.ndarray
    post_sids: np.ndarray | None
    objective_bits: np.ndarray | None


class RetrievalDataset(Enum):
    PAD = (0, None, None)
    HOME = (1, *_sid_window("1fav_1day.parquet"))
    IMMERSIVE2Day = (2, *_sid_window("video_2day.parquet"))
    RELEVANT_ADS = (
        3,
        _idx("relevant_ads/v6/post_id_author_id_pair.parquet"),
        _idx("relevant_ads/v6/post_id_author_id_pair.parquet"),
    )
    CAROUSEL_ADS = (
        4,
        _idx("relevant_ads/carousel/post_id_author_id_pair.parquet"),
        _idx("relevant_ads/carousel/post_id_author_id_pair.parquet"),
    )
    EVERGREEN = (5, *_sid_window("video_4to14day.parquet"))
    IMMERSIVENSFW = (6, *_sid_window("nsfw_video_2day.parquet"))
    ACTIVE_ADS = (
        7,
        settings.ADS_INDEX_URI,
        settings.ADS_INDEX_URI,
    )
    IMMERSIVE4Day = (8, *_sid_window("video_4day.parquet"))
    IMAGINE = (9, *_sid_window("imagine_4day.parquet"))
    TAIL = (
        10,
        _idx("post_sid_v5_256x6_tail_snapshots/tail_1day.parquet"),
        None,
    )
    DPA_PRODUCTS = (
        11,
        settings.DPA_INDEX_URI,
        settings.DPA_INDEX_URI,
    )
    HOME_COLD = (13, *_sid_window("1fav_1day.parquet"))
    HOME_HOT = (14, *_sid_window("1fav_1day.parquet"))
    IMMERSIVE32FavVideo14To30Day = (19, *_sid_window("32fav_video_14to30day.parquet"))

    def __new__(cls, value: int, path: str | None = None, backup_path: str | None = None):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.path = path
        obj.backup_path = backup_path
        return obj

    @classmethod
    def home_slices(cls) -> tuple[RetrievalDataset, RetrievalDataset]:
        return (cls.HOME_COLD, cls.HOME_HOT)

    @classmethod
    def expand_home_to_cold_hot(cls, datasets: list[RetrievalDataset]) -> list[RetrievalDataset]:
        out: list[RetrievalDataset] = []
        for ds in datasets:
            if ds is cls.HOME:
                out.extend(cls.home_slices())
            else:
                out.append(ds)
        return out

    @classmethod
    def mask_values(cls, ds_value: int) -> tuple[int, ...]:
        if ds_value == cls.HOME.value:
            return (cls.HOME_COLD.value, cls.HOME_HOT.value, cls.HOME.value)
        return (ds_value,)

    @classmethod
    def home_union_range(cls, ranges: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
        cold = ranges.get(cls.HOME_COLD.value)
        hot = ranges.get(cls.HOME_HOT.value)
        if cold is not None and hot is not None:
            if cold[1] == hot[0]:
                return (cold[0], hot[1])
            return ranges.get(cls.HOME.value)
        return ranges.get(cls.HOME.value) or cold or hot

    def _get_valid_path(self) -> str | None:
        for p in (self.path, self.backup_path):
            if p and (p.startswith(_O2_SCHEME) or Path(p).exists()):
                return p
        return None

    def load(
        self,
        *,
        read_post_sid: bool = False,
        sid_num_levels: int = 6,
        path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        rows = self.load_rows(read_post_sid=read_post_sid, sid_num_levels=sid_num_levels, path=path)
        return None if rows is None else rows[:3]

    def load_rows(
        self,
        *,
        read_post_sid: bool = False,
        sid_num_levels: int = 6,
        path: str | Path | None = None,
        read_objective_bits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
        load_path = str(path) if path is not None else self._get_valid_path()
        if load_path is None:
            logger.warning(f"Skipping {self.name}: path not found")
            return None
        if self is RetrievalDataset.ACTIVE_ADS and not load_path.startswith(_O2_SCHEME):
            raise ValueError(f"ACTIVE_ADS must load the O2 ads snapshot, got {load_path}")

        if load_path.startswith(_O2_SCHEME):
            pids, aids, row_bits = _load_from_o2(
                self.name,
                load_path,
                gcs_mirror=self.name in _GCS_MIRRORED,
                read_objective=read_objective_bits,
            )
            if self == RetrievalDataset.ACTIVE_ADS:
                pids, unique_idx, inverse = np.unique(pids, return_index=True, return_inverse=True)
                logger.info("%s: deduped to %d unique post_ids", self.name, len(pids))
                aids = aids[unique_idx]
                if row_bits is not None:
                    bits = np.zeros(len(pids), dtype=np.int32)
                    np.bitwise_or.at(bits, inverse, row_bits)
                    row_bits = bits
            return pids, aids, None, row_bits

        columns = ["post_id", "author_id"]
        if read_post_sid:
            columns.append("post_sid")
        table = pq.ParquetFile(load_path).read(columns=columns)
        pids = table.column("post_id").combine_chunks().to_numpy(zero_copy_only=True)
        aids = table.column("author_id").combine_chunks().to_numpy(zero_copy_only=True)

        if len(pids) == 0 or len(aids) == 0:
            raise ValueError(f"Read empty arrays from {load_path}: {len(pids)} {len(aids)}")
        assert len(pids) == len(aids), (
            f"Mismatched arrays in {self.name}: {len(pids)} post_ids vs {len(aids)} author_ids"
        )

        post_sids: np.ndarray | None = None
        if "post_sid" in [f.name for f in table.schema]:
            sid_col = table.column("post_sid").combine_chunks()
            n = len(sid_col)
            flat_values = sid_col.values.to_numpy(zero_copy_only=False)
            expected_total = n * sid_num_levels
            if n > 0 and flat_values.size != expected_total:
                raise ValueError(
                    f"post_sid schema invariant violated for {self.name} at {load_path}: "
                    f"expected {n} rows × {sid_num_levels} codes = {expected_total} flat ints, "
                    f"got flat_values.size={flat_values.size}"
                )
            post_sids = np.ascontiguousarray(
                flat_values.reshape(-1, sid_num_levels), dtype=np.int32
            )
            n_with = int((post_sids[:, 0] != -1).sum()) if n > 0 else 0
            logger.info(
                "%s: packed post_sid for %d/%d rows (%.1f%%) into [%d, %d] int32",
                self.name,
                n_with,
                n,
                100.0 * n_with / max(n, 1),
                n,
                sid_num_levels,
            )
        return pids, aids, post_sids, None

    @classmethod
    def loadable(cls) -> list[RetrievalDataset]:
        return [ds for ds in cls if ds._get_valid_path() is not None]

    @classmethod
    def load_datasets(
        cls,
        datasets: list[RetrievalDataset],
        max_posts: int | None = None,
        *,
        read_post_sid: bool = False,
        sid_num_levels: int = 6,
        cold_pool_metadata_path: str | Path | None = None,
        cold_start_max_age_seconds: float = 0.0,
        paths: dict[RetrievalDataset, str | Path] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        corpus = cls.load_corpus(
            datasets,
            max_posts,
            read_post_sid=read_post_sid,
            sid_num_levels=sid_num_levels,
            cold_pool_metadata_path=cold_pool_metadata_path,
            cold_start_max_age_seconds=cold_start_max_age_seconds,
            paths=paths,
        )
        return corpus.post_ids, corpus.author_ids, corpus.dataset_types, corpus.post_sids

    @classmethod
    def load_corpus(
        cls,
        datasets: list[RetrievalDataset],
        max_posts: int | None = None,
        *,
        read_post_sid: bool = False,
        sid_num_levels: int = 6,
        cold_pool_metadata_path: str | Path | None = None,
        cold_start_max_age_seconds: float = 0.0,
        paths: dict[RetrievalDataset, str | Path] | None = None,
        read_objective_bits: bool = False,
    ) -> RetrievalCorpus:
        start = time.time()
        post_ids_list, author_ids_list, types_list, sids_list = [], [], [], []
        bits_list: list[np.ndarray] = []
        any_sid_loaded = False
        home_source: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None | bool = False
        home_paths = {
            str(path)
            for dataset, path in (paths or {}).items()
            if dataset in (cls.HOME, *cls.home_slices())
        }
        if len(home_paths) > 1:
            raise ValueError("HOME, HOME_COLD and HOME_HOT must pin the same source file")
        home_path = next(iter(home_paths), None)

        def _append(
            pids: np.ndarray,
            aids: np.ndarray,
            sids: np.ndarray | None,
            ds_value: int,
            bits: np.ndarray | None = None,
        ) -> None:
            nonlocal any_sid_loaded
            post_ids_list.append(pids)
            author_ids_list.append(aids)
            types_list.append(np.full(len(pids), ds_value, dtype=np.int32))
            bits_list.append(np.zeros(len(pids), dtype=np.int32) if bits is None else bits)
            if read_post_sid:
                if sids is None:
                    sids_list.append(np.full((len(pids), sid_num_levels), -1, dtype=np.int32))
                else:
                    sids_list.append(sids)
                    any_sid_loaded = True

        def _load_home_source() -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
            nonlocal home_source
            if home_source is False:
                home_source = cls.HOME.load(
                    read_post_sid=read_post_sid,
                    sid_num_levels=sid_num_levels,
                    **({"path": home_path} if home_path is not None else {}),
                )
            return None if home_source is False else home_source

        home_cold_keep: list[np.ndarray | None] = []

        def _home_cold_keep(pids: np.ndarray) -> np.ndarray | None:
            if not home_cold_keep:
                meta_path = cold_pool_metadata_path or default_cold_pool_metadata_path()
                mask = cold_pool_mask(pids, meta_path)
                home_cold_keep.append(
                    None
                    if mask is None
                    else mask & fresh_post_mask(pids, cold_start_max_age_seconds)
                )
            return home_cold_keep[0]

        for ds in datasets:
            if ds in cls.home_slices():
                result = _load_home_source()
                if result is None:
                    continue
                pids, aids, sids = result
                cold_keep = _home_cold_keep(pids)
                if cold_keep is None:
                    keep = (
                        np.ones(len(pids), dtype=bool)
                        if ds is cls.HOME_HOT
                        else np.zeros(len(pids), dtype=bool)
                    )
                    logger.warning(
                        "%s: cold-pool metadata missing; %s",
                        ds.name,
                        "keeping all HOME rows" if ds is cls.HOME_HOT else "emitting 0 rows",
                    )
                else:
                    keep = cold_keep if ds is cls.HOME_COLD else ~cold_keep
                n_keep = int(np.count_nonzero(keep))
                logger.info(
                    "%s: %s/%s posts after cold-pool split",
                    ds.name,
                    f"{n_keep:,}",
                    f"{len(pids):,}",
                )
                if n_keep == 0:
                    continue
                _append(
                    pids[keep],
                    aids[keep],
                    None if sids is None else sids[keep],
                    ds.value,
                )
                continue

            path_kwargs = {"path": paths[ds]} if paths and ds in paths else {}
            if read_objective_bits and ds is cls.ACTIVE_ADS:
                rows = ds.load_rows(
                    read_post_sid=read_post_sid,
                    sid_num_levels=sid_num_levels,
                    read_objective_bits=True,
                    **path_kwargs,
                )
                if rows is None:
                    continue
                pids, aids, sids, bits = rows
                _append(pids, aids, sids, ds.value, bits)
                continue

            if (
                result := ds.load(
                    read_post_sid=read_post_sid, sid_num_levels=sid_num_levels, **path_kwargs
                )
            ) is None:
                continue
            pids, aids, sids = result
            _append(pids, aids, sids, ds.value)

        if not post_ids_list:
            raise FileNotFoundError("No dataset files found")

        post_ids = np.concatenate(post_ids_list)
        author_ids = np.concatenate(author_ids_list)
        types = np.concatenate(types_list)
        post_sids: np.ndarray | None = (
            np.concatenate(sids_list) if (read_post_sid and any_sid_loaded) else None
        )
        objective_bits = np.concatenate(bits_list) if read_objective_bits else None
        logger.info(f"Loaded {len(post_ids):,} pairs in {time.time() - start:.1f}s")

        if max_posts is not None and max_posts != len(post_ids):
            if max_posts > len(post_ids):
                pad = max_posts - len(post_ids)
                post_ids = np.pad(post_ids, (0, pad), constant_values=0)
                author_ids = np.pad(author_ids, (0, pad), constant_values=0)
                types = np.pad(types, (0, pad), constant_values=cls.PAD.value).astype(np.int32)
                if post_sids is not None:
                    post_sids = np.pad(post_sids, ((0, pad), (0, 0)), constant_values=-1)
                if objective_bits is not None:
                    objective_bits = np.pad(objective_bits, (0, pad), constant_values=0)
            else:
                post_ids = post_ids[-max_posts:]
                author_ids = author_ids[-max_posts:]
                types = types[-max_posts:]
                if post_sids is not None:
                    post_sids = post_sids[-max_posts:]
                if objective_bits is not None:
                    objective_bits = objective_bits[-max_posts:]

        return RetrievalCorpus(post_ids, author_ids, types, post_sids, objective_bits)


def _bind_deployment_corpus_paths() -> None:
    for member, (path, backup) in settings.RETRIEVAL_DATASET_PATHS.items():
        RetrievalDataset[member].path = path
        RetrievalDataset[member].backup_path = backup


_bind_deployment_corpus_paths()
