# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import json
import logging
import os
import pathlib
import random
import tempfile
import time

rank_logger = logging.getLogger("rank")

TREE_DEK_NAME = "_DEK"
TREE_DEK_CLAIM_NAME = "_DEK.claim"
PUBLISH_TIMEOUT_SECS = 120.0

_TRANSIENT_KMS_MARKERS = ("429", "502", "503", "504", "KMS transport")
_KMS_RETRY_BUDGET_SECS = 300.0


def _retry_delay(error: BaseException, attempt: int, deadline: float) -> float | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not any(m in str(error) for m in _TRANSIENT_KMS_MARKERS):
        return None
    return min(min(2.0**attempt, 30.0) * (0.5 + random.random()), remaining)


def _unwrap_with_backoff(kms_client, dek_path, deadline: float) -> bytes:
    import xai_kms

    attempt = 0
    while True:
        try:
            return bytes(xai_kms.nfs.unwrap_shared_dek(kms_client, dek_path))
        except Exception as error:
            delay = _retry_delay(error, attempt, deadline)
            if delay is None:
                raise
            rank_logger.warning(
                "KMS unwrap of %s failed transiently (%s); retrying in %.1fs (attempt %d)",
                dek_path,
                error,
                delay,
                attempt + 1,
            )
            time.sleep(delay)
            attempt += 1


def _read_wrapped_dek(dek_path: pathlib.Path) -> dict | None:
    try:
        entry = json.loads(dek_path.read_text())
        return entry if entry.get("wrapped") else None
    except (OSError, ValueError):
        return None


def publish_tree_dek(path: pathlib.Path, kms_client) -> tuple[bytes, str, dict, dict]:
    import xai_kms

    dek_path = path / TREE_DEK_NAME
    claim = path / TREE_DEK_CLAIM_NAME

    deadline = time.monotonic() + _KMS_RETRY_BUDGET_SECS
    mint_failures = 0
    if _read_wrapped_dek(dek_path) is not None:
        rank_logger.info("adopted existing _DEK at %s", path)
    while (entry := _read_wrapped_dek(dek_path)) is None:
        try:
            claim.open("x").close()
            won_claim = True
        except FileExistsError:
            won_claim = False
        if won_claim:
            try:
                if _read_wrapped_dek(dek_path) is None:
                    xai_kms.nfs.write_shared_dek(kms_client, dek_path)
                    rank_logger.info(
                        "minted and KMS-wrapped new DEK at %s (key_id=%s)",
                        path,
                        (_read_wrapped_dek(dek_path) or {}).get("key_id"),
                    )
                else:
                    rank_logger.info("adopted existing _DEK at %s", path)
            except BaseException as error:
                if _read_wrapped_dek(dek_path) is None:
                    claim.unlink(missing_ok=True)
                delay = _retry_delay(error, mint_failures, deadline)
                if delay is None:
                    raise
                rank_logger.warning(
                    "KMS mint at %s failed transiently (%s); re-racing in %.1fs (attempt %d)",
                    path,
                    error,
                    delay,
                    mint_failures + 1,
                )
                time.sleep(delay)
                mint_failures += 1
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for the wrapped DEK at {dek_path}; its minter "
                "(the rank holding the .claim marker) likely died before publishing"
            )
        rank_logger.debug("waiting for wrapped DEK at %s", dek_path)
        time.sleep(0.1)
    raw = _unwrap_with_backoff(kms_client, dek_path, deadline)
    return raw, entry["wrapped"], entry.get("context") or {}, entry


_ENVELOPE_HEADER_SIZE = 4096
_ENVELOPE_FIXED_LEN = 36
_DERIVED_PREFIX = "xai-dek1:"


def adopt_tree_dek(path: pathlib.Path, kms_client) -> bytes:
    dek_path = path / TREE_DEK_NAME
    if dek_path.exists():
        entry = _read_wrapped_dek(dek_path) or {}
        raw = _unwrap_with_backoff(kms_client, dek_path, time.monotonic() + _KMS_RETRY_BUDGET_SECS)
        key_id = entry.get("key_id")
    else:
        raw = _unwrap_header_master(kms_client, path)
        key_id = "header"
    rank_logger.info("KMS unwrap OK for %s (key_id=%s)", path, key_id)
    return raw


def _unwrap_header_master(kms_client, path: pathlib.Path) -> bytes:
    wrapped, context = _header_wrapped_master(path)
    entry = {"key_id": "header", "context": context, "wrapped": wrapped}
    fd, tmp = tempfile.mkstemp(prefix="xai-ckpt-dek-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f)
        return _unwrap_with_backoff(kms_client, tmp, time.monotonic() + _KMS_RETRY_BUDGET_SECS)
    finally:
        os.unlink(tmp)


def _header_wrapped_master(path: pathlib.Path) -> tuple[str, dict]:
    envelope = _first_envelope(path)
    head = envelope.read_bytes()[:_ENVELOPE_HEADER_SIZE]
    if len(head) < _ENVELOPE_FIXED_LEN or head[:8] != b"XAIENC01":
        raise ValueError(f"not an envelope: {envelope}")
    wrapped_len = int.from_bytes(head[28:32], "big")
    context_len = int.from_bytes(head[32:36], "big")
    start = _ENVELOPE_FIXED_LEN
    field = head[start : start + wrapped_len].decode()
    context = json.loads(head[start + wrapped_len : start + wrapped_len + context_len])
    if not field.startswith(_DERIVED_PREFIX):
        raise ValueError(
            f"unsupported wrapped-data-key format at {envelope}: expected "
            f"'{_DERIVED_PREFIX}<salt>:<token>'; convert the tree to the "
            "master-DEK format to load it"
        )
    _salt, master = field[len(_DERIVED_PREFIX) :].split(":", 1)
    if not master:
        raise ValueError(f"malformed xai-dek1 header at {envelope}")
    return master, context


def _first_envelope(path: pathlib.Path) -> pathlib.Path:
    meta = path / "_METADATA"
    if meta.is_file():
        with meta.open("rb") as f:
            if f.read(8) == b"XAIENC01":
                return meta
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(filenames):
            candidate = pathlib.Path(dirpath) / name
            try:
                with candidate.open("rb") as f:
                    if f.read(8) == b"XAIENC01":
                        return candidate
            except OSError:
                continue
    raise FileNotFoundError(f"no envelope in {path}")
