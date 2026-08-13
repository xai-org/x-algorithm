import logging
import os
import time

import requests

log = logging.getLogger("starter_pack_follows")

STRATO_URL = (
    os.environ.get("STRATO_BASE_URL", "https://strato.example.com")
    + "/op/fetch/starterpacks/followPacksStore"
)

REQUEST_TIMEOUT_SEC = 1.0

REDIS_TTL_SEC = 6 * 60 * 60
_LOCAL_TTL_SEC = 6 * 60 * 60
_LOCAL_CACHE_MAX = 100_000
_LOCAL_CACHE: "dict[int, tuple[int, float]]" = {}


def _local_cache_put(uid: int, pack_count: int) -> None:
    if len(_LOCAL_CACHE) >= _LOCAL_CACHE_MAX:
        _LOCAL_CACHE.pop(next(iter(_LOCAL_CACHE)), None)
    _LOCAL_CACHE[uid] = (pack_count, time.time())


def _local_cache_get(uid: int) -> int | None:
    entry = _LOCAL_CACHE.get(uid)
    if entry is None:
        return None
    pack_count, fetched_at = entry
    if time.time() - fetched_at > _LOCAL_TTL_SEC:
        _LOCAL_CACHE.pop(uid, None)
        return None
    return pack_count


def _fetch_pack_count(user_id: int) -> int | None:
    try:
        resp = requests.post(
            STRATO_URL,
            json=[user_id, None],
            timeout=REQUEST_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        v = body.get("v")
        if v is None:
            return 0
        return len(v)
    except Exception as e:
        log.debug(f"starter_pack_follows fetch failed for {user_id}: {e}")
        return None


def get_followed_pack_count(
    user_id: int,
    redis_client=None,
    metrics: dict | None = None,
) -> int | None:
    cached = _local_cache_get(user_id)
    if cached is not None:
        if metrics is not None:
            metrics["starter_pack_cache_local"] = metrics.get("starter_pack_cache_local", 0) + 1
        return cached

    if redis_client is not None:
        try:
            raw = redis_client.get(f"starter_packs:{user_id}")
            if raw is not None:
                cached = int(raw)
                _local_cache_put(user_id, cached)
                if metrics is not None:
                    metrics["starter_pack_cache_redis"] = (
                        metrics.get("starter_pack_cache_redis", 0) + 1
                    )
                return cached
        except Exception:
            pass

    pack_count = _fetch_pack_count(user_id)
    if pack_count is None:
        if metrics is not None:
            metrics["starter_pack_strato_fail"] = metrics.get("starter_pack_strato_fail", 0) + 1
        return None

    _local_cache_put(user_id, pack_count)
    if redis_client is not None:
        try:
            redis_client.set(f"starter_packs:{user_id}", str(pack_count), ex=REDIS_TTL_SEC)
        except Exception:
            pass

    if metrics is not None:
        if pack_count > 0:
            metrics["starter_pack_strato_set"] = metrics.get("starter_pack_strato_set", 0) + 1
        else:
            metrics["starter_pack_strato_unset"] = metrics.get("starter_pack_strato_unset", 0) + 1

    return pack_count
