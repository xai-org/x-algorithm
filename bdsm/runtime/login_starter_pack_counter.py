import logging
import os
import time

import requests

log = logging.getLogger("login_starter_pack_counter")

STRATO_URL = (
    os.environ.get("STRATO_BASE_URL", "https://strato.example.com")
    + "/op/fetch/onboard/featureSwitches/loginStarterPacksCounter"
)

REQUEST_TIMEOUT_SEC = 1.0

REDIS_TTL_SEC = 60 * 60

_LOCAL_TTL_SEC = 60 * 60
_LOCAL_CACHE_MAX = 100_000
_LOCAL_CACHE: "dict[int, tuple[int, float]]" = {}


def _local_cache_put(uid: int, counter_ms: int) -> None:
    if len(_LOCAL_CACHE) >= _LOCAL_CACHE_MAX:
        _LOCAL_CACHE.pop(next(iter(_LOCAL_CACHE)), None)
    _LOCAL_CACHE[uid] = (counter_ms, time.time())


def _local_cache_get(uid: int) -> int | None:
    entry = _LOCAL_CACHE.get(uid)
    if entry is None:
        return None
    counter_ms, fetched_at = entry
    if time.time() - fetched_at > _LOCAL_TTL_SEC:
        _LOCAL_CACHE.pop(uid, None)
        return None
    return counter_ms


def _fetch_counter_ms(user_id: int) -> int | None:
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
            return -1
        return int(v)
    except Exception as e:
        log.debug(f"login_starter_pack_counter fetch failed for {user_id}: {e}")
        return None


def get_login_pack_age_minutes(
    user_id: int,
    redis_client=None,
    metrics: dict | None = None,
) -> int | None:
    now_msec = int(time.time() * 1000)

    cached = _local_cache_get(user_id)
    if cached is not None:
        if metrics is not None:
            metrics["login_pack_cache_local"] = metrics.get("login_pack_cache_local", 0) + 1
        if cached < 0:
            return None
        return max(0, (now_msec - cached) // 60_000)

    if redis_client is not None:
        try:
            raw = redis_client.get(f"login_pack_ctr:{user_id}")
            if raw is not None:
                cached = int(raw)
                _local_cache_put(user_id, cached)
                if metrics is not None:
                    metrics["login_pack_cache_redis"] = metrics.get("login_pack_cache_redis", 0) + 1
                if cached < 0:
                    return None
                return max(0, (now_msec - cached) // 60_000)
        except Exception:
            pass

    counter_ms = _fetch_counter_ms(user_id)
    if counter_ms is None:
        if metrics is not None:
            metrics["login_pack_strato_fail"] = metrics.get("login_pack_strato_fail", 0) + 1
        return None

    _local_cache_put(user_id, counter_ms)
    if redis_client is not None:
        try:
            redis_client.set(f"login_pack_ctr:{user_id}", str(counter_ms), ex=REDIS_TTL_SEC)
        except Exception:
            pass

    if metrics is not None:
        if counter_ms < 0:
            metrics["login_pack_strato_unset"] = metrics.get("login_pack_strato_unset", 0) + 1
        else:
            metrics["login_pack_strato_set"] = metrics.get("login_pack_strato_set", 0) + 1

    if counter_ms < 0:
        return None
    return max(0, (now_msec - counter_ms) // 60_000)
