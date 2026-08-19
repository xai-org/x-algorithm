import logging
import os
import time

import requests

log = logging.getLogger("gizmoduck_age")

STRATO_URL = (
    os.environ.get("STRATO_BASE_URL", "https://strato.example.com")
    + "/op/fetch/gizmoduck/composite.User"
)

REQUEST_TIMEOUT_SEC = 1.0

REDIS_TTL_SEC = 7 * 24 * 3600

_LOCAL_CACHE_MAX = 100_000
_LOCAL_CACHE: "dict[int, int]" = {}


def _local_cache_put(uid: int, created_at_msec: int) -> None:
    if len(_LOCAL_CACHE) >= _LOCAL_CACHE_MAX:
        _LOCAL_CACHE.pop(next(iter(_LOCAL_CACHE)), None)
    _LOCAL_CACHE[uid] = created_at_msec


def _fetch_created_at_msec(user_id: int) -> int | None:
    try:
        resp = requests.post(
            STRATO_URL,
            json=[user_id, [{}, []]],
            timeout=REQUEST_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        v = body.get("v")
        if not v:
            return None
        cam = v.get("createdAtMsec")
        if cam is None:
            return None
        return int(cam)
    except Exception as e:
        log.debug(f"gizmoduck fetch failed for {user_id}: {e}")
        return None


def get_user_age_hours(
    user_id: int,
    redis_client=None,
    metrics: dict | None = None,
) -> int | None:
    now_msec = int(time.time() * 1000)

    cam = _LOCAL_CACHE.get(user_id)
    if cam is not None:
        if metrics is not None:
            metrics["age_cache_local"] = metrics.get("age_cache_local", 0) + 1
        return max(0, (now_msec - cam) // 3_600_000)

    if redis_client is not None:
        try:
            raw = redis_client.get(f"user_age:{user_id}")
            if raw is not None:
                cam = int(raw)
                _local_cache_put(user_id, cam)
                if metrics is not None:
                    metrics["age_cache_redis"] = metrics.get("age_cache_redis", 0) + 1
                return max(0, (now_msec - cam) // 3_600_000)
        except Exception:
            pass

    cam = _fetch_created_at_msec(user_id)
    if cam is None:
        if metrics is not None:
            metrics["age_strato_fail"] = metrics.get("age_strato_fail", 0) + 1
        return None

    _local_cache_put(user_id, cam)
    if redis_client is not None:
        try:
            redis_client.set(f"user_age:{user_id}", str(cam), ex=REDIS_TTL_SEC)
        except Exception:
            pass
    if metrics is not None:
        metrics["age_strato_ok"] = metrics.get("age_strato_ok", 0) + 1
    return max(0, (now_msec - cam) // 3_600_000)
