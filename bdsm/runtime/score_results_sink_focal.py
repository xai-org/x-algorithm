#!/usr/bin/env python3

import argparse
import datetime
import logging
import os
import time
import threading
import uuid as _uuid
import zlib
import dataclasses
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

from gizmoduck_age import get_user_age_hours
from login_starter_pack_counter import get_login_pack_age_minutes
from pipeline_security import kafka_ssl_config, safe_pickle_loads
from starter_pack_follows import get_followed_pack_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("results_sink")

import sys as _sys

_sys.path.insert(0, os.environ.get("ABUSE_TRACE_LIB", ""))
try:
    import abuse_trace as _abuse_trace

    _trace = _abuse_trace.get_emitter()
except Exception as _e:
    log.warning(f"trace emitter unavailable, continuing without trace: {_e}")

    class _NoopTrace:
        enabled = False

        def emit(self, **kwargs):
            pass

        def is_active(self, uid):
            return False

        def is_active_many(self, uids):
            return set()

        def mark_active(self, uid):
            pass

        def get_metrics(self):
            return {"enabled": False}

    _trace = _NoopTrace()


def _challenge_salt() -> bytes:
    salt = os.environ.get("BDSM_CHALLENGE_SALT", "")
    if not salt:
        raise RuntimeError("BDSM_CHALLENGE_SALT env var is required for challenge routing")
    return salt.encode()


DEFAULT_BOOTSTRAP = "localhost:9092"

HEAD_NAMES = {
    0: "FollowBot",
    1: "LikeBot",
    2: "EngagementAmplifier",
    3: "ReplySpamBot",
    4: "TweetSpamBot",
    5: "RTBot",
    6: "MultiActionBot",
    7: "LegitimateUser",
}


def _resolve_head_names(_model_version):
    return HEAD_NAMES


# Public-release note: per-head appeal-note templates are REDACTED
# (the prose AND the key_actions each head interpolates from the
# histogram). What ships: MIN_ACTIONS gate, dominant bot-head pick,
# and the sentinel "<redacted>" plus a [model: Head=score] suffix.
# Production fills placeholders from an internal table that is not
# part of this release. ActionName proto enums are untouched.
_REDACTED_TEMPLATE = "<redacted>"

_ENFORCEMENT_TEMPLATES = {
    "FollowBot": _REDACTED_TEMPLATE,
    "LikeBot": _REDACTED_TEMPLATE,
    "EngagementAmplifier": _REDACTED_TEMPLATE,
    "ReplySpamBot": _REDACTED_TEMPLATE,
    "TweetSpamBot": _REDACTED_TEMPLATE,
    "RTBot": _REDACTED_TEMPLATE,
    "MultiActionBot": _REDACTED_TEMPLATE,
}


def build_enforcement_note(head_scores_list, action_hist_list):
    """Build an enforcement note from head scores + action histogram.

    Returns None when no bot head is above 0.5 or the sequence is shorter
    than 30 actions. In this public release the note is the sentinel
    "<redacted>" plus a model-head suffix; production interpolates from a
    private template table (prose and key_actions).
    """
    if not head_scores_list:
        return None

    total = sum(h["cnt"] for h in (action_hist_list or ()))
    if total < 30:
        return None

    bot_heads = [
        (h["head_name"], h["score"])
        for h in head_scores_list
        if h["head_name"] != "LegitimateUser" and h["score"] > 0.5
    ]
    if not bot_heads:
        return None

    bot_heads.sort(key=lambda x: x[1], reverse=True)
    dominant_name, dominant_score = bot_heads[0]
    note = _ENFORCEMENT_TEMPLATES.get(dominant_name, _REDACTED_TEMPLATE)
    note += f" [model: {dominant_name}={dominant_score:.2f}"
    if len(bot_heads) > 1:
        secondary = ", ".join(f"{n}={s:.2f}" for n, s in bot_heads[1:3])
        note += f", also: {secondary}"
    note += "]"
    return note


def _build_flag_config(head_names):
    indices = [
        i
        for i, n in head_names.items()
        if "legit" not in n.lower() and "legitimate" not in n.lower()
    ]
    thresholds = np.array([0.5] * len(indices), dtype=np.float32)
    return indices, thresholds


def claim_action(
    r, action: str, uid: int, ttl_sec: int, max_per_window: int = 1, fail_open: bool = False
) -> bool:
    if ttl_sec <= 0:
        return True
    if r is None:
        return fail_open
    key = f"act:{action}:{uid}"
    try:
        if max_per_window <= 1:
            return bool(r.set(key, "1", nx=True, ex=ttl_sec))
        n = r.incr(key)
        if n == 1:
            r.expire(key, ttl_sec)
        return n <= max_per_window
    except Exception:
        return fail_open


_CHALLENGE_ARKOSE_CAPTCHA = ("enforcement_cusp_arkose", "enforcement_cusp_captcha")
_CHALLENGE_LIVENESS_LABEL = "enforcement_cusp_liveness"


def pick_challenge_label(uid: int, liveness_per_10k: int, metrics=None) -> str:
    if (
        liveness_per_10k > 0
        and zlib.crc32(_challenge_salt() + b"challenge_liveness:" + str(uid).encode()) % 10000
        < liveness_per_10k
    ):
        if metrics is not None:
            metrics["challenge_pick_liveness"] += 1
        return _CHALLENGE_LIVENESS_LABEL
    label = _CHALLENGE_ARKOSE_CAPTCHA[zlib.crc32(str(uid).encode()) % 2]
    if metrics is not None:
        metrics[
            "challenge_pick_arkose" if label.endswith("arkose") else "challenge_pick_captcha"
        ] += 1
    return label


def client_dwell_dropout(r, uid: int, args, metrics) -> bool:
    key = f"client_dwell_dropout:{uid}"
    if r is not None:
        try:
            cached = r.get(key)
            if cached is not None:
                metrics["dwell_dropout_cache_hits"] = metrics.get("dwell_dropout_cache_hits", 0) + 1
                return cached == b"1" or cached == "1"
        except Exception:
            pass
    metrics["dwell_dropout_checks"] = metrics.get("dwell_dropout_checks", 0) + 1
    try:
        import urllib.request
        import json as _json

        url = args.dwell_dropout_check_url.rstrip("/") + f"/client_dwell_dropout/{uid}"
        with urllib.request.urlopen(url, timeout=args.dwell_dropout_timeout_sec) as resp:
            data = _json.loads(resp.read().decode())
        if "client_dwell_dropout" not in data:
            raise ValueError(str(data.get("error", "no field")))
        val = bool(data["client_dwell_dropout"])
    except Exception:
        metrics["dwell_dropout_errors"] = metrics.get("dwell_dropout_errors", 0) + 1
        return not args.dwell_dropout_fail_open
    if r is not None:
        try:
            r.set(key, b"1" if val else b"0", ex=int(args.dwell_dropout_cache_ttl_days * 86400))
        except Exception:
            pass
    return val


_CUSP_BUCKET_LABELS = _CHALLENGE_ARKOSE_CAPTCHA


@dataclass(frozen=True)
class SinkPolicy:
    version: str = "baked-in-defaults"
    source: str = "defaults"
    min_actions_for_enforcement: int = 30
    thresholds: dict = field(
        default_factory=lambda: {
            "FollowBot": (9.99, 9.99),
            "EngagementAmplifier": (9.99, 9.99),
            "RTBot": (9.99, 9.99),
        }
    )
    cusp_delta: float = 9.99
    cusp_heads: dict = field(
        default_factory=lambda: {
            "EngagementAmplifier": (9.99, 9.99),
        }
    )
    paused_liveness_thresholds: dict = field(
        default_factory=lambda: {
            "LikeBot": (9.99, 9.99),
            "MultiActionBot": (9.99, 9.99),
        }
    )
    spam_bounce_thresholds: dict = field(
        default_factory=lambda: {
            "TweetSpamBot": (9.99, 9.99),
            "ReplySpamBot": (9.99, 9.99),
        }
    )
    spam_bounce_action_key: dict = field(
        default_factory=lambda: {
            "TweetSpamBot": "create",
            "ReplySpamBot": "reply",
        }
    )
    official_client_app_ids: frozenset = frozenset({3033300, 129032, 258901, 191841})
    reply_spam_hard_suspend_tau: float = 9.99


DEFAULT_POLICY = SinkPolicy()

_POLICY_2TUPLE_TABLES = (
    "thresholds",
    "cusp_heads",
    "paused_liveness_thresholds",
    "spam_bounce_thresholds",
)


def _load_policy(path: str | None) -> SinkPolicy:
    resolved = (
        path
        or os.environ.get("BDSM_SINK_POLICY", "")
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sink_policy.yaml")
    )
    if not os.path.exists(resolved):
        log.info(f"sink policy: baked-in defaults (no policy file at {resolved})")
        return DEFAULT_POLICY
    try:
        import yaml
    except ImportError:
        log.warning(
            f"sink policy: pyyaml unavailable, IGNORING {resolved}; using baked-in defaults"
        )
        return DEFAULT_POLICY
    with open(resolved) as f:
        raw = yaml.safe_load(f) or {}
    known = {f_.name for f_ in dataclasses.fields(SinkPolicy)} - {"source"}
    unknown = set(raw) - known - {"notes"}
    if unknown:
        raise ValueError(f"sink policy {resolved}: unknown keys {sorted(unknown)}")
    kw = {}
    for k, v in raw.items():
        if k == "notes":
            continue
        if k in _POLICY_2TUPLE_TABLES:
            kw[k] = {h: (float(t[0]), float(t[1])) for h, t in v.items()}
        elif k == "official_client_app_ids":
            kw[k] = frozenset(int(x) for x in v)
        elif k == "spam_bounce_action_key":
            kw[k] = {str(h): str(a) for h, a in v.items()}
        elif k == "min_actions_for_enforcement":
            kw[k] = int(v)
        elif k in ("cusp_delta", "reply_spam_hard_suspend_tau"):
            kw[k] = float(v)
        else:
            kw[k] = str(v)
    pol = SinkPolicy(source=resolved, **kw)
    log.info(f"sink policy: LOADED {resolved} (version={pol.version})")
    return pol


@dataclass
class _Decision:
    enforcement_met: bool = False
    enforcement_head: str | None = None
    age_gate_skip_reason: str | None = None
    starter_pack_challenge: bool = False
    sp_emitted: bool = False
    tweet_create_dominant: bool = False
    cusp_met: bool = False
    cusp_head: str | None = None
    lv_head: str | None = None
    lv_head_emitted: bool = False
    sb_head: str | None = None
    sb_emitted: bool = False


def _parse_args():
    parser = argparse.ArgumentParser(description="Score results sink (BQ + Redis)")
    parser.add_argument("--kafka-bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--kafka-username", default="kafka-user")
    parser.add_argument("--kafka-password", required=True)
    parser.add_argument("--kafka-topic", default="abuse_scored_results")
    parser.add_argument("--kafka-group", default="abuse-results-sink")
    parser.add_argument("--redis-host", default="abuse-score-redis")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--cooldown-hours",
        type=float,
        default=4.0,
        help="DEPRECATED: legacy uniform cooldown. Used only if --cooldown-low-hours and "
        "--cooldown-high-hours are both unset.",
    )
    parser.add_argument(
        "--cooldown-low-hours",
        type=float,
        default=1.0,
        help="Cooldown for users with <--min-actions-for-long-cooldown total actions. "
        "Short so we re-evaluate fast-ramping accounts soon.",
    )
    parser.add_argument(
        "--cooldown-high-hours",
        type=float,
        default=24.0,
        help="Cooldown for users with >= --min-actions-for-long-cooldown total actions. "
        "Long because we already had enough data for a meaningful score.",
    )
    parser.add_argument(
        "--min-actions-for-long-cooldown",
        type=int,
        default=30,
        help="Action-count threshold separating short vs long cooldown buckets. "
        "Default matches MIN_ACTIONS_FOR_ENFORCEMENT (30).",
    )
    parser.add_argument(
        "--bq-project", default="your-gcp-project", help="GCP project for the scores table."
    )
    parser.add_argument(
        "--bq-dataset", default="abuse", help="BigQuery dataset for the scores table."
    )
    parser.add_argument("--bq-table", default="abuse_scores_v3_new")
    parser.add_argument(
        "--model-version",
        default="v3_6head_discrim_step2000",
        help="Model version string — determines head names and BQ model_version column",
    )
    parser.add_argument(
        "--output-topic",
        default="abuse.v3.score_results",
        help="Kafka topic to publish protobuf ScoreResult messages to",
    )
    parser.add_argument("--health-port", type=int, default=8081)
    parser.add_argument(
        "--new-user-age-hours",
        type=int,
        default=48,
        help="Skip FollowBot enforcement for accounts younger than this many hours. "
        "Set to 0 to disable the gate entirely.",
    )
    parser.add_argument(
        "--login-pack-window-min",
        type=int,
        default=1440,
        help="Skip FollowBot enforcement when the user was shown the login "
        "starter pack (loginStarterPacksCounter Strato column) within "
        "this many minutes. Set to 0 to disable the gate entirely. "
        "Default 1440 (24h); empirical p100 sus_ts - counter_ts in 7d "
        "of FollowBot suspensions was ~17.5h.",
    )
    parser.add_argument(
        "--starter-pack-gate-mode",
        default="off",
        choices=("off", "dry_run", "challenge"),
        help="FollowBot starter-pack-follows gate (starterpacks/followPacksStore "
        "Strato column, no TTL — covers resurrected accounts the 14d "
        "loginStarterPacksCounter misses). Only consulted when FollowBot "
        "met its 2D threshold, so Strato load is bounded to the FollowBot "
        "rate. 'off' (default) = disabled. 'dry_run' = suspend as today "
        "but add an inert enforcement_starter_pack_would_skip audit label. "
        "'challenge' = suppress suspension and route the user to the "
        "deterministic arkose/captcha split instead (pack membership is "
        "common among legit users — 21.8%% base rate in follow-heavy "
        "legit-scored users, 2026-07-15 — so challenge, never free-pass.)",
    )
    parser.add_argument(
        "--cusp-liveness-sample-per-10k",
        type=int,
        default=0,
        help="Tiny-volume canary: route this many out of every 10,000 "
        "cusp-eligible users to the new `enforcement_cusp_liveness` "
        "challenge instead of the arkose/captcha split. Stable per "
        "user_id (salted CRC32) so the same user is consistently in "
        "or out of the canary across re-scores. 0 disables (default); "
        "e.g. 10 ≈ 0.1%% of cusp-eligible users.",
    )
    parser.add_argument(
        "--challenge-liveness-per-10k",
        type=int,
        default=0,
        help="Weighted liveness share of the terminal challenge split used by "
        "EVERY bounce lane (EA cusp, paused-head LikeBot/MAB, starter-pack "
        "FollowBot, official-client spam bounce). N out of every 10,000 "
        "bounced users (stable SALTED CRC32 of user_id) get the "
        "enforcement_cusp_liveness label (Bouncer spam_liveness_check via "
        "the enforcement service act_cusp_liveness rule); the rest keep "
        "their historical unsalted CRC32(uid) %% 2 arkose/captcha "
        "assignment. 0 (default) = liveness off, pure 50/50 "
        "arkose/captcha. 9000 = 90%% liveness / 5%% arkose / 5%% recaptcha "
        "(owner directive 2026-08-12).",
    )
    parser.add_argument(
        "--liveness-cooldown-days",
        type=float,
        default=0.0,
        help="Actioning dedup: never send the same user to a liveness "
        "challenge more than --liveness-max-per-window times within "
        "this many days. Reuses the model-side Redis (abuse-score-redis) "
        "with a dedicated act:liveness:<uid> key (SET NX, fixed window). "
        "0 disables (default, no-op); set 7 in the deploy for a 7-day cap. "
        "On dedup, the user falls back to the arkose/captcha split.",
    )
    parser.add_argument(
        "--liveness-max-per-window",
        type=int,
        default=1,
        help="Max liveness challenges per user per --liveness-cooldown-days "
        "window. Default 1 (at most once per window).",
    )
    parser.add_argument(
        "--liveness-cooldown-fail-open",
        action="store_true",
        help="If set, emit liveness when Redis is unavailable (fail OPEN, may "
        "exceed the cap during an outage). Default fail CLOSED: skip "
        "liveness (fall back to arkose/captcha) so the cap is never exceeded.",
    )
    parser.add_argument(
        "--dwell-dropout-check-url",
        default="",
        help="If set (e.g. http://dwell-dropout-checker:8080), consult the "
        "client_dwell_dropout service before sending a canary user to "
        "liveness; skip (fall back to arkose/captcha) users whose dwell "
        "telemetry dropped out during the adblocker outage. Empty = disabled.",
    )
    parser.add_argument(
        "--dwell-dropout-dry-run",
        action="store_true",
        help="Log/count would-be dwell-dropout skips but STILL emit liveness "
        "(validation mode). Default off = actually skip.",
    )
    parser.add_argument(
        "--dwell-dropout-fail-open",
        action="store_true",
        help="On checker error, emit liveness anyway (fail OPEN). Default fail "
        "CLOSED: treat errors as dropout and skip liveness to protect FPs.",
    )
    parser.add_argument(
        "--dwell-dropout-cache-ttl-days",
        type=float,
        default=5.0,
        help="TTL for cached client_dwell_dropout:<uid> results.",
    )
    parser.add_argument(
        "--dwell-dropout-timeout-sec",
        type=float,
        default=1.5,
        help="HTTP timeout for the dwell-dropout checker call.",
    )
    parser.add_argument(
        "--paused-head-liveness-heads",
        default="",
        help="Comma-separated subset of the PAUSED suspension heads "
        "(LikeBot, MultiActionBot) to re-enable as the SOFT liveness "
        "challenge (NOT suspension) at their historical 2D thresholds "
        "(paused_liveness_thresholds in sink_policy.yaml). Empty = off "
        "(default). e.g. 'MultiActionBot' or 'LikeBot,MultiActionBot'.",
    )
    parser.add_argument(
        "--paused-head-liveness-sample-per-10k",
        type=int,
        default=0,
        help="Per-10,000 coin-toss sample of matching users to actually route to "
        "liveness (stable per user_id via salted CRC32). 0 = off (default). "
        "Inherits the 7-day liveness cooldown and the dwell-dropout exemption "
        "(adblocker victims are excluded — they are not bots).",
    )
    parser.add_argument(
        "--spam-bounce-heads",
        default="",
        help="Comma-separated subset of the DISABLED spam heads "
        "(TweetSpamBot, ReplySpamBot) to BOUNCE with an arkose/captcha "
        "challenge (NOT suspension) at their Tier-1 2D thresholds "
        "(spam_bounce_thresholds in sink_policy.yaml), gated on the "
        "user's spam-type actions being >= --spam-bounce-dom-frac via ONE "
        "official first-party client (RWeb/iPhone/Android/iPad, ids from "
        "client_utils.rs). Legit API services post via third-party OAuth "
        "apps and never pass this gate. Empty = off (default).",
    )
    parser.add_argument(
        "--spam-bounce-sample-per-10k",
        type=int,
        default=0,
        help="Per-10,000 stable sample (salted CRC32 of user_id) of gate-passing "
        "users to actually bounce. 0 = off (default); 100 = 1%%; "
        "10000 = everyone. This is the ramp knob.",
    )
    parser.add_argument(
        "--spam-bounce-mode",
        default="dry_run",
        choices=("dry_run", "challenge"),
        help="dry_run (default) = append only the inert audit label "
        "spam_bounce_would_challenge (enforcement service ignores it). "
        "challenge = append the deterministic challenge label from the "
        "shared weighted picker (liveness/arkose/captcha per "
        "--challenge-liveness-per-10k; pure 50/50 arkose/captcha "
        "when that is 0).",
    )
    parser.add_argument(
        "--spam-bounce-cooldown-days",
        type=float,
        default=7.0,
        help="Actioning dedup for challenge mode: at most one bounce per user "
        "per window, via its OWN act:spam_bounce:<uid> Redis key (never "
        "shared with act:liveness). 0 disables. Fails CLOSED on Redis "
        "errors (skip the challenge rather than exceed the cap).",
    )
    parser.add_argument(
        "--spam-bounce-dom-frac",
        type=float,
        default=0.90,
        help="Minimum fraction of the user's SERVER_TWEET_CREATE (TweetSpamBot) "
        "/ SERVER_TWEET_REPLY (ReplySpamBot) actions on the single dominant "
        "official client. The prefetcher computes the fraction with "
        "null/unknown app ids in the denominator (fail-safe).",
    )
    parser.add_argument(
        "--policy-file",
        default="",
        help="Path to the reviewed sink_policy.yaml holding the enforcement "
        "policy tables (2D thresholds, cusp band, paused/spam-bounce "
        "thresholds, official-client registry, hard-suspend tau). "
        "Resolution: this flag > BDSM_SINK_POLICY env > sink_policy.yaml "
        "next to this file > baked-in defaults. A present-but-invalid "
        "file fails startup loudly.",
    )
    return parser.parse_args()


def _init_metrics() -> dict:
    return {
        "consumed": 0,
        "scored": 0,
        "flagged": 0,
        "bq_inserted": 0,
        "bq_errors": 0,
        "redis_sets": 0,
        "redis_sets_low": 0,
        "redis_sets_high": 0,
        "liveness_emitted": 0,
        "liveness_dedup_skipped": 0,
        "challenge_pick_liveness": 0,
        "challenge_pick_arkose": 0,
        "challenge_pick_captcha": 0,
        "paused_head_liveness_emitted": 0,
        "liveness_skipped_dwell_dropout": 0,
        "enforcement_skipped_dwell_dropout": 0,
        "dwell_dropout_checks": 0,
        "dwell_dropout_cache_hits": 0,
        "dwell_dropout_errors": 0,
        "age_gate_skipped_followbot": 0,
        "age_gate_silenced_unknown": 0,
        "age_cache_local": 0,
        "age_cache_redis": 0,
        "age_strato_ok": 0,
        "age_strato_fail": 0,
        "login_pack_gate_skipped_followbot": 0,
        "login_pack_cache_local": 0,
        "login_pack_cache_redis": 0,
        "login_pack_strato_set": 0,
        "login_pack_strato_unset": 0,
        "login_pack_strato_fail": 0,
        "starter_pack_gate_challenged_followbot": 0,
        "starter_pack_gate_would_skip": 0,
        "spam_bounce_would_challenge": 0,
        "spam_bounce_challenged": 0,
        "spam_bounce_dedup_skipped": 0,
        "spam_bounce_no_dom_data": 0,
        "reply_spam_hard_suspend": 0,
        "reply_spam_hard_suspend_no_dom": 0,
        "starter_pack_cache_local": 0,
        "starter_pack_cache_redis": 0,
        "starter_pack_strato_set": 0,
        "starter_pack_strato_unset": 0,
        "starter_pack_strato_fail": 0,
        "_start": time.time(),
    }


def _build_policy_config(args, pol):
    liveness_cooldown_sec = int(args.liveness_cooldown_days * 86400)
    paused_liveness_heads = [
        h.strip()
        for h in args.paused_head_liveness_heads.split(",")
        if h.strip() in pol.paused_liveness_thresholds
    ]
    spam_bounce_heads = [
        h.strip()
        for h in args.spam_bounce_heads.split(",")
        if h.strip() in pol.spam_bounce_thresholds
    ]
    liveness_cfg = {
        "policy_version": pol.version,
        "policy_source": pol.source,
        "policy_thresholds": {k: list(v) for k, v in pol.thresholds.items()},
        "policy_cusp_delta": pol.cusp_delta,
        "policy_cusp_heads": {k: list(v) for k, v in pol.cusp_heads.items()},
        "policy_paused_liveness_thresholds": {
            k: list(v) for k, v in pol.paused_liveness_thresholds.items()
        },
        "policy_spam_bounce_thresholds": {
            k: list(v) for k, v in pol.spam_bounce_thresholds.items()
        },
        "policy_official_client_app_ids": sorted(pol.official_client_app_ids),
        "policy_min_actions_for_enforcement": pol.min_actions_for_enforcement,
        "liveness_dedup_enabled": liveness_cooldown_sec > 0,
        "liveness_cooldown_days": args.liveness_cooldown_days,
        "liveness_cooldown_sec": liveness_cooldown_sec,
        "liveness_max_per_window": args.liveness_max_per_window,
        "liveness_cooldown_fail_open": args.liveness_cooldown_fail_open,
        "cusp_liveness_sample_per_10k": args.cusp_liveness_sample_per_10k,
        "challenge_liveness_per_10k": args.challenge_liveness_per_10k,
        "dwell_dropout_check_url": args.dwell_dropout_check_url or None,
        "dwell_dropout_enabled": bool(args.dwell_dropout_check_url),
        "dwell_dropout_dry_run": args.dwell_dropout_dry_run,
        "dwell_dropout_fail_open": args.dwell_dropout_fail_open,
        "dwell_dropout_cache_ttl_days": args.dwell_dropout_cache_ttl_days,
        "paused_head_liveness_heads": paused_liveness_heads,
        "paused_head_liveness_sample_per_10k": args.paused_head_liveness_sample_per_10k,
        "starter_pack_gate_mode": args.starter_pack_gate_mode,
        "spam_bounce_heads": spam_bounce_heads,
        "spam_bounce_sample_per_10k": args.spam_bounce_sample_per_10k,
        "spam_bounce_mode": args.spam_bounce_mode,
        "spam_bounce_cooldown_days": args.spam_bounce_cooldown_days,
        "spam_bounce_dom_frac": args.spam_bounce_dom_frac,
        "reply_spam_hard_suspend_tau": pol.reply_spam_hard_suspend_tau,
    }
    if liveness_cfg["liveness_dedup_enabled"]:
        log.info(
            f"Liveness actioning-dedup ENABLED: max {args.liveness_max_per_window} "
            f"per {args.liveness_cooldown_days}d per user "
            f"(key act:liveness:<uid>, fail_{'open' if args.liveness_cooldown_fail_open else 'closed'})"
        )
    else:
        log.info("Liveness actioning-dedup DISABLED (--liveness-cooldown-days=0)")
    if args.challenge_liveness_per_10k > 0:
        _lv_pct = args.challenge_liveness_per_10k / 100.0
        _rest_pct = (10000 - args.challenge_liveness_per_10k) / 200.0
        log.info(
            f"Bounce challenge split (ALL lanes): {_lv_pct:.1f}% liveness / "
            f"{_rest_pct:.1f}% arkose / {_rest_pct:.1f}% recaptcha "
            f"(--challenge-liveness-per-10k={args.challenge_liveness_per_10k})"
        )
    else:
        log.info(
            "Bounce challenge split (ALL lanes): 50% arkose / 50% recaptcha "
            "(liveness OFF, --challenge-liveness-per-10k=0)"
        )
    if args.dwell_dropout_check_url:
        log.info(
            f"client_dwell_dropout gate {'DRY-RUN' if args.dwell_dropout_dry_run else 'ENABLED'}: "
            f"url={args.dwell_dropout_check_url} fail_{'open' if args.dwell_dropout_fail_open else 'closed'} "
            f"cache_ttl={args.dwell_dropout_cache_ttl_days}d"
        )
    else:
        log.info("client_dwell_dropout gate DISABLED (--dwell-dropout-check-url unset)")
    if args.starter_pack_gate_mode != "off":
        log.info(
            f"Starter-pack-follows gate ENABLED mode={args.starter_pack_gate_mode} "
            f"(starterpacks/followPacksStore; FollowBot-only; "
            f"{'audit label only' if args.starter_pack_gate_mode == 'dry_run' else 'suspension -> arkose/captcha challenge'})"
        )
    else:
        log.info("Starter-pack-follows gate DISABLED (--starter-pack-gate-mode=off)")
    if paused_liveness_heads and args.paused_head_liveness_sample_per_10k > 0:
        _thr = {h: pol.paused_liveness_thresholds[h] for h in paused_liveness_heads}
        log.info(
            f"Paused-head->liveness ENABLED: heads={paused_liveness_heads} "
            f"sample={args.paused_head_liveness_sample_per_10k}/10000 thresholds={_thr}"
        )
    else:
        log.info("Paused-head->liveness DISABLED (no heads or sample=0)")
    if spam_bounce_heads and args.spam_bounce_sample_per_10k > 0:
        _sb_thr = {h: pol.spam_bounce_thresholds[h] for h in spam_bounce_heads}
        log.info(
            f"Spam-bounce (official-client) ENABLED: heads={spam_bounce_heads} "
            f"mode={args.spam_bounce_mode} "
            f"sample={args.spam_bounce_sample_per_10k}/10000 "
            f"dom_frac>={args.spam_bounce_dom_frac} thresholds={_sb_thr} "
            f"cooldown={args.spam_bounce_cooldown_days}d"
        )
    else:
        log.info("Spam-bounce (official-client) DISABLED (no heads or sample=0)")
    log.info(
        f"ReplySpamBot hard-suspend ENABLED: "
        f"ReplySpamBot>{pol.reply_spam_hard_suspend_tau} AND "
        f"dom_reply official-client frac>={args.spam_bounce_dom_frac} "
        f"→ enforcement_threshold_reached (no legit ceiling)"
    )
    return liveness_cfg, paused_liveness_heads, spam_bounce_heads


def _start_health_server(args, metrics, liveness_cfg):
    import json as _json

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") == "/config":
                body = _json.dumps(liveness_cfg, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                f"scored={metrics['scored']}\nbq={metrics['bq_inserted']}\n"
                f"proto={metrics.get('proto_published', 0)}\n"
                f"liveness_emitted={metrics['liveness_emitted']}\n"
                f"liveness_dedup_skipped={metrics['liveness_dedup_skipped']}\n"
                f"liveness_cooldown_days={args.liveness_cooldown_days}\n".encode()
            )

        def log_message(self, *_):
            pass

    srv = HTTPServer(("0.0.0.0", args.health_port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _user_head_scores(score_row, head_names) -> list[dict]:
    user_head_scores = []
    for h in range(min(len(score_row), len(head_names))):
        s = float(score_row[h])
        if not (np.isnan(s) or np.isinf(s)):
            user_head_scores.append(
                {
                    "head_index": h,
                    "head_name": head_names.get(h, f"head_{h}"),
                    "score": round(s, 6),
                }
            )
    return user_head_scores


def _user_action_histogram(action_histograms, i):
    if action_histograms and i < len(action_histograms):
        try:
            return [
                {"action_type": n, "cnt": int(c)}
                for n, c in action_histograms[i]
                if n != "PAD" and c > 0
            ]
        except Exception:
            pass
    return None


def _decide_hard_enforcement(
    uid_int,
    user_labels,
    user_action_hist,
    scores_by_name,
    legit,
    total_actions_gate,
    bsummary,
    pol,
    args,
    r,
    metrics,
) -> _Decision:
    dec = _Decision()

    _tweet_create_cnt = sum(
        h["cnt"] for h in (user_action_hist or []) if h["action_type"] == "SERVER_TWEET_CREATE"
    )
    dec.tweet_create_dominant = (
        total_actions_gate > 0 and _tweet_create_cnt / total_actions_gate >= 0.5
    )

    _unfollow_cnt = sum(
        h["cnt"] for h in (user_action_hist or []) if h["action_type"] == "SERVER_PROFILE_UNFOLLOW"
    )
    _follow_cnt = sum(
        h["cnt"] for h in (user_action_hist or []) if h["action_type"] == "SERVER_PROFILE_FOLLOW"
    )
    _unfollow_dominant = _unfollow_cnt > _follow_cnt

    if total_actions_gate >= pol.min_actions_for_enforcement and not dec.tweet_create_dominant:
        for _h, (_tau, _lam) in pol.thresholds.items():
            if scores_by_name.get(_h, 0) >= _tau and legit <= _lam:
                if _h == "FollowBot" and _unfollow_dominant:
                    continue
                dec.enforcement_met = True
                dec.enforcement_head = _h
                break

    if dec.enforcement_met and args.new_user_age_hours > 0:
        _age_hours = get_user_age_hours(uid_int, r, metrics)
        if _age_hours is None:
            dec.enforcement_met = False
            dec.age_gate_skip_reason = "age_unknown"
            metrics["age_gate_silenced_unknown"] += 1
        elif dec.enforcement_head == "FollowBot" and _age_hours < args.new_user_age_hours:
            dec.enforcement_met = False
            dec.age_gate_skip_reason = "new_user_followbot"
            metrics["age_gate_skipped_followbot"] += 1

    if (
        dec.enforcement_met
        and dec.enforcement_head == "FollowBot"
        and args.login_pack_window_min > 0
    ):
        _pack_age_min = get_login_pack_age_minutes(uid_int, r, metrics)
        if _pack_age_min is not None:
            _action = "skip" if _pack_age_min <= args.login_pack_window_min else "allow"
            log.info(
                f"login_pack_gate uid={uid_int} counter_set=True "
                f"age_min={_pack_age_min} "
                f"window_min={args.login_pack_window_min} "
                f"action={_action}"
            )
            if _pack_age_min <= args.login_pack_window_min:
                dec.enforcement_met = False
                dec.age_gate_skip_reason = "login_starter_pack"
                metrics["login_pack_gate_skipped_followbot"] += 1

    if (
        dec.enforcement_met
        and dec.enforcement_head == "FollowBot"
        and args.starter_pack_gate_mode != "off"
    ):
        _pack_count = get_followed_pack_count(uid_int, r, metrics)
        if _pack_count is not None and _pack_count > 0:
            log.info(
                f"starter_pack_gate uid={uid_int} "
                f"pack_count={_pack_count} "
                f"mode={args.starter_pack_gate_mode}"
            )
            if args.starter_pack_gate_mode == "dry_run":
                metrics["starter_pack_gate_would_skip"] += 1
                user_labels.append("enforcement_starter_pack_would_skip")
            else:
                dec.enforcement_met = False
                dec.age_gate_skip_reason = "starter_pack_follows"
                dec.starter_pack_challenge = True
                metrics["starter_pack_gate_challenged_followbot"] += 1

    if dec.enforcement_met and args.dwell_dropout_check_url:
        if client_dwell_dropout(r, uid_int, args, metrics):
            metrics["enforcement_skipped_dwell_dropout"] += 1
            if args.dwell_dropout_dry_run:
                user_labels.append("enforcement_dwell_dropout_would_skip")
            else:
                dec.enforcement_met = False
                dec.age_gate_skip_reason = "dwell_dropout"

    if (
        not dec.enforcement_met
        and dec.age_gate_skip_reason is None
        and total_actions_gate >= pol.min_actions_for_enforcement
        and scores_by_name.get("ReplySpamBot", 0) > pol.reply_spam_hard_suspend_tau
    ):
        _rs_bs = bsummary or {}
        _rs_app = _rs_bs.get("dom_reply_app_id")
        _rs_frac = _rs_bs.get("dom_reply_app_frac") or 0.0
        if _rs_app is None:
            metrics["reply_spam_hard_suspend_no_dom"] += 1
        elif _rs_app in pol.official_client_app_ids and _rs_frac >= args.spam_bounce_dom_frac:
            dec.enforcement_met = True
            dec.enforcement_head = "ReplySpamBot"
            metrics["reply_spam_hard_suspend"] += 1

    if dec.enforcement_met:
        user_labels.append("enforcement_threshold_reached")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
    elif dec.age_gate_skip_reason == "new_user_followbot":
        user_labels.append("enforcement_skipped_new_user_followbot")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
    elif dec.age_gate_skip_reason == "login_starter_pack":
        user_labels.append("enforcement_skipped_login_starter_pack")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
    elif dec.age_gate_skip_reason == "starter_pack_follows":
        user_labels.append("enforcement_skipped_starter_pack")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
        if dec.starter_pack_challenge and claim_action(
            r,
            "liveness",
            uid_int,
            int(args.liveness_cooldown_days * 86400),
            args.liveness_max_per_window,
            args.liveness_cooldown_fail_open,
        ):
            dec.sp_emitted = True
            user_labels.append(
                pick_challenge_label(uid_int, args.challenge_liveness_per_10k, metrics)
            )
    elif dec.age_gate_skip_reason == "age_unknown":
        user_labels.append("enforcement_silenced_age_unknown")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
    elif dec.age_gate_skip_reason == "dwell_dropout":
        user_labels.append("enforcement_skipped_dwell_dropout")
        if dec.enforcement_head and dec.enforcement_head not in user_labels:
            user_labels.append(dec.enforcement_head)
    elif total_actions_gate < pol.min_actions_for_enforcement:
        for _h, (_tau, _lam) in pol.thresholds.items():
            if scores_by_name.get(_h, 0) >= _tau and legit <= _lam:
                user_labels.append("enforcement_gated_low_actions")
                break

    return dec


def _cusp_lane(
    uid_int, dec, user_labels, scores_by_name, legit, total_actions_gate, pol, args, r, metrics
):
    if (
        not dec.enforcement_met
        and dec.age_gate_skip_reason is None
        and total_actions_gate >= pol.min_actions_for_enforcement
        and not dec.tweet_create_dominant
    ):
        for _h, (_tau, _lam) in pol.cusp_heads.items():
            _bot = scores_by_name.get(_h, 0)
            if _bot >= _tau - pol.cusp_delta and legit <= _lam + pol.cusp_delta:
                dec.cusp_met = True
                dec.cusp_head = _h
                break

    if not dec.cusp_met:
        return

    _in_liveness_canary = (
        args.cusp_liveness_sample_per_10k > 0
        and zlib.crc32(_challenge_salt() + b"cusp_liveness:" + str(uid_int).encode()) % 10000
        < args.cusp_liveness_sample_per_10k
    )
    _dropout = False
    if _in_liveness_canary and args.dwell_dropout_check_url:
        _dropout = client_dwell_dropout(r, uid_int, args, metrics)
        if _dropout:
            metrics["liveness_skipped_dwell_dropout"] += 1

    if _dropout and not args.dwell_dropout_dry_run:
        user_labels.append("dwell_dropout_exempted")
    else:
        if _dropout:
            user_labels.append("dwell_dropout_would_skip")
        _emit_liveness = _in_liveness_canary and claim_action(
            r,
            "liveness",
            uid_int,
            int(args.liveness_cooldown_days * 86400),
            args.liveness_max_per_window,
            args.liveness_cooldown_fail_open,
        )
        if _emit_liveness:
            metrics["liveness_emitted"] += 1
            user_labels.append(
                pick_challenge_label(uid_int, args.challenge_liveness_per_10k, metrics)
            )
        else:
            if _in_liveness_canary:
                metrics["liveness_dedup_skipped"] += 1
            _cusp_bucket = zlib.crc32(str(uid_int).encode()) % len(_CUSP_BUCKET_LABELS)
            user_labels.append(_CUSP_BUCKET_LABELS[_cusp_bucket])
        if dec.cusp_head and dec.cusp_head not in user_labels:
            user_labels.append(dec.cusp_head)


def _paused_head_lane(
    uid_int,
    dec,
    user_labels,
    scores_by_name,
    legit,
    total_actions_gate,
    paused_liveness_heads,
    pol,
    args,
    r,
    metrics,
):
    if (
        paused_liveness_heads
        and args.paused_head_liveness_sample_per_10k > 0
        and not dec.enforcement_met
        and not dec.cusp_met
        and dec.age_gate_skip_reason is None
        and total_actions_gate >= pol.min_actions_for_enforcement
        and not dec.tweet_create_dominant
    ):
        for _h in paused_liveness_heads:
            _tau, _lam = pol.paused_liveness_thresholds[_h]
            if scores_by_name.get(_h, 0) >= _tau and legit <= _lam:
                dec.lv_head = _h
                break
    if dec.lv_head is None:
        return

    _in_lv_sample = (
        zlib.crc32(_challenge_salt() + b"paused_head_liveness:" + str(uid_int).encode()) % 10000
        < args.paused_head_liveness_sample_per_10k
    )
    if _in_lv_sample:
        _lv_dropout = bool(args.dwell_dropout_check_url) and client_dwell_dropout(
            r, uid_int, args, metrics
        )
        if _lv_dropout and not args.dwell_dropout_dry_run:
            metrics["liveness_skipped_dwell_dropout"] += 1
            user_labels.append("dwell_dropout_exempted")
        else:
            if _lv_dropout:
                metrics["liveness_skipped_dwell_dropout"] += 1
                user_labels.append("dwell_dropout_would_skip")
            if claim_action(
                r,
                "liveness",
                uid_int,
                int(args.liveness_cooldown_days * 86400),
                args.liveness_max_per_window,
                args.liveness_cooldown_fail_open,
            ):
                metrics["paused_head_liveness_emitted"] += 1
                dec.lv_head_emitted = True
                user_labels.append(
                    pick_challenge_label(uid_int, args.challenge_liveness_per_10k, metrics)
                )
                if dec.lv_head not in user_labels:
                    user_labels.append(dec.lv_head)


def _spam_bounce_lane(
    uid_int,
    dec,
    user_labels,
    scores_by_name,
    legit,
    total_actions_gate,
    bsummary,
    spam_bounce_heads,
    pol,
    args,
    r,
    metrics,
):
    if (
        spam_bounce_heads
        and args.spam_bounce_sample_per_10k > 0
        and not dec.enforcement_met
        and not dec.cusp_met
        and not dec.lv_head_emitted
        and dec.age_gate_skip_reason is None
        and total_actions_gate >= pol.min_actions_for_enforcement
    ):
        for _h in spam_bounce_heads:
            _tau, _lam = pol.spam_bounce_thresholds[_h]
            if scores_by_name.get(_h, 0) >= _tau and legit <= _lam:
                dec.sb_head = _h
                break
    if dec.sb_head is None:
        return

    _sb_bs = bsummary or {}
    _sb_key = pol.spam_bounce_action_key[dec.sb_head]
    _sb_app = _sb_bs.get(f"dom_{_sb_key}_app_id")
    _sb_frac = _sb_bs.get(f"dom_{_sb_key}_app_frac") or 0.0
    if _sb_app is None:
        metrics["spam_bounce_no_dom_data"] += 1
    elif (
        _sb_app in pol.official_client_app_ids
        and _sb_frac >= args.spam_bounce_dom_frac
        and zlib.crc32(_challenge_salt() + b"spam_bounce:" + str(uid_int).encode()) % 10000
        < args.spam_bounce_sample_per_10k
    ):
        if args.spam_bounce_mode == "dry_run":
            metrics["spam_bounce_would_challenge"] += 1
            user_labels.append("spam_bounce_would_challenge")
            if dec.sb_head not in user_labels:
                user_labels.append(dec.sb_head)
        elif claim_action(
            r, "spam_bounce", uid_int, int(args.spam_bounce_cooldown_days * 86400), 1, False
        ):
            metrics["spam_bounce_challenged"] += 1
            dec.sb_emitted = True
            user_labels.append(
                pick_challenge_label(uid_int, args.challenge_liveness_per_10k, metrics)
            )
            if dec.sb_head not in user_labels:
                user_labels.append(dec.sb_head)
        else:
            metrics["spam_bounce_dedup_skipped"] += 1


def _build_bq_row(
    uid_int, score_id, now_ts, user_head_scores, user_action_hist, user_labels, bsummary, args
):
    total_actions = None
    if user_action_hist:
        total_actions = sum(h["cnt"] for h in user_action_hist)

    _svr_count = 0
    _dwell = 0
    _render = 0
    if user_action_hist:
        for h in user_action_hist:
            atype = h["action_type"]
            cnt = h["cnt"]
            if atype.startswith("SERVER_"):
                _svr_count += cnt
            if "DWELLED" in atype and "NOT" not in atype:
                _dwell += cnt
            if "NOT_DWELLED" in atype or "RENDER" in atype:
                _render += cnt

    row = {
        "score_id": score_id,
        "user_id": uid_int,
        "score_ts": now_ts,
        "head_scores": user_head_scores,
        "action_histogram": user_action_hist,
        "total_actions": total_actions,
        "enforcement_note": build_enforcement_note(user_head_scores, user_action_hist),
        "labels": user_labels,
        "model_version": args.model_version,
        "pipeline_version": "gpu_scorer_kafka_v3",
        "sequence_length": 0,
        "sequence_format": "arrow_ipc_zstd",
        "server_pct": round(_svr_count / total_actions * 100, 1) if total_actions else None,
        "dwell_count": _dwell if _dwell > 0 else None,
        "render_count": _render if _render > 0 else None,
        "decoded_timeline": bsummary.get("decoded_timeline") if bsummary else None,
        "platform_distribution": bsummary.get("platform_distribution") if bsummary else None,
        "top_action_transitions": bsummary.get("top_action_transitions") if bsummary else None,
        "engagement_without_impression": bsummary.get("engagement_without_impression")
        if bsummary
        else None,
        "engagement_without_impression_pct": bsummary.get("engagement_without_impression_pct")
        if bsummary
        else None,
        "median_gap_sec": bsummary.get("median_gap_sec") if bsummary else None,
        "time_span_sec": bsummary.get("time_span_sec") if bsummary else None,
    }
    return row, total_actions


def _publish_score_result(
    score_producer,
    pb,
    uid_raw,
    row,
    user_head_scores,
    user_labels,
    total_actions,
    dec,
    scores_by_name,
    legit,
    pol,
    args,
    metrics,
):
    try:
        sr = pb.ScoreResult()
        sr.user_id = int(uid_raw)
        sr.model_version = args.model_version
        sr.score_ts_ms = int(time.time() * 1000)
        sr.score_status = "SCORED"
        sr.scored_sequence_fingerprint = row["score_id"]
        for hs in user_head_scores:
            sr.raw_head_scores.append(float(hs["score"]))
        sr.summary.labels.extend(user_labels)
        if total_actions:
            sr.summary.total_actions = total_actions
        bot_heads_for_summary = [
            (hs["head_name"], hs["score"])
            for hs in user_head_scores
            if hs["head_name"] != "LegitimateUser" and hs["score"] > 0.5
        ]
        if bot_heads_for_summary:
            bot_heads_for_summary.sort(key=lambda x: x[1], reverse=True)
            sr.summary.reason_code = bot_heads_for_summary[0][0]
        if dec.enforcement_met and dec.enforcement_head:
            _ph = sr.summary.fired_heads.add()
            _ph.name = dec.enforcement_head
            _ph.score = float(scores_by_name.get(dec.enforcement_head, 0.0))
            if dec.enforcement_head == "ReplySpamBot" and "ReplySpamBot" not in pol.thresholds:
                _ph.threshold = float(pol.reply_spam_hard_suspend_tau)
            else:
                _ph.threshold = float(pol.thresholds[dec.enforcement_head][0])
                for _h, (_tau, _lam) in pol.thresholds.items():
                    if _h == dec.enforcement_head:
                        continue
                    if scores_by_name.get(_h, 0) >= _tau and legit <= _lam:
                        _cf = sr.summary.fired_heads.add()
                        _cf.name = _h
                        _cf.score = float(scores_by_name.get(_h, 0.0))
                        _cf.threshold = float(_tau)
        elif dec.cusp_met and dec.cusp_head:
            _cph = sr.summary.fired_heads.add()
            _cph.name = dec.cusp_head
            _cph.score = float(scores_by_name.get(dec.cusp_head, 0.0))
            _cph.threshold = float(pol.cusp_heads[dec.cusp_head][0] - pol.cusp_delta)
        elif dec.lv_head_emitted:
            _lph = sr.summary.fired_heads.add()
            _lph.name = dec.lv_head
            _lph.score = float(scores_by_name.get(dec.lv_head, 0.0))
            _lph.threshold = float(pol.paused_liveness_thresholds[dec.lv_head][0])
        elif dec.sb_emitted:
            _sbh = sr.summary.fired_heads.add()
            _sbh.name = dec.sb_head
            _sbh.score = float(scores_by_name.get(dec.sb_head, 0.0))
            _sbh.threshold = float(pol.spam_bounce_thresholds[dec.sb_head][0])
        elif dec.sp_emitted:
            _sph = sr.summary.fired_heads.add()
            _sph.name = dec.enforcement_head
            _sph.score = float(scores_by_name.get(dec.enforcement_head, 0.0))
            _sph.threshold = float(pol.thresholds[dec.enforcement_head][0])
        if row["action_histogram"]:
            for entry in row["action_histogram"]:
                ac = sr.action_counts_non_zero.add()
                ac.action_name = entry["action_type"]
                ac.count = entry["cnt"]
        enforcement_note = row.get("enforcement_note")
        if enforcement_note:
            sr.enforcement_note = enforcement_note
        score_producer.produce(
            args.output_topic,
            key=str(uid_raw).encode(),
            value=sr.SerializeToString(),
        )
        metrics["proto_published"] += 1
    except Exception as e:
        if metrics["proto_published"] == 0:
            log.warning(f"Proto publish failed: {e}")


def _apply_scoring_cooldowns(
    r,
    uids,
    action_histograms,
    cooldown_threshold,
    cooldown_low_sec,
    cooldown_high_sec,
    metrics,
    traced_in_batch,
):
    if r is None:
        return
    B = len(uids)
    try:
        pipe = r.pipeline(transaction=False)
        n_low = 0
        n_high = 0
        cooldown_decisions: list[tuple[int, int]] = []
        for i, uid in enumerate(uids):
            n_actions = 0
            if action_histograms and i < len(action_histograms):
                try:
                    n_actions = sum(int(c) for n, c in action_histograms[i] if n != "PAD" and c > 0)
                except Exception:
                    n_actions = 0
            if n_actions >= cooldown_threshold:
                pipe.set(f"scored:{uid}", "1", ex=cooldown_high_sec)
                n_high += 1
                cooldown_decisions.append((int(uid), cooldown_high_sec))
            else:
                pipe.set(f"scored:{uid}", "1", ex=cooldown_low_sec)
                n_low += 1
                cooldown_decisions.append((int(uid), cooldown_low_sec))
        pipe.execute()
        metrics["redis_sets"] += B
        metrics["redis_sets_low"] += n_low
        metrics["redis_sets_high"] += n_high

        if traced_in_batch:
            for uid, ttl in cooldown_decisions:
                if uid in traced_in_batch:
                    _trace.emit(
                        stage="sink_cooldown_set",
                        uid=uid,
                        decision="ok",
                        detail={
                            "ttl_sec": ttl,
                            "kind": "high" if ttl == cooldown_high_sec else "low",
                        },
                    )
    except Exception as e:
        if traced_in_batch:
            for uid in traced_in_batch:
                _trace.emit(
                    stage="sink_cooldown_set",
                    uid=uid,
                    decision="err",
                    detail={"error": str(e)[:200]},
                )


def _flush_bq(bq_client, table_ref, bq_buffer, metrics):
    traced_in_buffer: set[int] = set()
    if _trace.enabled and bq_buffer:
        row_uids = [int(r.get("user_id", 0)) for r in bq_buffer]
        traced_in_buffer = _trace.is_active_many(row_uids)
    n_rows_in_buffer = len(bq_buffer)
    try:
        errors = bq_client.insert_rows_json(table_ref, bq_buffer)
        if errors:
            metrics["bq_errors"] += len(errors)
            if traced_in_buffer:
                for uid in traced_in_buffer:
                    _trace.emit(
                        stage="sink_bq_insert",
                        uid=uid,
                        decision="err",
                        detail={"error": str(errors[:1])[:200], "buffer_size": n_rows_in_buffer},
                    )
        else:
            metrics["bq_inserted"] += len(bq_buffer)
            if traced_in_buffer:
                for uid in traced_in_buffer:
                    _trace.emit(
                        stage="sink_bq_insert",
                        uid=uid,
                        decision="ok",
                        detail={"buffer_size": n_rows_in_buffer},
                    )
    except Exception as e:
        log.warning(f"BQ insert failed: {e}")
        metrics["bq_errors"] += len(bq_buffer)
        if traced_in_buffer:
            for uid in traced_in_buffer:
                _trace.emit(
                    stage="sink_bq_insert",
                    uid=uid,
                    decision="err",
                    detail={"error": str(e)[:200], "buffer_size": n_rows_in_buffer},
                )


def _log_periodic_stats(metrics, now):
    elapsed = now - metrics["_start"]
    log.info(
        f"[Sink] {elapsed:.0f}s | scored: {metrics['scored']:,} ({metrics['scored'] / elapsed:.0f}/sec) | "
        f"flagged: {metrics['flagged']:,} | bq: {metrics['bq_inserted']:,} | "
        f"redis: {metrics['redis_sets']:,} (low={metrics['redis_sets_low']:,} "
        f"high={metrics['redis_sets_high']:,}) | proto: {metrics['proto_published']:,} | "
        f"liveness: emit={metrics['liveness_emitted']:,} "
        f"ph_emit={metrics['paused_head_liveness_emitted']:,} "
        f"dedup_skip={metrics['liveness_dedup_skipped']:,} "
        f"dropout_skip={metrics['liveness_skipped_dwell_dropout']:,} "
        f"| challenge_pick: live={metrics['challenge_pick_liveness']:,} "
        f"ark={metrics['challenge_pick_arkose']:,} "
        f"cap={metrics['challenge_pick_captcha']:,} "
        f"(chk={metrics['dwell_dropout_checks']:,} hit={metrics['dwell_dropout_cache_hits']:,} "
        f"err={metrics['dwell_dropout_errors']:,}) | "
        f"enf_dropout_skip={metrics['enforcement_skipped_dwell_dropout']:,} | "
        f"spam_bounce: would={metrics['spam_bounce_would_challenge']:,} "
        f"chal={metrics['spam_bounce_challenged']:,} "
        f"dedup={metrics['spam_bounce_dedup_skipped']:,} "
        f"nodata={metrics['spam_bounce_no_dom_data']:,} | "
        f"age_gate: fb_skip={metrics['age_gate_skipped_followbot']:,} "
        f"silenced={metrics['age_gate_silenced_unknown']:,} "
        f"(strato_ok={metrics['age_strato_ok']:,} "
        f"strato_fail={metrics['age_strato_fail']:,} "
        f"cache_local={metrics['age_cache_local']:,} "
        f"cache_redis={metrics['age_cache_redis']:,}) | "
        f"login_pack_gate: fb_skip={metrics['login_pack_gate_skipped_followbot']:,} "
        f"(strato_set={metrics['login_pack_strato_set']:,} "
        f"strato_unset={metrics['login_pack_strato_unset']:,} "
        f"strato_fail={metrics['login_pack_strato_fail']:,} "
        f"cache_local={metrics['login_pack_cache_local']:,} "
        f"cache_redis={metrics['login_pack_cache_redis']:,}) | "
        f"starter_pack_gate: challenged={metrics['starter_pack_gate_challenged_followbot']:,} "
        f"would_skip={metrics['starter_pack_gate_would_skip']:,} "
        f"(strato_set={metrics['starter_pack_strato_set']:,} "
        f"strato_unset={metrics['starter_pack_strato_unset']:,} "
        f"strato_fail={metrics['starter_pack_strato_fail']:,} "
        f"cache_local={metrics['starter_pack_cache_local']:,} "
        f"cache_redis={metrics['starter_pack_cache_redis']:,})"
    )


def main():
    args = _parse_args()

    head_names = _resolve_head_names(args.model_version)
    _flag_indices, _flag_thresholds = _build_flag_config(head_names)
    log.info(
        f"Model version: {args.model_version}, {len(head_names)} heads: {list(head_names.values())}"
    )

    metrics = _init_metrics()

    pol = _load_policy(args.policy_file)
    liveness_cfg, paused_liveness_heads, spam_bounce_heads = _build_policy_config(args, pol)

    _start_health_server(args, metrics, liveness_cfg)

    from confluent_kafka import Consumer, Producer, KafkaError
    import zstandard as zstd

    try:
        from proto_gen import abuse_inference_pb2 as pb

        _has_proto = True
        log.info(f"Proto loaded — will publish ScoreResult to {args.output_topic}")
    except ImportError:
        pb = None
        _has_proto = False
        log.warning("abuse_inference_pb2 not found — Kafka proto publishing disabled")

    consumer = Consumer(
        {
            "bootstrap.servers": args.kafka_bootstrap,
            "group.id": args.kafka_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "SCRAM-SHA-512",
            "sasl.username": args.kafka_username,
            "sasl.password": args.kafka_password,
            **kafka_ssl_config(),
            "fetch.wait.max.ms": 100,
        }
    )
    consumer.subscribe([args.kafka_topic])
    log.info(f"Subscribed to {args.kafka_topic}")

    score_producer = Producer(
        {
            "bootstrap.servers": args.kafka_bootstrap,
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "SCRAM-SHA-512",
            "sasl.username": args.kafka_username,
            "sasl.password": args.kafka_password,
            **kafka_ssl_config(),
            "message.max.bytes": 1 * 1024 * 1024,
            "linger.ms": 5,
            "batch.num.messages": 500,
            "acks": "1",
        }
    )
    metrics["proto_published"] = 0

    import redis as redis_lib

    r = redis_lib.Redis(host=args.redis_host, port=args.redis_port)
    cooldown_low_sec = int(args.cooldown_low_hours * 3600)
    cooldown_high_sec = int(args.cooldown_high_hours * 3600)
    cooldown_threshold = args.min_actions_for_long_cooldown
    try:
        r.ping()
        log.info(
            f"Redis connected (cooldown: {args.cooldown_low_hours}h if <{cooldown_threshold} actions, "
            f"{args.cooldown_high_hours}h if >={cooldown_threshold} actions)"
        )
    except Exception as e:
        log.warning(f"Redis unavailable: {e}")
        r = None

    from google.cloud import bigquery

    bq_client = bigquery.Client(project=args.bq_project)
    table_ref = bq_client.dataset(args.bq_dataset).table(args.bq_table)
    log.info(f"BQ writer ready → {args.bq_table}")

    dctx = zstd.ZstdDecompressor()
    bq_buffer = []
    last_bq_flush = time.time()
    BQ_BATCH = 500
    BQ_INTERVAL = 10.0

    while True:
        msg = consumer.poll(0.5)

        if msg is not None and not msg.error():
            payload = safe_pickle_loads(dctx.decompress(msg.value()))
            uids = payload["uids"]
            scores = payload["scores"]
            score_ids = payload.get("score_ids")
            action_histograms = payload.get("action_histograms")
            behavioral_summaries = payload.get("behavioral_summaries")
            B = len(uids)

            metrics["consumed"] += 1
            metrics["scored"] += B

            uids_int = [int(u) for u in uids]
            traced_in_batch = _trace.is_active_many(uids_int) if _trace.enabled else set()
            if traced_in_batch:
                for uid in traced_in_batch:
                    _trace.emit(
                        stage="sink_consume_score",
                        uid=uid,
                        decision="ok",
                        detail={"model_version": args.model_version, "batch_size": B},
                    )

            n_heads = min(scores.shape[1], len(head_names))
            safe_indices = [idx for idx in _flag_indices if idx < n_heads]
            if safe_indices:
                flag_scores = scores[:, safe_indices]
                safe_thresholds = _flag_thresholds[: len(safe_indices)]
                flagged_mask = np.any(flag_scores > safe_thresholds, axis=1)
            else:
                flagged_mask = np.zeros(B, dtype=bool)
            metrics["flagged"] += int(flagged_mask.sum())
            for i in np.where(flagged_mask)[0]:
                parts = [f"{head_names.get(h, f'h{h}')}={scores[i, h]:.4f}" for h in range(n_heads)]
                log.info(f"[Score] {uids[i]}|{'|'.join(parts)}")

            now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            for i in range(B):
                uid_int = int(uids[i])
                user_head_scores = _user_head_scores(scores[i], head_names)
                user_action_hist = _user_action_histogram(action_histograms, i)

                user_labels = [
                    hs["head_name"]
                    for hs in user_head_scores
                    if hs["score"] > 0.5 and "legit" not in hs["head_name"].lower()
                ]

                total_actions_gate = (
                    sum(h["cnt"] for h in user_action_hist) if user_action_hist else 0
                )
                scores_by_name = {hs["head_name"]: hs["score"] for hs in user_head_scores}
                legit = scores_by_name.get("LegitimateUser", 1.0)
                bsummary = None
                if behavioral_summaries and i < len(behavioral_summaries):
                    bsummary = behavioral_summaries[i]

                dec = _decide_hard_enforcement(
                    uid_int,
                    user_labels,
                    user_action_hist,
                    scores_by_name,
                    legit,
                    total_actions_gate,
                    bsummary,
                    pol,
                    args,
                    r,
                    metrics,
                )
                _cusp_lane(
                    uid_int,
                    dec,
                    user_labels,
                    scores_by_name,
                    legit,
                    total_actions_gate,
                    pol,
                    args,
                    r,
                    metrics,
                )
                _paused_head_lane(
                    uid_int,
                    dec,
                    user_labels,
                    scores_by_name,
                    legit,
                    total_actions_gate,
                    paused_liveness_heads,
                    pol,
                    args,
                    r,
                    metrics,
                )
                _spam_bounce_lane(
                    uid_int,
                    dec,
                    user_labels,
                    scores_by_name,
                    legit,
                    total_actions_gate,
                    bsummary,
                    spam_bounce_heads,
                    pol,
                    args,
                    r,
                    metrics,
                )

                score_id = score_ids[i] if score_ids else str(_uuid.uuid4())
                row, total_actions = _build_bq_row(
                    uid_int,
                    score_id,
                    now_ts,
                    user_head_scores,
                    user_action_hist,
                    user_labels,
                    bsummary,
                    args,
                )
                bq_buffer.append(row)

                if _has_proto:
                    _publish_score_result(
                        score_producer,
                        pb,
                        uids[i],
                        row,
                        user_head_scores,
                        user_labels,
                        total_actions,
                        dec,
                        scores_by_name,
                        legit,
                        pol,
                        args,
                        metrics,
                    )

            if _has_proto:
                score_producer.poll(0)

            _apply_scoring_cooldowns(
                r,
                uids,
                action_histograms,
                cooldown_threshold,
                cooldown_low_sec,
                cooldown_high_sec,
                metrics,
                traced_in_batch,
            )

        elif msg is not None and msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                log.warning(f"Kafka error: {msg.error()}")

        now = time.time()
        if len(bq_buffer) >= BQ_BATCH or (bq_buffer and now - last_bq_flush > BQ_INTERVAL):
            _flush_bq(bq_client, table_ref, bq_buffer, metrics)
            bq_buffer = []
            last_bq_flush = now

        if metrics["consumed"] > 0 and metrics["consumed"] % 50 == 0:
            _log_periodic_stats(metrics, now)


if __name__ == "__main__":
    main()
