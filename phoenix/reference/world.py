#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import dataclasses

import numpy as np

TOPIC_NAMES = (
    "space",
    "cooking",
    "ml",
    "gaming",
    "music",
    "travel",
    "sports",
    "finance",
    "art",
    "science",
    "fashion",
    "food",
    "history",
    "film",
    "nature",
    "tech",
)

MM_DIM = 1024

TWITTER_EPOCH_MS = 1_288_834_974_657

DEFAULT_NOW_MS = 1_784_592_000_000

POST_MAX_AGE_MS = 36 * 3600 * 1000
POST_MIN_AGE_MS = 1 * 3600 * 1000

SURFACE_HOME = 1
SURFACE_GALLERY = 10

ACTION_FAV = 1
ACTION_REPLY = 4
ACTION_QUOTE = 5
ACTION_REPOST = 6
ACTION_VIDEO_VIEW = 13
ACTION_CLICK = 29
ACTION_BOOKMARK = 36
ACTION_SHARE = 37
ACTION_DWELLED = 11
ACTION_NOT_DWELLED = 12

ACTION_TABLE = (
    (ACTION_FAV, "fav", 0.14),
    (ACTION_CLICK, "click", 0.10),
    (ACTION_VIDEO_VIEW, "video_view", 0.08),
    (ACTION_REPOST, "repost", 0.045),
    (ACTION_REPLY, "reply", 0.035),
    (ACTION_BOOKMARK, "bookmark", 0.02),
    (ACTION_QUOTE, "quote", 0.015),
    (ACTION_SHARE, "share", 0.012),
)
ACTION_ORDINALS = np.array([row[0] for row in ACTION_TABLE], dtype=np.int64)
ACTION_BASE_RATES = np.array([row[2] for row in ACTION_TABLE], dtype=np.float64)

ACTION_NAMES = {row[0]: row[1] for row in ACTION_TABLE}
ACTION_NAMES[ACTION_DWELLED] = "dwelled"
ACTION_NAMES[ACTION_NOT_DWELLED] = "not_dwelled"

LABEL_WIDTH = 64

DWELL_MARK_SECONDS = 10.0
DWELL_CAP_SECONDS = 300.0

CLIENT_APP_IOS = 129_032
CLIENT_APP_ANDROID = 258_901

TZ_UTC_OFFSET_HOURS = (
    0.0,
    -5.0,
    -6.0,
    -8.0,
    -7.0,
    -3.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    3.0,
    3.0,
    9.0,
    5.5,
    8.0,
    9.0,
    7.0,
    7.0,
    4.0,
    3.0,
    8.0,
    8.0,
    11.0,
    -10.0,
    -7.0,
    -5.0,
    -6.0,
    -3.0,
    1.0,
    0.0,
)

_ADJECTIVES = (
    "quiet",
    "bold",
    "tiny",
    "vivid",
    "odd",
    "sleek",
    "warm",
    "sharp",
    "gentle",
    "brisk",
    "curious",
    "steady",
    "bright",
    "rough",
    "calm",
    "wild",
)
_NOUNS_BY_TOPIC = (
    ("orbit", "nebula", "lander", "comet"),
    ("stew", "knife", "sourdough", "broth"),
    ("gradient", "encoder", "dataset", "benchmark"),
    ("speedrun", "quest", "pixel", "boss"),
    ("chord", "vinyl", "encore", "riff"),
    ("hostel", "passport", "trail", "harbor"),
    ("derby", "overtime", "transfer", "midfield"),
    ("dividend", "ledger", "hedge", "rally"),
    ("canvas", "gallery", "sketch", "mural"),
    ("enzyme", "quasar", "fossil", "reactor"),
    ("runway", "denim", "capsule", "stitch"),
    ("noodle", "market", "harvest", "roast"),
    ("archive", "empire", "treaty", "relic"),
    ("premiere", "script", "director", "reel"),
    ("canopy", "riverbank", "meadow", "aurora"),
    ("compiler", "gadget", "silicon", "protocol"),
)


@dataclasses.dataclass(frozen=True)
class Topic:
    topic_id: int
    name: str
    anchor: np.ndarray

    def __str__(self) -> str:
        return f"topic[{self.topic_id}] {self.name!r}"


@dataclasses.dataclass(frozen=True)
class Author:
    author_id: int
    handle: str
    primary_topic: int
    secondary_topic: int
    quality: float
    style: np.ndarray

    def __str__(self) -> str:
        return (
            f"@{self.handle} (id={self.author_id}) "
            f"topics={TOPIC_NAMES[self.primary_topic]}/{TOPIC_NAMES[self.secondary_topic]} "
            f"quality={self.quality:.2f}"
        )


@dataclasses.dataclass(frozen=True)
class User:
    user_id: int
    handle: str
    topic_ids: tuple[int, ...]
    topic_weights: tuple[float, ...]
    activity: float
    timezone: int
    country: int
    language: int
    gender: int
    age_bracket: int
    age_years: int
    dma: int
    state: int
    lon: float
    lat: float
    installed_apps: tuple[int, ...]
    ip_int: int
    client_app_id: int

    def prefs_str(self) -> str:
        return " ".join(
            f"{TOPIC_NAMES[t]}:{w:.2f}" for t, w in zip(self.topic_ids, self.topic_weights)
        )

    def __str__(self) -> str:
        return (
            f"user_{self.user_id:05d} prefs=[{self.prefs_str()}] "
            f"tz={self.timezone} activity={self.activity:.2f} apps={len(self.installed_apps)}"
        )


@dataclasses.dataclass(frozen=True)
class Post:
    post_id: int
    author_id: int
    topic_id: int
    created_ms: int
    quality: float
    popularity: float
    product_surface: int
    text: str

    def __str__(self) -> str:
        return f"post {self.post_id} (q={self.quality:.2f}): {self.text}"


class World:
    def __init__(self, seed: int, now_ms: int, arrays: dict[str, np.ndarray | list]):
        self.seed = seed
        self.now_ms = now_ms
        for name, value in arrays.items():
            setattr(self, name, value)
        self.num_topics = len(self.topic_anchor)
        self.num_authors = len(self.author_quality)
        self.num_users = len(self.user_activity)
        self.num_posts = len(self.post_id)
        self._build_topic_pools()

    def _build_topic_pools(self) -> None:
        author_q = self.author_quality[self.author_row_of_post]
        engage_w = self.post_popularity * (0.1 + 0.9 * author_q * self.post_quality)
        serve_w = self.post_popularity
        self._topic_posts: list[np.ndarray] = []
        self._engage_cum: list[np.ndarray] = []
        self._serve_cum: list[np.ndarray] = []
        for t in range(self.num_topics):
            idx = np.flatnonzero(self.post_topic == t)
            if len(idx) == 0:
                idx = np.arange(self.num_posts)
            self._topic_posts.append(idx)
            self._engage_cum.append(np.cumsum(engage_w[idx]))
            self._serve_cum.append(np.cumsum(serve_w[idx]))

    def topic(self, t: int) -> Topic:
        return Topic(topic_id=t, name=TOPIC_NAMES[t], anchor=self.topic_anchor[t])

    def author(self, i: int) -> Author:
        return Author(
            author_id=int(self.author_id[i]),
            handle=f"author_{i:05d}",
            primary_topic=int(self.author_primary_topic[i]),
            secondary_topic=int(self.author_secondary_topic[i]),
            quality=float(self.author_quality[i]),
            style=self.author_style[i],
        )

    def user(self, i: int) -> User:
        k = int(self.user_num_topics[i])
        return User(
            user_id=int(self.user_id[i]),
            handle=f"user_{int(self.user_id[i]):05d}",
            topic_ids=tuple(int(t) for t in self.user_topic_ids[i, :k]),
            topic_weights=tuple(float(w) for w in self.user_topic_w[i, :k]),
            activity=float(self.user_activity[i]),
            timezone=int(self.user_timezone[i]),
            country=int(self.user_country[i]),
            language=int(self.user_language[i]),
            gender=int(self.user_gender[i]),
            age_bracket=int(self.user_age_bracket[i]),
            age_years=int(self.user_age_years[i]),
            dma=int(self.user_dma[i]),
            state=int(self.user_state[i]),
            lon=float(self.user_lon[i]),
            lat=float(self.user_lat[i]),
            installed_apps=tuple(np.flatnonzero(self.user_apps[i]).tolist()),
            ip_int=int(self.user_ip[i]),
            client_app_id=int(self.user_client_app[i]),
        )

    def post(self, i: int) -> Post:
        return Post(
            post_id=int(self.post_id[i]),
            author_id=int(self.author_id[self.author_row_of_post[i]]),
            topic_id=int(self.post_topic[i]),
            created_ms=int(self.post_created_ms[i]),
            quality=float(self.post_quality[i]),
            popularity=float(self.post_popularity[i]),
            product_surface=int(self.post_surface[i]),
            text=self.post_text[i],
        )

    def affinity(self, user_row: int, post_rows: np.ndarray) -> np.ndarray:
        prefs = self.user_pref_matrix[user_row]
        author_q = self.author_quality[self.author_row_of_post[post_rows]]
        return prefs[self.post_topic[post_rows]] * author_q * self.post_quality[post_rows]

    @staticmethod
    def gate(affinity: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-14.0 * (np.asarray(affinity, dtype=np.float64) - 0.10)))

    def action_probs(self, affinity: np.ndarray) -> np.ndarray:
        return ACTION_BASE_RATES[None, :] * self.gate(affinity)[:, None]

    def sample_engagement(
        self,
        rng: np.random.Generator,
        user_row: int,
        post_rows: np.ndarray,
        ensure_engaged: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(post_rows)
        aff = self.affinity(user_row, post_rows)
        probs = self.action_probs(aff)
        if ensure_engaged:
            probs = np.minimum(probs * 2.5, 0.97)
        fired = rng.random(probs.shape) < probs
        if ensure_engaged:
            silent = ~fired.any(axis=1)
            fired[silent, np.argmax(probs[silent], axis=1)] = True

        labels = np.zeros((n, LABEL_WIDTH), dtype=bool)
        labels[:, ACTION_ORDINALS] = fired
        engaged = fired.any(axis=1)

        loc = np.log(6.0) + 2.0 * self.gate(aff)
        dwell = np.exp(rng.normal(loc, 0.8, size=n))
        dwell = np.clip(dwell, 0.5, DWELL_CAP_SECONDS) * engaged
        labels[:, ACTION_DWELLED] = engaged & (dwell >= DWELL_MARK_SECONDS)
        labels[:, ACTION_NOT_DWELLED] = ~engaged
        return labels, dwell.astype(np.float32)

    def _topic_prefix(self, t: int, max_created_ms: int | None) -> np.ndarray:
        pool = self._topic_posts[t]
        if max_created_ms is None:
            return pool
        k = int(np.searchsorted(self.post_created_ms[pool], max_created_ms, side="right"))
        return pool[: max(k, 1)]

    def _sample_topic_pool(
        self, rng, cum: list[np.ndarray], topics: np.ndarray, max_created_ms: int | None = None
    ) -> np.ndarray:
        out = np.empty(len(topics), dtype=np.int64)
        for t in np.unique(topics):
            sel = np.flatnonzero(topics == t)
            k = len(self._topic_prefix(t, max_created_ms))
            c = cum[t][:k]
            j = np.searchsorted(c, rng.random(len(sel)) * c[-1], side="right")
            out[sel] = self._topic_posts[t][np.minimum(j, k - 1)]
        return out

    def _user_topic_draw(self, rng: np.random.Generator, user_row: int, n: int) -> np.ndarray:
        k = int(self.user_num_topics[user_row])
        return self.user_topic_ids[user_row, :k][
            rng.choice(k, size=n, p=self.user_topic_w[user_row, :k])
        ]

    def sample_engaged_posts(
        self, rng: np.random.Generator, user_row: int, n: int, max_created_ms: int | None = None
    ) -> np.ndarray:
        topics = self._user_topic_draw(rng, user_row, 3 * n + 8)
        drawn = self._sample_topic_pool(rng, self._engage_cum, topics, max_created_ms)
        distinct = drawn[np.sort(np.unique(drawn, return_index=True)[1])][:n]
        if len(distinct) < n:
            k = int(self.user_num_topics[user_row])
            pool = np.concatenate(
                [self._topic_prefix(t, max_created_ms) for t in self.user_topic_ids[user_row, :k]]
            )
            extra = np.setdiff1d(pool, distinct, assume_unique=False)
            take = min(n - len(distinct), len(extra))
            distinct = np.concatenate([distinct, rng.permutation(extra)[:take]])
        return distinct

    def sample_candidate_posts(self, rng: np.random.Generator, user_row: int, n: int) -> np.ndarray:
        m = 2 * n + 8
        aligned = rng.random(m) < 0.70
        topics = np.where(
            aligned,
            self._user_topic_draw(rng, user_row, m),
            rng.integers(0, self.num_topics, size=m),
        )
        drawn = self._sample_topic_pool(rng, self._serve_cum, topics)
        distinct = drawn[np.sort(np.unique(drawn, return_index=True)[1])][:n]
        while len(distinct) < n:
            more = rng.integers(0, self.num_posts, size=n)
            distinct = np.unique(np.concatenate([distinct, more]))[:n]
        return distinct

    def engagement_times(
        self, rng: np.random.Generator, user_row: int, post_rows: np.ndarray, cap_ms: int
    ) -> np.ndarray:
        n = len(post_rows)
        created = self.post_created_ms[post_rows]
        assert (created < cap_ms).all(), "engagement_times needs posts created before cap_ms"
        delays = np.exp(rng.normal(np.log(2 * 3600e3), 1.0, size=(n, 3)))
        cand = created[:, None] + delays.astype(np.int64)
        offset_ms = int(TZ_UTC_OFFSET_HOURS[int(self.user_timezone[user_row])] * 3600e3)
        local_hour = ((cand + offset_ms) // 3600_000) % 24
        weight = self.user_hour_curve[user_row][local_hour]
        weight = np.where(cand <= cap_ms, weight, -1.0)
        times = cand[np.arange(n), np.argmax(weight, axis=1)]
        over = times > cap_ms
        if over.any():
            span = (cap_ms - created[over]).astype(np.float64)
            times[over] = created[over] + (rng.random(over.sum()) * span).astype(np.int64)
        return np.maximum(times, created + 1)

    def mm_vector(self, post_rows: np.ndarray | int) -> np.ndarray:
        rows = np.atleast_1d(np.asarray(post_rows, dtype=np.int64))
        anchors = self.topic_anchor[self.post_topic[rows]]
        styles = self.author_style[self.author_row_of_post[rows]]
        jitter = np.empty((len(rows), MM_DIM), dtype=np.float32)
        for k, r in enumerate(rows):
            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(9, int(self.post_id[r])))
            jitter[k] = np.random.default_rng(ss).standard_normal(MM_DIM, dtype=np.float32)
        jitter /= np.maximum(np.linalg.norm(jitter, axis=1, keepdims=True), 1e-6)
        vec = anchors + 0.35 * styles + 0.15 * jitter
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
        out = vec.astype(np.float32)
        return out[0] if np.isscalar(post_rows) or np.ndim(post_rows) == 0 else out

    def preview(self, n: int = 3) -> str:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(99,)))
        lines = [
            f"World(seed={self.seed}, users={self.num_users}, authors={self.num_authors}, "
            f"posts={self.num_posts}, topics={self.num_topics})",
            "topics: " + ", ".join(TOPIC_NAMES[: self.num_topics]),
        ]
        for i in range(min(n, self.num_users)):
            lines.append(f"  {self.user(i)}")
        for i in range(min(n, self.num_posts)):
            lines.append(f"  {self.post(i)}")
        u = int(rng.integers(0, self.num_users))
        posts = self.sample_engaged_posts(rng, u, 3)
        labels, dwell = self.sample_engagement(rng, u, posts, ensure_engaged=True)
        lines.append(f"  sample session for {self.user(u).handle}:")
        for j, p in enumerate(posts):
            acts = ",".join(ACTION_NAMES[o] for o in np.flatnonzero(labels[j]) if o in ACTION_NAMES)
            lines.append(f"    {self.post(int(p)).text}  actions=[{acts}] dwell={dwell[j]:.1f}s")
        return "\n".join(lines)


def _rng(seed: int, key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))


def generate_world(
    seed: int = 20260721,
    num_users: int = 10_000,
    num_authors: int = 2_000,
    num_posts: int = 50_000,
    num_topics: int = 16,
    now_ms: int | None = None,
) -> World:
    assert 2 <= num_topics <= len(TOPIC_NAMES), f"num_topics must be in [2, {len(TOPIC_NAMES)}]"
    now = DEFAULT_NOW_MS if now_ms is None else int(now_ms)
    arrays: dict[str, np.ndarray | list] = {}

    trng = _rng(seed, 1)
    anchor = trng.standard_normal((num_topics, MM_DIM)).astype(np.float32)
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-6)
    arrays["topic_anchor"] = anchor

    arng = _rng(seed, 2)
    arrays["author_id"] = np.arange(num_users + 1, num_users + num_authors + 1, dtype=np.int64)
    arrays["author_primary_topic"] = arng.integers(0, num_topics, size=num_authors)
    arrays["author_secondary_topic"] = (
        arrays["author_primary_topic"] + arng.integers(1, num_topics, size=num_authors)
    ) % num_topics
    arrays["author_quality"] = arng.beta(2.0, 2.0, size=num_authors)
    style = arng.standard_normal((num_authors, MM_DIM)).astype(np.float32)
    style /= np.maximum(np.linalg.norm(style, axis=1, keepdims=True), 1e-6)
    arrays["author_style"] = style
    arrays["author_is_nsfw"] = _rng(seed, 8).random(num_authors) < 0.06

    urng = _rng(seed, 3)
    arrays["user_id"] = np.arange(1, num_users + 1, dtype=np.int64)
    k = urng.integers(2, 5, size=num_users)
    topic_ids = np.full((num_users, 4), -1, dtype=np.int64)
    topic_w = np.zeros((num_users, 4), dtype=np.float64)
    pref_matrix = np.zeros((num_users, num_topics), dtype=np.float64)
    perm_scores = urng.random((num_users, num_topics))
    order = np.argsort(perm_scores, axis=1)
    gam = urng.gamma(1.5, size=(num_users, 4))
    for i in range(num_users):
        ki = int(k[i])
        chosen = order[i, :ki]
        w = gam[i, :ki] / gam[i, :ki].sum()
        desc = np.argsort(-w)
        topic_ids[i, :ki] = chosen[desc]
        topic_w[i, :ki] = w[desc]
        pref_matrix[i, chosen] = w
    arrays["user_num_topics"] = k
    arrays["user_topic_ids"] = topic_ids
    arrays["user_topic_w"] = topic_w
    arrays["user_pref_matrix"] = pref_matrix
    arrays["user_activity"] = np.exp(urng.normal(0.0, 0.6, size=num_users))
    arrays["user_timezone"] = urng.integers(0, 32, size=num_users)
    for name, vocab in (
        ("user_country", 30),
        ("user_language", 20),
        ("user_gender", 3),
        ("user_age_bracket", 7),
        ("user_dma", 100),
        ("user_state", 50),
    ):
        arrays[name] = urng.integers(0, vocab, size=num_users).astype(np.uint16)
    arrays["user_age_years"] = urng.integers(13, 90, size=num_users).astype(np.uint8)
    arrays["user_lon"] = urng.uniform(-180.0, 180.0, size=num_users).astype(np.float32)
    lat = np.clip(urng.normal(20.0, 25.0, size=num_users), -60.0, 70.0)
    arrays["user_lat"] = lat.astype(np.float32)
    apps = np.zeros((num_users, 64), dtype=bool)
    napps = urng.integers(2, 9, size=num_users)
    app_scores = urng.random((num_users, 64)).argsort(axis=1)
    for i in range(num_users):
        apps[i, app_scores[i, : napps[i]]] = True
    arrays["user_apps"] = apps
    arrays["user_ip"] = urng.integers(1 << 24, 1 << 32, size=num_users, dtype=np.int64)
    ios = urng.random(num_users) < 0.6
    arrays["user_client_app"] = np.where(ios, CLIENT_APP_IOS, CLIENT_APP_ANDROID).astype(np.int64)
    peak = 19.0 + urng.integers(-3, 4, size=num_users)
    hours = np.arange(24, dtype=np.float64)
    curve = 0.25 + np.exp(1.4 * np.cos(2.0 * np.pi * (hours[None, :] - peak[:, None]) / 24.0))
    arrays["user_hour_curve"] = curve / curve.sum(axis=1, keepdims=True)

    prng = _rng(seed, 4)
    author_row = prng.integers(0, num_authors, size=num_posts)
    arrays["author_row_of_post"] = author_row
    use_secondary = prng.random(num_posts) < 0.20
    arrays["post_topic"] = np.where(
        use_secondary,
        arrays["author_secondary_topic"][author_row],
        arrays["author_primary_topic"][author_row],
    )
    created = prng.integers(now - POST_MAX_AGE_MS, now - POST_MIN_AGE_MS, size=num_posts)
    order = np.argsort(created)
    created = created[order]
    arrays["post_created_ms"] = created
    arrays["post_id"] = ((created - TWITTER_EPOCH_MS) << 22) | (
        np.arange(num_posts, dtype=np.int64) % (1 << 22)
    )
    arrays["post_quality"] = prng.beta(2.0, 2.0, size=num_posts)
    rank = prng.permutation(num_posts).astype(np.float64)
    arrays["post_popularity"] = (rank + 1.0) ** -1.05
    home = prng.random(num_posts) < 0.85
    arrays["post_surface"] = np.where(home, SURFACE_HOME, SURFACE_GALLERY).astype(np.int32)

    crng = _rng(seed, 6)
    quality_gate = 0.1 + 0.9 * arrays["author_quality"][author_row] * arrays["post_quality"]
    views = arrays["post_popularity"] * 3e6 * np.exp(crng.normal(0.0, 0.4, size=num_posts))
    views = np.maximum(views, 8.0)
    arrays["post_view_count"] = views.astype(np.int64)
    for name, base_rate in (
        ("post_fav_count", 0.03),
        ("post_repost_count", 0.008),
        ("post_reply_count", 0.005),
        ("post_quote_count", 0.002),
    ):
        c = views * base_rate * quality_gate * np.exp(crng.normal(0.0, 0.25, size=num_posts))
        arrays[name] = np.minimum(c, views).astype(np.int64)
    arrays["post_fav_count"] = np.maximum(arrays["post_fav_count"], 1)

    xrng = _rng(seed, 5)
    adj = xrng.integers(0, len(_ADJECTIVES), size=num_posts)
    nn = xrng.integers(0, 4, size=num_posts)
    topics = arrays["post_topic"]
    arrays["post_text"] = [
        f"[{TOPIC_NAMES[topics[i]]}] {_ADJECTIVES[adj[i]]} "
        f"{_NOUNS_BY_TOPIC[topics[i] % len(_NOUNS_BY_TOPIC)][nn[i]]} #{i} "
        f"by @author_{author_row[i]:05d}"
        for i in range(num_posts)
    ]

    return World(seed=seed, now_ms=now, arrays=arrays)


def _self_check() -> None:
    w = generate_world(seed=7, num_users=300, num_authors=60, num_posts=2_000, num_topics=8)
    w2 = generate_world(seed=7, num_users=300, num_authors=60, num_posts=2_000, num_topics=8)

    assert np.array_equal(w.post_id, w2.post_id), "post ids not deterministic"
    assert np.array_equal(w.user_pref_matrix, w2.user_pref_matrix), "users not deterministic"
    assert w.post_text == w2.post_text, "post text not deterministic"

    assert (w.user_id >= 1).all() and (w.author_id >= 1).all() and (w.post_id >= 1).all()
    assert len(np.intersect1d(w.user_id, w.author_id)) == 0, "user/author id ranges overlap"
    assert len(np.unique(w.post_id)) == w.num_posts, "post ids collide"
    decoded_ms = (w.post_id >> 22) + TWITTER_EPOCH_MS
    assert (decoded_ms >= w.now_ms - POST_MAX_AGE_MS).all()
    assert (decoded_ms <= w.now_ms - POST_MIN_AGE_MS).all()
    assert np.allclose(w.user_pref_matrix.sum(axis=1), 1.0), "prefs must sum to 1"

    a = np.linspace(0.0, 1.0, 101)
    g = World.gate(a)
    assert (np.diff(g) > 0).all(), "gate must be strictly increasing"
    probs = w.action_probs(a)
    assert (np.diff(probs, axis=0) > 0).all(), "P(action) must increase with affinity"
    assert probs.max() <= ACTION_BASE_RATES.max() + 1e-9
    assert ACTION_NAMES[int(ACTION_ORDINALS[0])] == "fav" and LABEL_WIDTH == 64
    assert int(ACTION_ORDINALS.max()) < LABEL_WIDTH, "an ordinal escapes the label vector"

    rng = _rng(123, 77)
    posts = w.sample_engaged_posts(rng, 0, 200)
    assert len(posts) == 200 and len(np.unique(posts)) == 200, "history posts must be distinct"
    labels, dwell = w.sample_engagement(rng, 0, posts, ensure_engaged=True)
    assert labels.shape == (200, LABEL_WIDTH) and labels.any(axis=1).all(), "history must engage"
    assert (dwell[labels[:, ACTION_DWELLED]] >= DWELL_MARK_SECONDS).all()
    assert dwell.max() <= DWELL_CAP_SECONDS

    cands = w.sample_candidate_posts(rng, 0, 64)
    assert len(np.unique(cands)) == 64
    clabels, cdwell = w.sample_engagement(rng, 0, cands, ensure_engaged=False)
    assert (cdwell[~clabels[:, ACTION_ORDINALS].any(axis=1)] == 0).all(), "dwell w/o engagement"
    assert clabels[:, ACTION_NOT_DWELLED].any(), "expect some impression negatives"

    cap = w.now_ms - 1_800_000
    times = w.engagement_times(rng, 0, posts, cap_ms=cap)
    assert (times > w.post_created_ms[posts]).all() and (times <= cap).all()
    assert len(TZ_UTC_OFFSET_HOURS) == 32 and w.user_hour_curve.shape[1] == 24
    assert SURFACE_HOME == 1 and SURFACE_GALLERY == 10
    both = set(w.user_client_app)
    assert CLIENT_APP_IOS in both and CLIENT_APP_ANDROID in both

    assert (w.post_fav_count >= 1).all() and (w.post_view_count >= w.post_fav_count).all()
    for name in ("post_reply_count", "post_repost_count", "post_quote_count"):
        counts = getattr(w, name)
        assert (counts >= 0).all() and (counts <= w.post_view_count).all(), name
    by_pop = np.argsort(-w.post_popularity)
    head, tail = by_pop[:200], by_pop[-200:]
    assert w.post_view_count[head].mean() > 50 * w.post_view_count[tail].mean(), (
        "views must follow popularity"
    )
    nsfw_rate = float(w.author_is_nsfw.mean())
    assert 0.0 < nsfw_rate < 0.2, f"NSFW author rate {nsfw_rate} outside sane band"

    sample = rng.choice(w.num_posts, size=120, replace=False)
    vecs = w.mm_vector(sample)
    assert vecs.shape == (120, MM_DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
    assert np.array_equal(vecs, w.mm_vector(sample)), "mm vectors not deterministic"
    sims = vecs @ vecs.T
    same = w.post_topic[sample][:, None] == w.post_topic[sample][None, :]
    off_diag = ~np.eye(120, dtype=bool)
    same_mean = sims[same & off_diag].mean()
    diff_mean = sims[~same & off_diag].mean()
    assert same_mean > diff_mean + 0.2, f"MM not topic-clustered ({same_mean=} {diff_mean=})"

    text = "\n".join([str(w.topic(0)), str(w.author(0)), str(w.user(0)), str(w.post(0))])
    assert "topic[0]" in text and "@author_00000" in text
    overview = w.preview(2)
    assert "sample session" in overview

    print(
        "world self-check PASS  "
        f"users={w.num_users} authors={w.num_authors} posts={w.num_posts} "
        f"topics={w.num_topics} mm_topic_cos={same_mean:.3f} vs other={diff_mean:.3f}\n" + overview
    )


if __name__ == "__main__":
    _self_check()
