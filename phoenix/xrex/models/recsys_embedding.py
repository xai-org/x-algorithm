# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import numba
import numpy as np
from jax.sharding import NamedSharding
from xai_configlib import Config, configclass

from xrex.models.model_utils import Parameter


class EmbTable(NamedTuple):
    x: Any


logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@dataclass
class RecsysEmbeddingsParameter:
    user_embeddings: Parameter | EmbTable | None
    history_post_embeddings: Parameter | EmbTable | None
    candidate_post_embeddings: Parameter | EmbTable | None
    history_author_embeddings: Parameter | EmbTable
    candidate_author_embeddings: Parameter | EmbTable
    user_ip_embeddings: Parameter | EmbTable | None = None


@dataclass
class RecsysEmbeddings:
    user_embeddings: jax.Array | None
    history_post_embeddings: jax.Array | None
    candidate_post_embeddings: jax.Array | None
    history_author_embeddings: jax.Array
    candidate_author_embeddings: jax.Array
    user_ip_embeddings: jax.Array | None = None


def get_recsys_embed_param_to_jax_array(
    recsys_embeddings_parameter: RecsysEmbeddingsParameter,
) -> RecsysEmbeddings:
    return RecsysEmbeddings(
        user_embeddings=(
            recsys_embeddings_parameter.user_embeddings.x
            if recsys_embeddings_parameter.user_embeddings is not None
            else None
        ),
        history_post_embeddings=(
            recsys_embeddings_parameter.history_post_embeddings.x
            if recsys_embeddings_parameter.history_post_embeddings is not None
            else None
        ),
        candidate_post_embeddings=(
            recsys_embeddings_parameter.candidate_post_embeddings.x
            if recsys_embeddings_parameter.candidate_post_embeddings is not None
            else None
        ),
        history_author_embeddings=recsys_embeddings_parameter.history_author_embeddings.x,
        candidate_author_embeddings=recsys_embeddings_parameter.candidate_author_embeddings.x,
        user_ip_embeddings=(
            recsys_embeddings_parameter.user_ip_embeddings.x
            if recsys_embeddings_parameter.user_ip_embeddings is not None
            else None
        ),
    )


def get_recsys_embed_jax_array_to_param(
    recsys_embeddings: RecsysEmbeddings, sharding: NamedSharding
) -> RecsysEmbeddingsParameter:
    return RecsysEmbeddingsParameter(
        user_embeddings=(
            Parameter(x=recsys_embeddings.user_embeddings, pspec=sharding.spec)
            if recsys_embeddings.user_embeddings is not None
            else None
        ),
        history_post_embeddings=(
            Parameter(x=recsys_embeddings.history_post_embeddings, pspec=sharding.spec)
            if recsys_embeddings.history_post_embeddings is not None
            else None
        ),
        candidate_post_embeddings=(
            Parameter(x=recsys_embeddings.candidate_post_embeddings, pspec=sharding.spec)
            if recsys_embeddings.candidate_post_embeddings is not None
            else None
        ),
        history_author_embeddings=Parameter(
            x=recsys_embeddings.history_author_embeddings, pspec=sharding.spec
        ),
        candidate_author_embeddings=Parameter(
            x=recsys_embeddings.candidate_author_embeddings, pspec=sharding.spec
        ),
        user_ip_embeddings=(
            Parameter(x=recsys_embeddings.user_ip_embeddings, pspec=sharding.spec)
            if recsys_embeddings.user_ip_embeddings is not None
            else None
        ),
    )


from jax.tree_util import register_pytree_node


def _recsys_embeddings_parameter_flatten(obj):
    return (
        (
            obj.user_embeddings,
            obj.history_post_embeddings,
            obj.candidate_post_embeddings,
            obj.history_author_embeddings,
            obj.candidate_author_embeddings,
            obj.user_ip_embeddings,
        ),
        None,
    )


def _recsys_embeddings_parameter_unflatten(aux_data, children):
    return RecsysEmbeddingsParameter(*children)


def _recsys_embeddings_flatten(obj):
    return (
        (
            obj.user_embeddings,
            obj.history_post_embeddings,
            obj.candidate_post_embeddings,
            obj.history_author_embeddings,
            obj.candidate_author_embeddings,
            obj.user_ip_embeddings,
        ),
        None,
    )


def _recsys_embeddings_unflatten(aux_data, children):
    return RecsysEmbeddings(*children)


register_pytree_node(
    RecsysEmbeddingsParameter,
    _recsys_embeddings_parameter_flatten,
    _recsys_embeddings_parameter_unflatten,
)

register_pytree_node(RecsysEmbeddings, _recsys_embeddings_flatten, _recsys_embeddings_unflatten)


@numba.njit
def _hash_ids_batch(
    arr: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    modulus: int,
    num_buckets: int,
) -> np.ndarray:
    ids = arr.ravel()
    n = len(ids)
    m = len(scales)
    out = np.empty((n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            raw = (ids[i] * scales[j] + biases[j]) % modulus
            out[i, j] = 0 if ids[i] == 0 else ((raw % (num_buckets - 1)) + 1)

    return out.astype(np.int32)


@configclass
class HashKeys(Config):
    user_id_table_size: int = 100_000
    user_hash_scales: list[int] = field(default_factory=lambda: [196742702, 1852108266])
    user_biases: list[int] = field(default_factory=lambda: [1935840681, 167407236])
    user_modulus: int = 2_859_568_897

    item_id_table_size: int = 100_000
    item_hash_vocab_size: int = 0
    item_hash_scales: list[int] = field(default_factory=lambda: [2161410491, 1754358832])
    item_biases: list[int] = field(default_factory=lambda: [1935840681, 167407236])
    item_modulus: int = 2_361_375_383

    author_id_table_size: int = 10_000
    author_hash_scales: list[int] = field(default_factory=lambda: [371965780, 328930218])
    author_biases: list[int] = field(default_factory=lambda: [139686260, 37755056])
    author_modulus: int = 631_860_353

    ip_id_table_size: int = 0
    ip_hash_scales: list[int] = field(default_factory=lambda: [529_482_163, 1_327_604_891])
    ip_biases: list[int] = field(default_factory=lambda: [742_961_053, 318_205_477])
    ip_modulus: int = 1_073_741_789

    def __post_init__(self):
        self._usr_scales: np.ndarray = np.array(self.user_hash_scales, dtype=np.int64)
        self._usr_bias: np.ndarray = np.array(self.user_biases, dtype=np.int64)
        self._itm_scales: np.ndarray = np.array(self.item_hash_scales, dtype=np.int64)
        self._itm_bias: np.ndarray = np.array(self.item_biases, dtype=np.int64)
        self._aut_scales: np.ndarray = np.array(self.author_hash_scales, dtype=np.int64)
        self._aut_bias: np.ndarray = np.array(self.author_biases, dtype=np.int64)
        self._ip_scales: np.ndarray = np.array(self.ip_hash_scales, dtype=np.int64)
        self._ip_bias: np.ndarray = np.array(self.ip_biases, dtype=np.int64)

    @property
    def effective_item_hash_buckets(self) -> int:
        return self.item_hash_vocab_size or self.item_id_table_size

    @property
    def num_user_hashes(self) -> int:
        return len(self.user_hash_scales)

    @property
    def num_item_hashes(self) -> int:
        return len(self.item_hash_scales)

    @property
    def num_author_hashes(self) -> int:
        return len(self.author_hash_scales)

    @property
    def num_ip_hashes(self) -> int:
        return len(self.ip_hash_scales)


@configclass
class HashTable(Config):
    output_vocab_size: int
    hash_keys: HashKeys = HashKeys()

    @property
    def offset_user(self):
        return 1 + 64

    @property
    def offset_item(self):
        return self.offset_user + self.hash_keys.user_id_table_size

    @property
    def offset_author(self):
        return self.offset_item + self.hash_keys.item_id_table_size

    @property
    def offset_ip(self):
        return self.offset_author + self.hash_keys.author_id_table_size

    @property
    def num_user_hashes(self):
        return self.hash_keys.num_user_hashes

    @property
    def num_item_hashes(self):
        return self.hash_keys.num_item_hashes

    @property
    def num_author_hashes(self):
        return self.hash_keys.num_author_hashes

    @property
    def num_ip_hashes(self):
        return self.hash_keys.num_ip_hashes

    @property
    def user_id_table_size(self):
        return self.hash_keys.user_id_table_size

    @property
    def item_id_table_size(self):
        return self.hash_keys.item_id_table_size

    @property
    def author_id_table_size(self):
        return self.hash_keys.author_id_table_size

    @property
    def ip_id_table_size(self):
        return self.hash_keys.ip_id_table_size

    def get_user_hash(self, user_ids: np.ndarray) -> np.ndarray:
        input_shape = user_ids.shape
        if self.hash_keys.user_id_table_size == 0:
            result = np.zeros(input_shape + (self.hash_keys.num_user_hashes,), dtype=np.int32)
            result[user_ids != 0, 0] = 1
            return result
        user_hashed = _hash_ids_batch(
            user_ids,
            self.hash_keys._usr_scales,
            self.hash_keys._usr_bias,
            self.hash_keys.user_modulus,
            self.hash_keys.user_id_table_size,
        )
        user_hashed = np.where(user_hashed == 0, 0, user_hashed + self.offset_user)
        return user_hashed.reshape(input_shape + (-1,))

    def get_author_hash(self, author_ids: np.ndarray) -> np.ndarray:
        input_shape = author_ids.shape
        author_hashed = _hash_ids_batch(
            author_ids,
            self.hash_keys._aut_scales,
            self.hash_keys._aut_bias,
            self.hash_keys.author_modulus,
            self.hash_keys.author_id_table_size,
        )
        author_hashed = np.where(author_hashed == 0, 0, author_hashed + self.offset_author)
        return author_hashed.reshape(input_shape + (-1,))

    def get_item_hash(self, item_ids: np.ndarray) -> np.ndarray:
        input_shape = item_ids.shape
        item_hashed = _hash_ids_batch(
            item_ids,
            self.hash_keys._itm_scales,
            self.hash_keys._itm_bias,
            self.hash_keys.item_modulus,
            self.hash_keys.effective_item_hash_buckets,
        )
        item_hashed = np.where(item_hashed == 0, 0, item_hashed + self.offset_item)
        return item_hashed.reshape(input_shape + (-1,))

    def get_ip_hash(self, ip_ids: np.ndarray) -> np.ndarray:
        if self.hash_keys.ip_id_table_size == 0:
            return np.zeros(ip_ids.shape + (self.hash_keys.num_ip_hashes,), dtype=np.int32)
        input_shape = ip_ids.shape
        ip_hashed = _hash_ids_batch(
            ip_ids,
            self.hash_keys._ip_scales,
            self.hash_keys._ip_bias,
            self.hash_keys.ip_modulus,
            self.hash_keys.ip_id_table_size,
        )
        ip_hashed = np.where(ip_hashed == 0, 0, ip_hashed + self.offset_ip)
        return ip_hashed.reshape(input_shape + (-1,))
