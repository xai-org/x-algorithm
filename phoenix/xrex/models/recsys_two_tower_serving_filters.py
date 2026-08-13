# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import os

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P

from xrex.cuda.top_k_by_key import top_k_by_key
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.topic_categories import NUM_TOPIC_INT32S


def forward_with_filters(
    model,
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    post_embeddings: jax.Array,
    dataset_types: jax.Array | None,
    top_k: int = 0,
    target_dataset_types: tuple[int, ...] = (1,),
    eligible_mask: jax.Array | None = None,
    topic_bitmaps: jax.Array | None = None,
    topic_user_bitmasks: jax.Array | None = None,
    eval_bs_per_device: int = 0,
    dataset_ranges: tuple[tuple[int, int], ...] | None = None,
    use_async_topk: bool = False,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    saxis = "expert"

    user_representation, _, _ = model(
        batch=batch,
        recsys_embeddings=recsys_embeddings,
        is_training=False,
    )

    if post_embeddings is not None:
        mesh = model.sharding_context.mesh

    skip_dataset_mask = os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1"

    def local_top_k(scores: jax.Array, k: int):
        N = scores.shape[1]
        k_actual = min(k, N)

        return top_k_by_key(scores, k_actual, heuristic_pivot_ratio=0.1, use_async=use_async_topk)

    use_topic_filter = topic_bitmaps is not None and topic_user_bitmasks is not None

    @shard_map(
        mesh=mesh,
        in_specs=(P(saxis), P()),
        out_specs=P(),
        check_vma=False,
    )
    def compute_top_k(post_embeddings: jax.Array, user_embedding: jax.Array) -> jax.Array:
        if eval_bs_per_device > 0:
            users_per_expert = user_embedding.shape[0] // mesh.shape[saxis]
            max_eval_users = eval_bs_per_device * mesh.shape[saxis]
            if eval_bs_per_device < users_per_expert:
                user_embedding = user_embedding.reshape(
                    mesh.shape[saxis], users_per_expert, user_embedding.shape[-1]
                )[:, :eval_bs_per_device].reshape(max_eval_users, user_embedding.shape[-1])
        scores = jnp.matmul(user_embedding.astype(jnp.bfloat16), post_embeddings.T)
        scores = jax.lax.all_to_all(
            scores, axis_name=saxis, split_axis=0, concat_axis=1, tiled=True
        )
        return scores

    mask_in_spec = P(saxis) if eligible_mask is not None else P()

    topic_bitmaps_in_spec = P(saxis, None) if use_topic_filter else P()
    topic_user_bitmasks_in_spec = P(None, None) if use_topic_filter else P()

    @shard_map(
        mesh=mesh,
        in_specs=(P(), P(), mask_in_spec, topic_bitmaps_in_spec, topic_user_bitmasks_in_spec),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def mask_and_top_k(
        all_scores: jax.Array,
        type_mask: jax.Array,
        user_eligible_mask: jax.Array,
        topic_bitmaps_shard: jax.Array,
        topic_user_bitmasks_full: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        combined = type_mask & user_eligible_mask
        masked_scores = jnp.where(combined, all_scores, jnp.finfo(jnp.bfloat16).min)

        if use_topic_filter:
            topic_bitmaps_full = jax.lax.all_gather(
                topic_bitmaps_shard, axis_name=saxis, axis=0, tiled=True
            )
            B_local = masked_scores.shape[0]
            shard_idx = jax.lax.axis_index(saxis)
            start_idx = shard_idx * B_local
            user_bitmasks_local = jax.lax.dynamic_slice(
                topic_user_bitmasks_full, (start_idx, 0), (B_local, NUM_TOPIC_INT32S)
            )
            topic_mask = jnp.any(
                (topic_bitmaps_full[None, :, :] & user_bitmasks_local[:, None, :]) != 0,
                axis=-1,
            )
            no_filter_mask = jnp.all(user_bitmasks_local == 0, axis=-1)[:, None]
            topic_mask = jnp.where(no_filter_mask, True, topic_mask)
            masked_scores = jnp.where(topic_mask, masked_scores, jnp.finfo(jnp.bfloat16).min)

        sorted_scores, sorted_indices = local_top_k(masked_scores, top_k)
        top_k_scores = jax.lax.all_gather(sorted_scores, axis_name=saxis, axis=0, tiled=True)
        top_k_indices = jax.lax.all_gather(sorted_indices, axis_name=saxis, axis=0, tiled=True)
        return top_k_scores, top_k_indices

    all_scores = compute_top_k(post_embeddings, user_representation)

    use_slice_path = (
        dataset_ranges is not None
        and eligible_mask is None
        and not use_topic_filter
        and not skip_dataset_mask
    )
    if use_slice_path:
        assert len(dataset_ranges) == len(target_dataset_types), (
            f"dataset_ranges ({len(dataset_ranges)}) must align with "
            f"target_dataset_types ({len(target_dataset_types)})"
        )
        results = []
        for start, end in dataset_ranges:

            @shard_map(mesh=mesh, in_specs=(P(),), out_specs=(P(), P()), check_vma=False)
            def slice_and_top_k(all_scores, _start=start, _end=end):
                sliced = jax.lax.slice_in_dim(all_scores, _start, _end, axis=1)
                sorted_scores, sorted_indices = local_top_k(sliced, top_k)
                sorted_indices = sorted_indices + _start
                top_k_scores = jax.lax.all_gather(
                    sorted_scores, axis_name=saxis, axis=0, tiled=True
                )
                top_k_indices = jax.lax.all_gather(
                    sorted_indices, axis_name=saxis, axis=0, tiled=True
                )
                return top_k_scores, top_k_indices

            top_k_scores, top_k_indices = slice_and_top_k(all_scores)
            results.append((top_k_indices, top_k_scores.astype(jnp.float32)))
        return tuple(results)

    B = all_scores.shape[0]
    M = post_embeddings.shape[0]
    if eligible_mask is None:
        user_eligible_mask = jnp.ones((B, M), dtype=jnp.bool_)
    else:
        user_eligible_mask = eligible_mask

    results = []
    for ds_type in target_dataset_types:
        if dataset_types is not None and not skip_dataset_mask:
            type_mask = dataset_types.squeeze() == ds_type
        else:
            type_mask = jnp.ones(post_embeddings.shape[0], dtype=jnp.bool_)

        _topic_bitmaps = topic_bitmaps if use_topic_filter else jnp.zeros((), dtype=jnp.int32)
        _topic_user_bitmasks = (
            topic_user_bitmasks if use_topic_filter else jnp.zeros((), dtype=jnp.int32)
        )
        top_k_scores, top_k_indices = mask_and_top_k(
            all_scores, type_mask, user_eligible_mask, _topic_bitmaps, _topic_user_bitmasks
        )
        results.append((top_k_indices, top_k_scores.astype(jnp.float32)))

    return tuple(results)
