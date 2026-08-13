# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import jax

Mesh = jax.sharding.Mesh
Shape = tuple[int, ...]
ShardingSpec = str | tuple[str] | None
ShardingRule = Callable[[str], ShardingSpec]

rank_logger = logging.getLogger("rank")


PyTree = Any


def axis_group_size(axis_names: PyTree, mesh: Mesh) -> int:
    flattened_axis_names, _ = jax.tree.flatten(axis_names)
    if not flattened_axis_names:
        return 0
    return math.prod([mesh.shape[n] for n in flattened_axis_names])


class NamedShape:
    names: tuple[str, ...] | None

    def __init__(self, shape: Shape | None = None, names: tuple[str, ...] | None = None):
        if names is None and len(shape) > 0:
            raise ValueError(
                f"Unable to create named shape with unnamed dimensions (shape: {shape})"
            )
        if names is not None and len(names) != len(shape):
            raise ValueError(
                f"Number of names must match number of dimensions (shape: {shape}, names: {names})"
            )
        self.shape = shape
        self.names = names

    def __repr__(self) -> str:
        shape_str = ", ".join(
            f"{name or 'unnamed'}={size}" for name, size in zip(self.names, self.shape)
        )
        return f"NamedShape({shape_str})"


class ShardingContext:
    def __init__(
        self,
        name: str,
        mesh: jax.sharding.Mesh,
        sharding_rules: dict[str, ShardingRule] | None = None,
    ):
        self.name = name
        self.mesh = mesh
        self.sharding_rules = {} if sharding_rules is None else sharding_rules

    def register_sharding_rule(self, namespace: str) -> Callable[[ShardingRule], None]:
        if namespace in self.sharding_rules:
            raise ValueError(
                f"{namespace} is already exist in ShardingContext {self.name},{self.sharding_rules}"
            )

        def _register_sharding_rule(rule: ShardingRule) -> "ShardingContext":
            self.sharding_rules[namespace] = rule

        return _register_sharding_rule

    def logical_axis_to_physical(
        self, logical_axis_name: str, namespace: str = "default", fallback_default: bool = True
    ):
        try:
            sharding_rule = self.sharding_rules.get(namespace)
            assert sharding_rule is not None, f"Unknown sharding namespace: {namespace}"
            physical_axes = sharding_rule(logical_axis_name)
        except Exception as e:
            if fallback_default:
                rank_logger.debug(f"Falling back to default namespace from {namespace}.")
                return self.logical_axis_to_physical(
                    logical_axis_name, namespace="default", fallback_default=False
                )
            else:
                raise e

        rank_logger.debug(
            f"  Logical axis {logical_axis_name} is mapped to {physical_axes} (resolved by {namespace})"
        )
        return physical_axes

    def sharding_specs(
        self, named_shape: NamedShape, namespace: str = "default", fallback_default: bool = True
    ) -> jax.sharding.PartitionSpec:
        specs = tuple(
            self.logical_axis_to_physical(
                name, namespace=namespace, fallback_default=fallback_default
            )
            for name in named_shape.names
        )
        rank_logger.debug(
            f"Input named shape {named_shape} is sharded as {specs} (resolved by {namespace})"
        )
        return jax.sharding.PartitionSpec(*specs)

    def logical_group_size(self, logical_axis_names: PyTree | str, namespace: str = "default"):
        if isinstance(logical_axis_names, str):
            physical_axes = self.logical_axis_to_physical(logical_axis_names, namespace)
        else:
            flattened_logical_axis_names, _ = jax.tree.flatten(logical_axis_names)
            physical_axes = [
                self.logical_axis_to_physical(n, namespace) for n in flattened_logical_axis_names
            ]
        flattened_physical_axes, _ = jax.tree.flatten(physical_axes)
        return axis_group_size(flattened_physical_axes, self.mesh)


PerNamespaceShardingConfig = dict[str, ShardingSpec]
ShardingConfig = dict[str, PerNamespaceShardingConfig]


def make_sharding_context_from_config(
    name: str, mesh: Mesh, config: ShardingConfig
) -> ShardingContext:
    ctx = ShardingContext(name, mesh)

    def _sharding_rule(
        named_dim: str, config: PerNamespaceShardingConfig = None, namespace: str = None
    ) -> ShardingSpec:
        if named_dim not in config:
            raise ValueError(f"Unknown named dimension: {named_dim} for namespace: {namespace}")
        return config[named_dim]

    for namespace, c in config.items():
        ctx.register_sharding_rule(namespace)(
            partial(_sharding_rule, config=c, namespace=namespace)
        )
    return ctx


default_sharding_config = {
    "default": {
        "batch": ("expert", "replica", "data"),
        "batch_attn": ("expert", "replica", "data"),
        "dense_activation_model": "model",
        "dense_activation_seq": "seq",
        "embed": None,
        "head": ("seq", "model"),
        "hidden": None,
        "model": None,
        "replicated": None,
        "sequence": ("seq", "model"),
        "unsharded": None,
        "unspecified": None,
    },
    "attention": {
        "q_head": ("seq", "model"),
        "kv_head": ("seq", "model"),
        "sequence": None,
        "activation_seq": ("seq", "model"),
    },
}


def make_legacy_sharding_context(mesh: Mesh) -> ShardingContext:
    return make_sharding_context_from_config("legacy", mesh, default_sharding_config)
