# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def get_layer_name(name_scope: str) -> str:
    patterns = [r"decoder_layer_(\d+)", r"decoder_layer_dense_(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, name_scope)
        if match:
            return match.group(0)
    return "unknown"


@dataclass
class Parameter:
    x: jax.Array
    pspec: P
    lr_multiplier: float = 1.0
    rms_clip_axes: Sequence[int] | None = None
    weight_decay_mask: float = 0.0

    @staticmethod
    def flatten(p):
        return (p.x,), p.fields

    @staticmethod
    def unflatten(fields, arrays):
        return Parameter(x=arrays[0], **fields)

    @property
    def fields(self):
        return dict(
            pspec=self.pspec,
            lr_multiplier=self.lr_multiplier,
            rms_clip_axes=self.rms_clip_axes,
            weight_decay_mask=self.weight_decay_mask,
        )

    @property
    def shape(self):
        return self.x.shape

    @property
    def dtype(self):
        return self.x.dtype

    def unwrap(self):
        return self.x

    def __mul__(self, m):
        if isinstance(m, Parameter):
            assert self.fields == m.fields
            return Parameter(self.x * m.x, **self.fields)
        else:
            return Parameter(self.x * m, **self.fields)

    def __add__(self, m):
        if isinstance(m, Parameter):
            assert self.fields == m.fields
            return Parameter(self.x + m.x, **self.fields)
        else:
            return Parameter(self.x + m, **self.fields)

    __rmul__ = __mul__
    __radd__ = __add__


jax.tree_util.register_pytree_node(Parameter, Parameter.flatten, Parameter.unflatten)


class GlobalReductionOp(Enum):
    SUM = 0
    MAX = 1
    MIN = 2
    MEAN = 3


def make_global_reduction_fn(
    local_reducer: Callable[[jax.Array], jax.Array],
    global_reduction_op: GlobalReductionOp,
    param_pspec: P,
    param_state_pspec: P,
    mesh: jax.sharding.Mesh,
    parameter_name: str,
) -> Callable[[jax.Array], jax.Array]:
    def _get_parallel_reduce_fn(global_reduction_op: GlobalReductionOp):
        match global_reduction_op:
            case GlobalReductionOp.SUM:
                return jax.lax.psum
            case GlobalReductionOp.MAX:
                return jax.lax.pmax
            case GlobalReductionOp.MIN:
                return jax.lax.pmin
            case GlobalReductionOp.MEAN:
                return jax.lax.pmean

    def global_reduction_fn(x: jax.Array) -> jax.Array:
        @jax.shard_map(
            mesh=mesh,
            in_specs=param_pspec,
            out_specs=param_state_pspec,
            check_vma=False,
        )
        def _run_reducer(x):
            parallel_reduce_fn = _get_parallel_reduce_fn(global_reduction_op)
            with jax.named_scope(f"local_reduction/{parameter_name}"):
                locally_reduced_x = local_reducer(jax.lax.stop_gradient(x))
            with jax.named_scope(f"global_reduction/{parameter_name}"):
                flattened_param_pspec, _ = jax.tree.flatten(tuple(param_pspec))
                replicated_axes = tuple(
                    [axis for axis in flattened_param_pspec if axis is not None]
                )
                return parallel_reduce_fn(locally_reduced_x, replicated_axes)

        return _run_reducer(x)

    return global_reduction_fn


def wrap_init_fn(init_fn, **kwargs):
    def wrapped(*a, **b):
        params = init_fn(*a, **b)
        return Parameter(
            x=params,
            **kwargs,
        )

    return wrapped


def unwrap_params(x):
    if isinstance(x, Parameter):
        return x.unwrap()
    else:
        return x


def unwrap_tree(tree):
    return jax.tree.map(unwrap_params, tree, is_leaf=lambda x: isinstance(x, Parameter))


def unwrap_getter(dtype_unwrap: jnp.dtype):
    def _unwrap_getter(next_getter, value, context):
        del context
        value = value.unwrap().astype(dtype_unwrap)
        return next_getter(value)

    return _unwrap_getter


def get_parameter(name, shape, dtype, init, rms_clip_axes=None, **kwargs):
    kwargs["rms_clip_axes"] = rms_clip_axes

    dtype_unwrap = kwargs.pop("dtype_unwrap", None)
    if dtype_unwrap is None:
        dtype_unwrap = jnp.bfloat16
    unwrap_getter_fn = unwrap_getter(dtype_unwrap)
    with hk.custom_getter(unwrap_getter_fn):
        p = hk.get_parameter(name, shape, dtype, wrap_init_fn(init, **kwargs))
    return p
