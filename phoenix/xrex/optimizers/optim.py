# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from xai_configlib import Config, configclass

from xrex.models.model_utils import Parameter
from xrex.optimizers.schedule import BaseSchedule


def _convert_floats(x, dtype):
    if jax.dtypes.scalar_type_of(x) == float:
        return jnp.asarray(x, dtype=dtype)
    return x


def _instantiate(x: BaseSchedule | Any):
    return x.make() if isinstance(x, BaseSchedule) else x


class InjectHyperparamsState(NamedTuple):
    count: jnp.ndarray
    hyperparams: dict[str, chex.Numeric]
    inner_state: optax.OptState


def inject_hyperparams(
    inner_factory: Callable[..., optax.GradientTransformation],
    static_args: str | Iterable[str] = (),
) -> Callable[..., optax.GradientTransformationExtraArgs]:
    static_args = {static_args} if isinstance(static_args, str) else set(static_args)
    inner_signature = inspect.signature(inner_factory)

    if not static_args.issubset(inner_signature.parameters):
        raise ValueError(
            "`static_args` must specify a subset of `inner_factory`'s parameters. "
            f"Given `static_args`: {static_args}. `inner_factory` parameters: "
            f"{set(inner_signature.parameters.keys())}"
        )

    @functools.wraps(inner_factory)
    def wrapped_transform(*args, **kwargs) -> optax.GradientTransformation:
        bound_arguments = inner_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        sched_hps, numeric_hps, other_hps = {}, {}, {}
        for name, value in bound_arguments.arguments.items():
            if name in static_args or isinstance(value, bool):
                other_hps[name] = value
            elif callable(value):
                sched_hps[name] = value
            elif isinstance(value, (int, float, chex.Array)):
                numeric_hps[name] = value
            else:
                other_hps[name] = value

        def schedule_fn(count, dtype):
            return {k: _convert_floats(f(count), dtype) for k, f in sched_hps.items()}

        def init_fn(params):
            count = jnp.zeros([], jnp.int32)
            dtype = getattr(next(iter(jax.tree.leaves(params)), None), "dtype", None)
            hparams = {k: jnp.asarray(_convert_floats(v, dtype)) for k, v in numeric_hps.items()}
            hparams.update(schedule_fn(count, dtype))
            return InjectHyperparamsState(
                count, hparams, inner_factory(**other_hps, **hparams).init(params)
            )

        def update_fn(updates, state, params=None, **extra_args):
            count_inc = optax.safe_int32_increment(state.count)
            dtype = getattr(next(iter(jax.tree.leaves(updates)), None), "dtype", None)
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(count_inc, dtype))
            updates, inner_state = optax.with_extra_args_support(
                inner_factory(**other_hps, **hparams)
            ).update(updates, state.inner_state, params, **extra_args)

            return updates, InjectHyperparamsState(count_inc, hparams, inner_state)

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)

    return wrapped_transform


def apply_updates(
    params,
    updates,
    lr: float,
):
    return jax.tree.map(
        lambda p, u: jnp.asarray(p + lr * u).astype(jnp.asarray(p).dtype), params, updates
    )


class EmptyState(NamedTuple):
    pass


def init_empty_state(params):
    del params
    return EmptyState()


def scale_by_lr_multiplier() -> optax.GradientTransformation:
    def update_fn(updates, state, params=None):
        updates = jax.tree.map(
            lambda g, p: p.lr_multiplier * g,
            updates,
            params,
            is_leaf=lambda x: isinstance(x, Parameter),
        )
        return updates, state

    return optax.GradientTransformation(init_empty_state, update_fn)


_OPTIM_ALIASES: dict[str, str] = {}


@configclass
class OptimConfig(Config):
    optim: str
    learning_rate: BaseSchedule | float = 1.0
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.0
    clip_by_global_norm: float = 1.0

    def make(self):
        optim = _OPTIM_ALIASES.get(self.optim, self.optim)
        if optim not in ("adam",):
            raise NotImplementedError(f"unknown optim={optim!r}")

        @inject_hyperparams
        def schedule_optim(learning_rate, b1, b2, weight_decay):
            core = optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)
            return optax.chain(
                optax.clip_by_global_norm(self.clip_by_global_norm),
                core,
                scale_by_lr_multiplier(),
            )

        schedule_args = map(_instantiate, (self.learning_rate, self.b1, self.b2, self.weight_decay))
        return schedule_optim(*schedule_args)
