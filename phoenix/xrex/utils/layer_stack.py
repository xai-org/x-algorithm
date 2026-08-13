# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Protocol

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from xrex.models.model_utils import Parameter, make_global_reduction_fn

logger = logging.getLogger(__name__)


LayerStackCarry = collections.namedtuple("LayerStackCarry", ["idx", "x"])
LayerStackScanned = collections.namedtuple(
    "LayerStackScanned", ["params", "rng", "state", "args_ys"]
)

NestedArray = Any
WrappedFn = Callable[..., NestedArray | tuple[NestedArray]]


def _check_no_varargs(f: Callable[..., Any]) -> None:
    if list(inspect.signature(f).parameters.values())[0].kind == inspect.Parameter.VAR_POSITIONAL:
        raise ValueError(
            "The function `f` should not have any `varargs` (that is *args) "
            "argument. Instead, it should only use explicit positional"
            "arguments"
        )


def _get_rng_stack(count: int) -> jax.Array | None:
    rng = hk.maybe_next_rng_key()
    if rng is None:
        return None
    return jax.random.split(rng, count)


def _apply_generator_fn_to_parameter_states(
    params: hk.Params, param_states: hk.MutableState
) -> hk.MutableState:
    all_common_keys = list(set(params.keys()) & set(param_states.keys()))
    all_common_keys.sort()
    for key in all_common_keys:
        per_layers_params = params[key]
        param_names = set(per_layers_params.keys())
        for param_name, param_state in param_states[key].items():
            if param_state.parameter_name in param_names:
                param = per_layers_params[param_state.parameter_name]
                assert isinstance(param, Parameter)

                global_reduction_fn = make_global_reduction_fn(
                    local_reducer=partial(
                        lambda x, ps: jax.vmap(ps.local_reducer, in_axes=0, out_axes=0)(x),
                        ps=param_state,
                    ),
                    global_reduction_op=param_state.global_reduction_op,
                    param_pspec=param.pspec,
                    param_state_pspec=P(None, *param_state.param_state_sharding.spec),
                    mesh=param_state.param_state_sharding.mesh,
                    parameter_name=param_state.parameter_name,
                )
                new_state = global_reduction_fn(param.x)
                param_states[key][param_name] = replace(
                    param_state, state=new_state, reduced_from_param=True
                )
    return param_states


def _get_lifted_state_keys(state: hk.MutableState) -> set[tuple[str, str]]:
    keys = set()
    for key in state.keys():
        for k in state[key].keys():
            keys.add((key, k))
    return keys


def _merge_inner_state(
    state: hk.MutableState, recollected_state: hk.MutableState
) -> hk.MutableState:
    lifted_state_keys = _get_lifted_state_keys(state)

    for key in recollected_state.keys():
        if key not in state:
            state[key] = recollected_state[key]
            continue

        per_layers_state = state[key]
        for k, v in recollected_state[key].items():
            if (key, k) in lifted_state_keys:
                continue
            if k in per_layers_state:
                raise ValueError(f"Found conflicting values for param {key}/{k} with outer state.")
            per_layers_state[k] = v
        state[key] = per_layers_state
    return state


class TransparencyMapping(Protocol):
    def stacked_to_flat(self, stacked_module_name: str, scan_idx: int) -> str: ...

    def flat_to_stacked(self, unstacked_module_name: str) -> tuple[str, int] | None: ...


def split_params(
    stacked_params: hk.Params,
    num_layers: int,
    name_map: TransparencyMapping,
    repeat_param_name: str | None = None,
) -> hk.Params:
    def _split(x):
        return [jnp.squeeze(s, axis=0) for s in jnp.split(x, x.shape[0], axis=0)]

    params = {}
    for mod_name, mod_params in stacked_params.items():
        if repeat_param_name and repeat_param_name not in mod_name:
            params[mod_name] = mod_params
            continue
        split_mod_params = {k: _split(v) for k, v in mod_params.items()}
        for i in range(num_layers):
            new_mod_name = name_map.stacked_to_flat(mod_name, i)
            if new_mod_name in params:
                raise ValueError(
                    f"Found conflicting unstacked module name for {mod_name} at {new_mod_name}."
                )
            params[new_mod_name] = {k: v[i] for k, v in split_mod_params.items()}

    return params


def stack_params(
    split_params: hk.Params,
    num_layers: int,
    name_map: TransparencyMapping,
    repeat_param_name: str | None = None,
) -> hk.Params:
    params = {}
    make_empty_param_stack = lambda: ([None] * num_layers)

    for mod_name, mod_params in split_params.items():
        if repeat_param_name and repeat_param_name not in mod_name:
            params[mod_name] = mod_params
            continue
        stacked_name_idx = name_map.flat_to_stacked(mod_name)
        if stacked_name_idx is None:
            continue
        stacked_mod_name, idx = stacked_name_idx
        if stacked_mod_name not in params:
            params[stacked_mod_name] = collections.defaultdict(make_empty_param_stack)

        for k, v in mod_params.items():
            if params[stacked_mod_name][k][idx] is not None:
                raise ValueError(
                    f"Found conflicting values for param {stacked_mod_name}/{k} at index {idx}."
                )
            params[stacked_mod_name][k][idx] = v

    for mod_name, mod_params in params.items():
        if repeat_param_name and repeat_param_name not in mod_name:
            continue
        for k, v in mod_params.items():
            if None in v:
                raise ValueError(f"Couldn't find all params for {mod_name}/{k}: {v}")
            mod_params[k] = jnp.stack(v, axis=0)

    return params


class _LayerStack:
    def __init__(
        self,
        count: int,
        unroll: int,
        pass_reverse_to_layer_fn: bool = False,
        transparency_map: TransparencyMapping | None = None,
        name: str = "",
        scan_impl: Callable = jax.lax.scan,
    ):
        self._name = name
        self._count = count
        self._unroll = unroll
        self._pass_reverse_to_layer_fn = pass_reverse_to_layer_fn
        self._transparency_map = transparency_map
        self._scan_impl = scan_impl

    def __call__(self, x, *args_ys, reverse=False, **kwargs):
        count = self._count
        init_fn, apply_fn = hk.transform_with_state(self._call_wrapped)

        def per_layer_init_fn(c, a):
            c, rng = c
            if rng is not None:
                rng, next_rng, apply_rng = jax.random.split(rng, 3)
            else:
                rng, next_rng, apply_rng = None, None, None

            params, state = init_fn(rng, c, *a, **kwargs)
            (c, _), state = apply_fn(params, state, apply_rng, c, *a, **kwargs)
            return (c, next_rng), (params, state)

        def scanned_init_fn(x, rng):
            _, (params, state) = self._scan_impl(
                per_layer_init_fn, (x, rng), args_ys, length=self._count
            )
            if self._transparency_map is not None:
                return (
                    split_params(params, self._count, self._transparency_map),
                    split_params(state, self._count, self._transparency_map),
                )
            else:

                def update_pspec(x):
                    pspec = P(None) if x.pspec is None else x.pspec
                    return replace(x, pspec=P(None, *pspec))

                params = jax.tree.map(
                    update_pspec, params, is_leaf=lambda x: isinstance(x, Parameter)
                )
                return params, state

        rng = hk.maybe_next_rng_key()

        if self._transparency_map is not None:
            params_and_state_fn, updater = hk.experimental.transparent_lift_with_state(
                scanned_init_fn, allow_reuse=True
            )
        else:
            params_and_state_fn, updater = hk.lift_with_state(
                scanned_init_fn, allow_reuse=True, name=self._name
            )
        params, state = params_and_state_fn(x, rng)
        if not hk.running_init():
            _, recollected_state = scanned_init_fn(jax.lax.stop_gradient(x), rng)
            state = _merge_inner_state(state, recollected_state)

        def layer(
            carry: LayerStackCarry, scanned: LayerStackScanned
        ) -> tuple[LayerStackCarry, Any]:
            rng = scanned.rng
            params = scanned.params
            state = scanned.state

            kwargs = {}
            kwargs["global_layer_index"] = carry.idx
            if self._pass_reverse_to_layer_fn:
                kwargs["reverse"] = reverse
            (out_x, z), state = apply_fn(params, state, rng, carry.x, *scanned.args_ys, **kwargs)
            return LayerStackCarry(idx=carry.idx + 1, x=out_x), (z, state)

        rng = _get_rng_stack(count)

        if self._transparency_map is not None:
            params = stack_params(params, self._count, self._transparency_map)
            state = stack_params(state, self._count, self._transparency_map)

        global_layer_index = jnp.zeros([], dtype=int)
        carry = LayerStackCarry(idx=global_layer_index, x=x)

        with jax.named_scope("lifted_parameter_states"):
            lifted_state = _apply_generator_fn_to_parameter_states(params, state)
        scanned = LayerStackScanned(params=params, rng=rng, state=lifted_state, args_ys=args_ys)

        if self._unroll == self._count:
            yss = []
            states = []
            for i in range(self._count):
                xs = jax.tree.map(lambda x, i=i: x[i], scanned)
                carry, (ys, state) = layer(carry, xs)
                yss.append(ys)
                states.append(state)

            states = jax.tree.map(jnp.stack, *states)
            updater.update(states)
            return carry.x, jax.tree.map(jnp.stack, *yss)
        else:
            carry, (zs, states) = self._scan_impl(
                layer, carry, scanned, length=count, unroll=self._unroll, reverse=reverse
            )
            updater.update(states)
            return carry.x, zs

    def _call_wrapped(
        self,
        x: jax.Array,
        *args,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array | None]:
        raise NotImplementedError()


class _LayerStackNoPerLayer(_LayerStack):
    def __init__(
        self,
        f: WrappedFn,
        count: int,
        unroll: int,
        pass_reverse_to_layer_fn: bool = False,
        transparency_map: TransparencyMapping | None = None,
        name: str = "",
        scan_impl: Callable = jax.lax.scan,
    ):
        super().__init__(
            count=count,
            unroll=unroll,
            pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
            transparency_map=transparency_map,
            name=name,
            scan_impl=scan_impl,
        )
        _check_no_varargs(f)
        self._f = f

    @hk.transparent
    def _call_wrapped(self, args, y, **kwargs):
        del y
        ret = self._f(*args, **kwargs)
        if len(args) == 1:
            ret = (ret,)
        return ret, None


class _LayerStackWithPerLayer(_LayerStack):
    def __init__(
        self,
        f: WrappedFn,
        count: int,
        unroll: int,
        pass_reverse_to_layer_fn: bool = False,
        transparency_map: TransparencyMapping | None = None,
        name: str = "",
        scan_impl: Callable = jax.lax.scan,
    ):
        super().__init__(
            count=count,
            unroll=unroll,
            pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
            transparency_map=transparency_map,
            name=name,
            scan_impl=scan_impl,
        )
        self._f = f

    @hk.transparent
    def _call_wrapped(self, x, *args, **kwargs):
        return self._f(x, *args, **kwargs)


def layer_stack(
    num_layers: int,
    with_per_layer_inputs=False,
    unroll: int = 1,
    pass_reverse_to_layer_fn: bool = False,
    transparent: bool = False,
    transparency_map: TransparencyMapping | None = None,
    name: str = "layer_stack",
    scan_impl: Callable = jax.lax.scan,
):
    if transparent and transparency_map is None:
        raise ValueError("transparency_map must be provided with transparent=True.")
    if not transparent:
        transparency_map = None

    def iterate(f):
        if with_per_layer_inputs:

            @functools.wraps(f)
            def wrapped(x, *args, **kwargs):
                for ys in jax.tree_util.tree_leaves(args):
                    assert ys.shape[0] == num_layers
                return _LayerStackWithPerLayer(
                    f,
                    num_layers,
                    unroll=unroll,
                    pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
                    transparency_map=transparency_map,
                    name=name,
                    scan_impl=scan_impl,
                )(x, *args, **kwargs)

        else:
            _check_no_varargs(f)

            @functools.wraps(f)
            def wrapped(*args, **kwargs):
                ret = _LayerStackNoPerLayer(
                    f,
                    num_layers,
                    unroll=unroll,
                    pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
                    transparency_map=transparency_map,
                    name=name,
                    scan_impl=scan_impl,
                )(args, None, **kwargs)[0]
                if len(args) == 1:
                    ret = ret[0]
                return ret

        return wrapped

    return iterate
