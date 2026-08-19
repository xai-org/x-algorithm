# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import ctypes
import functools
import logging
import os
import sys
import unittest
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.lax
import jaxlib.mlir.ir as ir
from jax._src import dispatch, source_info_util
from jax._src.ad_checkpoint import remat_p
from jax._src.core import Atom, DropVar
from jax._src.deprecations import deprecation_getattr
from jax._src.interpreters import ad
from jax._src.lax.lax import (
    _dot_general_transpose_lhs,
    _dot_general_transpose_rhs,
    dot_general_p,
)
from jax._src.util import merge_lists, partition_list
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Primitive, Var
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import batching, mlir

rank_logger = logging.getLogger("rank")


def patch_dot_general():
    _dot_general_transpose_lhs_patched = source_info_util.extend_name_stack("transpose_lhs")(
        _dot_general_transpose_lhs
    )
    _dot_general_transpose_rhs_patched = source_info_util.extend_name_stack("transpose_rhs")(
        _dot_general_transpose_rhs
    )
    ad.defbilinear(
        dot_general_p, _dot_general_transpose_lhs_patched, _dot_general_transpose_rhs_patched
    )


def patch_xla_extensions():
    from jax import errors
    from jax.lib import xla_extension

    if "XlaRuntimeError" in xla_extension._deprecations:
        del xla_extension._deprecations["XlaRuntimeError"]
        xla_extension.__getattr__ = deprecation_getattr(
            xla_extension.__name__, xla_extension._deprecations
        )
        xla_extension.XlaRuntimeError = errors.JaxRuntimeError


def check_jax_cuda_version():
    import jax

    rank_logger.info(f"Running with Jax version: {jax.__version__}")

    if sys.platform != "linux":
        return

    try:
        from jaxlib.cuda import _versions as cuda_versions
    except ImportError:
        try:
            from jax_cuda13_plugin import _versions as cuda_versions
        except ImportError:
            rank_logger.warning("Failed to import CUDA plugin versions. Maybe running on CPU?")
            return

    try:
        with open("/sys/module/nvidia/version") as f:
            cuda_driver_version = f.read().strip()
        rank_logger.info(f"Running with CUDA driver version: {cuda_driver_version}")
    except OSError:
        rank_logger.error("Could not read CUDA driver version")

    rank_logger.info(
        f"Running with CUDA runtime version: {cuda_versions.cuda_runtime_get_version()}"
    )
    rank_logger.info(f"Running with CUDNN version: {cuda_versions.cudnn_get_version()}")
    rank_logger.info(f"Running with CUBLAS version: {cuda_versions.cublas_get_version()}")

    if cuda_versions.cuda_runtime_get_version() // 10 == 1204:
        rank_logger.info("Skipping 'nvidia.nccl' import for CUDA 12.4")
        return

    try:
        from nvidia import nccl
    except ImportError:
        rank_logger.exception("Failed to import nvidia.nccl to check version")
        return

    libnccl = ctypes.CDLL(os.path.join(nccl.__path__[0], "lib/libnccl.so.2"))
    version = ctypes.c_int()
    status = libnccl.ncclGetVersion(ctypes.byref(version))
    if status != 0:
        raise RuntimeError(f"Failed to get NCCL version from loaded libnccl.so.2: {status=}")

    rank_logger.info(f"Running with NCCL version: {version.value}")


def patch_shard_map_transpose():
    pass


xla_metadata_value_p = Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(partial(dispatch.apply_primitive, xla_metadata_value_p))
xla_metadata_value_p.def_abstract_eval(
    lambda aval, *, xla_metadata_kvs, frontend_attributes_fn=None: aval
)
batching.defvectorized(xla_metadata_value_p)
ad.deflinear2(xla_metadata_value_p, lambda ct, _, **kwargs: (ct,))


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, val: ir.Value, *, xla_metadata_kvs, frontend_attributes_fn=None
):
    xla_metadata = dict(xla_metadata_kvs)
    if frontend_attributes_fn is not None:
        xla_metadata = frontend_attributes_fn(ctx) | xla_metadata
    op_to_attach_metadata = _target_op_to_attach_metadata(val)
    if op_to_attach_metadata is not None:
        _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
    return [val]


if jax.__version__ >= "0.8.0":
    mlir.register_lowering(xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False)
else:
    mlir.register_lowering(xla_metadata_value_p, _xla_metadata_value_lowering_rule)


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
    op = value_mlir.owner
    if op is None or isinstance(op, ir.Block):
        return None
    return op


def _attach_xla_metadata_to_op(xla_metadata: dict[str, Any], op: ir.Operation) -> None:
    if xla_metadata:
        ctx_attributes, existing_attributes = {}, {}
        for k, v in xla_metadata.items():
            ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
        for attr in op.attributes:
            if attr == "mhlo.frontend_attributes":
                for a in op.attributes[attr]:
                    existing_attributes[a.name] = a.attr
        op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
            ctx_attributes | existing_attributes
        )


def set_xla_metadata_with_context(
    x=None,
    frontend_attributes_fn: Callable[[mlir.LoweringRuleContext], dict[str, Any]] = None,
    **kwargs,
):
    hashable_metadata = tuple(sorted(kwargs.items()))
    return jax.tree.map(
        lambda v: xla_metadata_value_p.bind(
            v, xla_metadata_kvs=hashable_metadata, frontend_attributes_fn=frontend_attributes_fn
        ),
        x,
    )


no_barrier_p = Primitive("no_barrier")
no_barrier_p.multiple_results = False


no_barrier_p.def_impl(lambda x: x)
no_barrier_p.def_abstract_eval(lambda x: x)


def no_barrier_jvp(primals, tangents):
    (x,), (xdot,) = primals, tangents
    return no_barrier_p.bind(x), xdot


ad.primitive_jvps[no_barrier_p] = no_barrier_jvp

mlir.register_lowering(no_barrier_p, lambda ctx, x: [x])


def no_barrier_batcher(args, dims):
    (x,), (d,) = args, dims
    return no_barrier_p.bind(x), d


batching.primitive_batchers[no_barrier_p] = no_barrier_batcher


def no_barrier(x):
    return jax.tree.map(no_barrier_p.bind, x)


def _find_no_barrier_inputs(jaxpr: Jaxpr) -> tuple[bool, ...]:
    env: dict[Var, bool] = {}

    def is_marked(var: Atom) -> bool:
        if isinstance(var, Literal):
            return False
        return env.get(var, False)

    def mark_var(var: Atom):
        if isinstance(var, Var):
            env[var] = True

    for eqn in jaxpr.eqns[::-1]:
        if eqn.primitive is no_barrier_p:
            assert len(eqn.invars) == 1 and len(eqn.outvars) == 1
            mark_var(eqn.invars[0])
        else:
            nested_jaxprs = []
            for _, param_val in eqn.params.items():
                if isinstance(param_val, Jaxpr):
                    nested_jaxprs.append(param_val)
                elif isinstance(param_val, ClosedJaxpr):
                    nested_jaxprs.append(param_val.jaxpr)

            if nested_jaxprs:
                for nested_jaxpr in nested_jaxprs:
                    nested_skip = _find_no_barrier_inputs(nested_jaxpr)
                    for i, should_skip in enumerate(nested_skip):
                        if should_skip and i < len(eqn.invars):
                            mark_var(eqn.invars[i])
            else:
                non_drop_outvars = [v for v in eqn.outvars if not isinstance(v, DropVar)]
                if len(eqn.invars) == 1 and len(non_drop_outvars) == 1:
                    if is_marked(non_drop_outvars[0]):
                        mark_var(eqn.invars[0])

    return tuple(is_marked(var) for var in jaxpr.invars)


def _patched_remat_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: Jaxpr,
    prevent_cse: bool,
    differentiated: bool,
    policy,
):
    if isinstance(prevent_cse, bool):
        if prevent_cse:
            no_barrier_skip = _find_no_barrier_inputs(jaxpr)
            prevent_cse = tuple(not skip for skip in no_barrier_skip)
        else:
            prevent_cse = (False,) * len(ctx.avals_in)
    assert isinstance(prevent_cse, tuple)
    if differentiated and any(prevent_cse):
        _, barrier_avals = partition_list(prevent_cse, ctx.avals_in)
        other_args, barrier_args = partition_list(prevent_cse, args)
        barrier_op = hlo.OptimizationBarrierOp(mlir.flatten_ir_values(barrier_args))
        barrier_results = mlir.unflatten_ir_values_like_types(
            barrier_op.results, map(mlir.aval_to_ir_type, barrier_avals)
        )
        args = merge_lists(prevent_cse, other_args, barrier_results)
    outs, tokens_out = mlir.jaxpr_subcomp(
        ctx.module_context,
        jaxpr,
        ctx.name_stack.extend("checkpoint"),
        ctx.tokens_in,
        (),
        *args,
        dim_var_values=ctx.dim_var_values,
        const_lowering=ctx.const_lowering,
    )
    ctx.set_tokens_out(tokens_out)
    return outs


mlir.register_lowering(remat_p, _patched_remat_lowering)


def is_upstream_jax() -> bool:
    return "xai" not in jax.__version__


def skip_test_if(predicate: Callable[[], bool], reason: str):
    def skip(test_method):
        @functools.wraps(test_method)
        def test_method_wrapper(self, *args, **kwargs):
            if predicate():
                test_name = getattr(test_method, "__name__", "[unknown test]")
                raise unittest.SkipTest(f"Skipping {test_name} because {reason}")
            return test_method(self, *args, **kwargs)

        return test_method_wrapper

    return skip


def skip_test_if_upstream_jax():
    return skip_test_if(
        is_upstream_jax, f" not running upstream JAX but running {jax.__version__}."
    )


def initialize_extensions():
    patch_dot_general()
    patch_xla_extensions()
    patch_shard_map_transpose()
    check_jax_cuda_version()
