# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import json
import logging
import os
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from xai_gimmick import contract

from xai_checkpointing import common
from xai_checkpointing.tree_util import tree_to_dict

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

ADLER_BASE = 65521

CHECKSUM_CORRUPT_FILENAME = "checksum_corrupt.json"


class ChecksumInconsistencyError(RuntimeError):
    pass


def compute_checksums(state, state_sharding, mesh):
    state_dict = tree_to_dict(state)
    sharding_dict = tree_to_dict(state_sharding)

    none_keys = [k for k, v in state_dict.items() if v is None]
    for k in none_keys:
        del state_dict[k]
        sharding_dict.pop(k, None)
        sharding_dict.pop(k.removesuffix(".0"), None)

    for key in state_dict:
        sharding = sharding_dict.pop(key.removesuffix(".0"))
        sharding_dict[key] = sharding

    sharding_dict = {key: sharding_dict[key] for key in state_dict}

    allgather_sharding = jax.sharding.NamedSharding(mesh, P())

    return dict(
        shard_checksums=jax.lax.with_sharding_constraint(
            checksum_tree(state_dict, sharding_dict, mesh, combine=False),
            {key: allgather_sharding for key in state_dict},
        ),
        global_checksums=global_checksum_tree(state_dict, sharding_dict, mesh),
    )


@functools.cache
def _slices2devices(shape, spec, mesh_items):
    slices2devices = {}
    for device, slices in _get_slices(
        shape=shape,
        spec=spec,
        mesh=dict(mesh_items),
    ):
        slices = tuple(s.indices(d) for s, d in zip(slices, shape, strict=True))
        slices2devices.setdefault(slices, []).append(device)
    return slices2devices


def check_consistency_python(state, shard_checksums, label="state"):
    jax.copy_to_host_async(shard_checksums)
    state_dict = tree_to_dict(state)
    shard_checksums = tree_to_dict(shard_checksums)

    errors = []
    for key, tensor in state_dict.items():
        slices2devices = _slices2devices(
            tuple(tensor.shape), tensor.sharding.spec, tuple(tensor.sharding.mesh.shape.items())
        )
        try:
            check_consistent(
                slices2devices, np.array(shard_checksums[key]), label, key, list(tensor.shape)
            )
        except RuntimeError as e:
            errors.append(e)
    if errors:
        raise RuntimeError("\n".join(str(e) for e in errors))


def get_checksum_dict(state, state_sharding, mesh, checksums=None):
    shardings = {}
    dtypes = {}
    shapes = {}

    state_dict = tree_to_dict(state)
    sharding_dict = tree_to_dict(state_sharding)

    for key in state_dict:
        tensor = state_dict[key]
        if tensor is None:
            continue

        sharding = sharding_dict[key]

        shapes[key] = list(tensor.shape)
        local_mesh = getattr(sharding, "mesh", None)

        shardings[key] = {
            "spec": list(sharding.spec),
        }
        if local_mesh:
            shardings[key]["mesh"] = local_mesh.shape
        else:
            shardings[key]["mesh"] = None
        dtypes[key] = jnp.dtype(tensor.dtype).str

    checksum_dict = {
        "version": 3,
        "format": {},
        "shapes": shapes,
        "dtypes": dtypes,
        "shardings": shardings,
    }

    if checksums is not None:
        if "shard_checksums" in checksums:
            shard_checksums = tree_to_dict(checksums["shard_checksums"])
            checksum_dict["format"]["shard_checksums"] = "adler32"
            checksum_dict["shard_checksums"] = {
                k: v.tolist() for k, v in shard_checksums.items() if k in shapes
            }
        if "global_checksums" in checksums:
            global_checksums = tree_to_dict(checksums["global_checksums"])
            checksum_dict["format"]["global_checksums"] = "global_adler32"
            checksum_dict["global_checksums"] = {
                k: v.item() for k, v in global_checksums.items() if k in shapes
            }

    return checksum_dict


def adler32_combine(adlers, lengths, BASE=ADLER_BASE):
    A = adlers & 0xFFFF
    B = (adlers >> 16) & 0xFFFF
    S = (A - 1) % BASE

    lengths = lengths % BASE

    L_after = jnp.roll(lengths[::-1].cumsum(axis=0), 1, axis=0).at[0].set(0)[::-1] % BASE

    B_terms = (B + S * L_after) % BASE
    B_total = jnp.sum(B_terms, axis=0) % BASE
    A_total = (1 + jnp.sum(S, axis=0)) % BASE

    adler32_combined = (B_total << 16) | A_total
    return adler32_combined


def checksum_tree(tree, shardings, mesh, combine=True):
    from xai_checkpointing import adler32_zlib as adler32

    in_specs = jax.tree.map(lambda s: s.spec, shardings)

    out_spec = P(mesh.axis_names)
    out_specs = jax.tree.map(lambda _: out_spec, shardings)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(in_specs,),
        out_specs=(out_specs, out_specs),
        check_vma=False,
    )
    def local_adler32(tree):
        adlers = jax.tree.map(adler32.adler32, tree)
        lengths = jax.tree.map(lambda a: jnp.uint32(a.nbytes % ADLER_BASE), tree)
        return jax.tree.map(lambda a: a.reshape([1]), (adlers, lengths))

    adlers, lengths = local_adler32(tree)

    if not combine:
        return adlers

    adlers_, _ = jax.tree.flatten(adlers)
    lengths_, _ = jax.tree.flatten(lengths)
    combined = adler32_combine(jnp.stack(adlers_), jnp.stack(lengths_))

    if combine == "both":
        return adlers, combined

    return combined


def global_checksum_tree(arrays, shardings, mesh):
    from xai_checkpointing import adler32_zlib as adler32

    return jax.tree.map(
        lambda array, sharding: adler32.global_adler32_fast(
            data=array, mesh=mesh, pspec=sharding.spec
        ),
        arrays,
        shardings,
    )


def check_consistency_xla(
    shard_checksums: common.PyTree[jax.Array],
    sharding: common.PyTree[jax.sharding.NamedSharding],
    mesh: jax.sharding.Mesh,
):
    axes = common.get_replicated_axes(sharding)

    checksums_pspecs = jax.tree.map(lambda _: P(mesh.axis_names), shard_checksums)
    replicated_pspecs = jax.tree.map(lambda _: P(), shard_checksums)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(checksums_pspecs,),
        out_specs=replicated_pspecs,
        check_vma=False,
    )
    def reduce_and_compare(shard_checksums):
        reduced_checksums = jax.tree.map(
            lambda array, axes: jax.lax.psum(array, tuple(axes)), shard_checksums, axes
        )
        axis_sizes = jax.tree.map(lambda axes: jax.lax.axis_size(tuple(axes)), axes)
        return jax.tree.map(
            lambda a, n, b: jnp.array_equal(a, jnp.uint32(n) * b),
            reduced_checksums,
            axis_sizes,
            shard_checksums,
        )

    return reduce_and_compare(shard_checksums)


class ConsistencyState(NamedTuple):
    shard_checksums: common.PyTree[jax.Array]
    combined_shard_checksums: jax.Array
    consistency: common.PyTree[jax.Array]
    is_consistent: jax.Array


def get_consistency(
    tree: common.PyTree[jax.Array],
    sharding: common.PyTree[jax.sharding.NamedSharding],
    mesh: jax.sharding.Mesh,
) -> ConsistencyState:
    shard_checksums, combined_shard_checksums = checksum_tree(tree, sharding, mesh, combine="both")
    consistency = check_consistency_xla(shard_checksums, sharding, mesh)

    is_consistent = jax.tree.reduce(jnp.logical_and, consistency)
    is_consistent = jax.shard_map(
        functools.partial(jax.lax.pmin, axis_name=mesh.axis_names),
        mesh=mesh,
        in_specs=(P(),),
        out_specs=P(),
        check_vma=False,
    )(is_consistent)

    return ConsistencyState(
        shard_checksums=shard_checksums,
        combined_shard_checksums=combined_shard_checksums,
        consistency=consistency,
        is_consistent=is_consistent,
    )


class _FakeDevice:
    def __init__(self, id):
        self.id = id


def _get_slices(shape: list[int], spec: list[None | str | list[str]], mesh: dict[str, int]):
    mesh_shape = list(mesh.values())
    num_devices = np.prod(mesh_shape)
    devices = np.array([_FakeDevice(i) for i in range(num_devices)]).reshape(mesh_shape)

    jax_mesh = jax.sharding.Mesh(devices, tuple(mesh.keys()))
    jax_pspec = jax.sharding.PartitionSpec(*spec)

    sharding = jax.sharding.NamedSharding(jax_mesh, jax_pspec)
    for device, slices in sharding.devices_indices_map(tuple(shape)).items():
        yield device.id, slices


def format_slices(slices, shape):
    parts = []
    for dim, (start, stop, step) in zip(shape, slices, strict=True):
        assert step == 1
        if not start:
            start = ""
        if stop == dim:
            stop = ""
        parts.append(f"{start}:{stop}")
    return "[%s]" % ", ".join(parts)


def check_consistent(slices2devices, checksums, fn, key, shape):
    errors = []
    for slices, devices in slices2devices.items():
        subset = [checksums[device] for device in devices]
        values = {}
        for device, checksum in zip(devices, subset, strict=True):
            values.setdefault(f"0x{checksum:08x}", []).append(device)
        if len(values) > 1:
            values = {adler: contract(ranks) for adler, ranks in values.items()}
            errors.append(
                f"In slice {format_slices(slices, shape)}: Seeing values {values} across {len(devices)} devices."
            )
    errors = "\n".join(errors)
    if errors:
        raise ChecksumInconsistencyError(
            "Checksums internally inconsistent: "
            f"Mismatch in {fn} for tensor {key!r} ({shape=}).\n{errors}"
        )


def check_internal_consistency(checksum_dict, fn="<in-memory checksum_dict>", names=None):
    shard_checksums = checksum_dict.get("shard_checksums")
    if not shard_checksums:
        return

    for key, checksums in shard_checksums.items():
        if names is not None and key not in names:
            continue

        sharding = checksum_dict["shardings"][key]
        mesh = sharding.get("mesh")
        if not mesh:
            continue

        shape = checksum_dict["shapes"][key]
        spec = tuple(tuple(e) if isinstance(e, list) else e for e in sharding["spec"])
        slices2devices = _slices2devices(tuple(shape), spec, tuple(mesh.items()))
        check_consistent(slices2devices, checksums, fn, key, shape)


def verify_at_save(
    checksum_dict, path, elapsed_samples, process_index
) -> ChecksumInconsistencyError | None:
    try:
        check_internal_consistency(checksum_dict)
        return None
    except ChecksumInconsistencyError as e:
        if process_index == 0:
            try:
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "checksums.0.json"), "w") as f:
                    json.dump(checksum_dict, f)
                marker_path = os.path.join(path, CHECKSUM_CORRUPT_FILENAME)
                with open(marker_path, "w") as f:
                    json.dump({"elapsed_samples": elapsed_samples, "reason": str(e)}, f)
                rank_logger.error("Wrote corruption marker to %s", marker_path)
            except OSError:
                rank_logger.exception("Failed to write checksum corruption diagnostics to %s", path)
        return e


def compare_checksum_dicts(fn, computed, rename=None, names=None):
    with open(fn) as f:
        loaded = json.load(f)

    version = loaded.get("version", 1)
    if version < 2:
        return False

    if rename:
        for field in ("dtypes", "shapes", "shard_checksums", "shardings"):
            loaded[field] = {rename(k): v for k, v in loaded[field].items()}

    if version < 3:
        for key in ("shapes", "dtypes", "shardings", "shard_checksums"):
            for name in tuple(loaded[key].keys()):
                value = loaded[key].pop(name)
                loaded[key][name.removesuffix(".0")] = value

    success = True
    for key, cchecksums in computed["shard_checksums"].items():
        if names and key not in names:
            continue

        lsharding = loaded["shardings"].get(key)
        csharding = dict(computed["shardings"][key])

        cshape = computed["shapes"].get(key)
        lshape = loaded["shapes"].get(key)

        lchecksums = loaded["shard_checksums"].get(key)

        if not lsharding:
            if success:
                rank_logger.warning(
                    "Checkpoint checksum file %s doesn't contain expected key %s (only first name logged). Available keys: %s",
                    fn,
                    key,
                    json.dumps(list(loaded["shardings"]), indent=2),
                )
                success = False
            continue

        assert cshape == lshape, f"Shape of {key} differs: {cshape} vs {lshape} in {fn}"

        lsharding["spec"] = [tuple(e) if isinstance(e, list) else e for e in lsharding["spec"]]

        lslices2devices = _slices2devices(
            tuple(lshape), tuple(lsharding["spec"]), tuple(lsharding["mesh"].items())
        )

        check_consistent(lslices2devices, lchecksums, fn, key, lshape)

        cdevice2slices = {}
        for device, slices in _get_slices(
            shape=cshape, spec=csharding["spec"], mesh=csharding["mesh"]
        ):
            slices = tuple(s.indices(d) for s, d in zip(slices, cshape, strict=True))
            cdevice2slices[device] = slices

        for cdevice, cvalue in enumerate(cchecksums):
            slices = cdevice2slices[cdevice]
            try:
                ldevice = min(lslices2devices[slices])
            except KeyError:
                if success:
                    rank_logger.warning(
                        "Sharding of key %s changed from %s to %s, cannot check checksums",
                        key,
                        lsharding,
                        csharding,
                    )
                    success = False
                continue

            lvalue = lchecksums[ldevice]

            if cvalue != lvalue:
                extra = ""
                if len(cchecksums) > 1:
                    extra = f" at device {cdevice}, slice {format_slices(slices, cshape)}"
                raise RuntimeError(
                    f"Checksum of key {key} in {fn} differs"
                    f"{extra}: 0x{cvalue:08x} vs 0x{lvalue:08x}"
                )

    if version == 2:
        return success

    for key, cchecksum in computed["global_checksums"].items():
        if names and key not in names:
            continue

        try:
            lchecksum = loaded["global_checksums"][key]
        except KeyError as e:
            if success:
                rank_logger.warning("%s reading checksum file %s: %s", e.__class__.__name__, fn, e)
                success = False
            continue
        if cchecksum != lchecksum:
            raise RuntimeError(
                f"Global checksum of key {key} in {fn} differs: 0x{cchecksum:08x} vs 0x{lchecksum:08x}"
            )

    return success
