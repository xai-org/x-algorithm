# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import struct

import zstandard as zstd

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = data[pos]
        val |= (b & 0x7F) << shift
        shift += 7
        pos += 1
        if not (b & 0x80):
            break
    return val, pos


def _read_names(
    data: bytes, pos: int, base_paths_present: bool
) -> tuple[list[str], int, list[int]]:
    num_values, pos = _read_varint(data, pos)

    prefix_lens = [0]
    for _i in range(num_values - 1):
        x, pos = _read_varint(data, pos)
        prefix_lens.append(x)
    suffix_lens = []
    for _i in range(num_values):
        x, pos = _read_varint(data, pos)
        suffix_lens.append(x)

    base_path_lens = []
    if base_paths_present:
        for _i in range(num_values):
            x, pos = _read_varint(data, pos)
            base_path_lens.append(x)

    names = []
    for p_len, s_len in zip(prefix_lens, suffix_lens):
        x = data[pos : pos + s_len].decode()
        pos += s_len
        names.append(names[-1][:p_len] + x if p_len != 0 else x)

    return names, pos, base_path_lens


def _read_lists(data: bytes, pos: int, num_values: int, *lists) -> int:
    for l in lists:
        for _i in range(num_values):
            x, pos = _read_varint(data, pos)
            l.append(x)
    return pos


def _decompress(zd, data: bytes) -> bytes:
    zdo = zd.decompressobj()

    def _get():
        x = data
        while not zdo.eof:
            yield zdo.decompress(x)
            x = b""

    return b"".join(_get())


def _load_node(
    zd: object,
    ocdbt_path: str,
    prefix: str,
    fname: str,
    offset: int,
    size: int,
    shard_sources: list[tuple[str, str, int, int] | tuple[str, bytes]],
) -> None:
    with open(f"{ocdbt_path}/{fname}", "rb") as f:
        f.seek(offset)
        btree_header = f.read(size)
        if len(btree_header) != size:
            raise NotImplementedError("bad OCDBT B+tree node")

    if btree_header[:4] != b"\x0c\xdb\x20\xde":
        raise NotImplementedError("bad OCDBT B+tree node magic")
    if struct.unpack("<Q", btree_header[4:12])[0] != len(btree_header):
        raise NotImplementedError("bad OCDBT B+tree node size")
    if btree_header[12:14] != b"\0\1":
        raise NotImplementedError("OCDBT B+tree node is not version 0 zstd-compressed")

    btree = _decompress(zd, btree_header[14:-4])
    height = btree[0]
    pos = 1
    fnames, pos, _base_path_lens = _read_names(btree, pos, True)

    names, pos, _ = _read_names(btree, pos, height != 0)

    if height != 0:
        data_file_ids = []
        data_file_offsets = []
        data_file_lens = []
        nums_keys = []
        nums_tree_bytes = []
        nums_indirect_value_bytes = []
        pos = _read_lists(
            btree,
            pos,
            len(names),
            data_file_ids,
            data_file_offsets,
            data_file_lens,
            nums_keys,
            nums_tree_bytes,
            nums_indirect_value_bytes,
        )
        for fid, offset, size in zip(data_file_ids, data_file_offsets, data_file_lens):
            _load_node(zd, ocdbt_path, prefix, fnames[fid], offset, size, shard_sources)
    else:
        value_lens = []
        value_kinds = []
        pos = _read_lists(btree, pos, len(names), value_lens, value_kinds)

        num_indirect_entries = sum(value_kinds)
        value_data_file_ids = []
        value_data_offsets = []
        pos = _read_lists(btree, pos, num_indirect_entries, value_data_file_ids, value_data_offsets)

        j = 0
        for i, name in enumerate(names):
            if name.startswith(prefix):
                shard_sources.append(
                    (name, btree[pos : pos + value_lens[i]])
                    if value_kinds[i] == 0
                    else (
                        name,
                        fnames[value_data_file_ids[j]],
                        value_data_offsets[j],
                        value_lens[i] - 20,
                    )
                )
            if value_kinds[i] == 0:
                pos += value_lens[i]
            else:
                j += 1
    if pos != len(btree):
        raise NotImplementedError("bad OCDBT B+tree node")


def load_shard_sources(
    ocdbt_path: str, prefix: str = ""
) -> list[tuple[str, str, int, int] | tuple[str, bytes]]:
    zd = zstd.ZstdDecompressor()

    with open(f"{ocdbt_path}/manifest.ocdbt", "rb") as f:
        manif_header = f.read()

    if manif_header[:4] != b"\x0c\xdb\x3a\x2a":
        raise NotImplementedError("bad OCDBT manifest magic")
    if struct.unpack("<Q", manif_header[4:12])[0] != len(manif_header):
        raise NotImplementedError("bad OCDBT manifest size")
    if manif_header[12:14] != b"\0\1":
        raise NotImplementedError("OCDBT manifest is not version 0 zstd-compressed")

    manif = _decompress(zd, manif_header[14:-4])
    if manif[16] != 0:
        raise NotImplementedError("OCDBT manifest is not single")
    _max_inline_value_bytes, pos = _read_varint(manif, 17)
    _max_decoded_node_bytes, pos = _read_varint(manif, pos)
    _version_tree_arity_log2, pos = _read_varint(manif, pos)
    if manif[pos] != 1:
        raise NotImplementedError("OCDBT compression is not zstd")
    pos += 5

    manif_fnames, pos, _base_path_lens = _read_names(manif, pos, True)

    num_versions, pos = _read_varint(manif, pos)
    generations = []
    pos = _read_lists(manif, pos, num_versions, generations)
    root_heights = []
    for _i in range(num_versions):
        x, pos = _read_varint(manif, pos)
        root_heights.append(x)

    data_file_ids = []
    data_file_offsets = []
    data_file_lens = []
    nums_keys = []
    nums_tree_bytes = []
    nums_indirect_value_bytes = []
    pos = _read_lists(
        manif,
        pos,
        num_versions,
        data_file_ids,
        data_file_offsets,
        data_file_lens,
        nums_keys,
        nums_tree_bytes,
        nums_indirect_value_bytes,
    )
    pos += 8 * num_versions

    version = num_versions - 1
    shard_sources = []
    _load_node(
        zd,
        ocdbt_path,
        prefix,
        manif_fnames[data_file_ids[version]],
        data_file_offsets[version],
        data_file_lens[version],
        shard_sources,
    )
    return shard_sources
