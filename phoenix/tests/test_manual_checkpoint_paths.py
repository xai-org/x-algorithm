# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xrex.utils.metadata import FileSystemProvider, LoadType, write_metadata_file


def completed_checkpoint(root, samples, run_id):
    path = root / f"elapsed_samples_{samples:018d}" / run_id
    (path / "orbax-ckpt").mkdir(parents=True)
    write_metadata_file(path, samples, 3, "2030-01-01T00:00:00Z", samples * 64, 5)
    return path


@pytest.mark.parametrize("ancestor", ["normal", "elapsed_samples_999"])
def test_manual_run_path_uses_its_own_names(tmp_path, ancestor):
    path = completed_checkpoint(tmp_path / ancestor / "model", 128, "run-a_1")

    checkpoint = FileSystemProvider(tmp_path / "provider").discover_checkpoint(
        None, load=str(path)
    )

    assert checkpoint is not None
    assert checkpoint.run_id == "run-a_1"
    assert checkpoint.elapsed_samples == 128
    assert checkpoint.path == str(path.resolve())
    assert checkpoint.load_type == LoadType.MANUAL_LOAD
    assert checkpoint.checkpoint_index == 3
    assert checkpoint.elapsed_tokens == 8192


def test_manual_sample_directory_selects_a_completed_child(tmp_path):
    root = tmp_path / "model"
    completed_checkpoint(root, 128, "run-a_0")
    expected = completed_checkpoint(root, 128, "run-b_0")
    incomplete = completed_checkpoint(root, 128, "run-z_0")
    (incomplete / "orbax-ckpt").rename(incomplete / "orbax-ckpt.orbax-checkpoint-tmp-1")

    checkpoint = FileSystemProvider(tmp_path / "provider").discover_checkpoint(
        None, load=str(expected.parent)
    )

    assert checkpoint is not None
    assert checkpoint.run_id == "run-b_0"
    assert checkpoint.elapsed_samples == 128
    assert checkpoint.path == str(expected.resolve())


def test_missing_manual_path_is_not_discovered(tmp_path):
    assert FileSystemProvider(tmp_path / "provider").discover_checkpoint(
        None, load=str(tmp_path / "missing")
    ) is None


def test_automatic_discovery_still_selects_largest_sample_count(tmp_path):
    root = tmp_path / "model"
    completed_checkpoint(root, 64, "run-z_0")
    expected = completed_checkpoint(root, 128, "run-a_0")

    checkpoint = FileSystemProvider(tmp_path).discover_checkpoint("model")

    assert checkpoint is not None
    assert checkpoint.run_id == "run-a_0"
    assert checkpoint.elapsed_samples == 128
    assert checkpoint.path == str(expected)
    assert checkpoint.load_type == LoadType.RESTART
