# SPDX-License-Identifier: Apache-2.0
import dataclasses
from typing import Any

import pytest
from xai_configlib import Config, configclass, replace_cli_subs


@configclass
class ContainerConfig(Config):
    smoothing_windows: list[int] = dataclasses.field(default_factory=lambda: [32, 128])
    weights: dict[str, float] = dataclasses.field(default_factory=lambda: {"click": 0.25})
    flags: list[bool] = dataclasses.field(default_factory=lambda: [True])
    optional_windows: list[int] | None = dataclasses.field(default_factory=lambda: [4])
    numeric_keys: dict[int, int] = dataclasses.field(default_factory=lambda: {3: 7})
    untyped: list[Any] = dataclasses.field(default_factory=lambda: [0])


@pytest.mark.parametrize(
    "override, field, key, expected",
    [
        ("smoothing_windows.1=256", "smoothing_windows", 1, 256),
        ("weights.click=0.5", "weights", "click", 0.5),
        ("flags.0=false", "flags", 0, False),
        ("optional_windows.0=8", "optional_windows", 0, 8),
        ("numeric_keys.3=9", "numeric_keys", 3, 9),
    ],
)
def test_indexed_override_casts_declared_element_type(override, field, key, expected):
    original = ContainerConfig()

    updated, used = replace_cli_subs(original, [override])

    result = getattr(updated, field)[key]
    assert result == expected
    assert type(result) is type(expected)
    assert used == {override.split("=", 1)[0]: [f"{field}.{key}"]}
    assert original == ContainerConfig()


@pytest.mark.parametrize("override", ["smoothing_windows.0=invalid", "flags.0=invalid"])
def test_invalid_typed_element_is_rejected(override):
    with pytest.raises(ValueError):
        replace_cli_subs(ContainerConfig(), [override])


def test_any_element_keeps_string_override_behavior():
    updated, used = replace_cli_subs(ContainerConfig(), ["untyped.0=raw"])

    assert updated.untyped == ["raw"]
    assert used == {"untyped.0": ["untyped.0"]}


def test_whole_container_override_keeps_typed_behavior():
    updated, used = replace_cli_subs(ContainerConfig(), ["smoothing_windows=16,64"])

    assert updated.smoothing_windows == [16, 64]
    assert used == {"smoothing_windows": ["smoothing_windows"]}
