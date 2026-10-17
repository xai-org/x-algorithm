# SPDX-License-Identifier: Apache-2.0
from typing import Tuple  # noqa: UP035 - exercise legacy annotation support

import pytest
from xai_configlib import Config, cast_type, configclass, replace_cli_subs


@configclass
class ConversionConfig(Config):
    conversion_label_types: tuple[str, ...] = ()


@pytest.mark.parametrize("annotation", [tuple[int, ...], Tuple[int, ...]])  # noqa: UP006
@pytest.mark.parametrize(
    "raw, expected", [("", ()), ("7", (7,)), ("7,11", (7, 11)), ("7,11,13", (7, 11, 13))]
)
def test_variadic_tuple_casts_each_element(annotation, raw, expected):
    assert cast_type(annotation, raw) == expected


def test_variadic_tuple_override_updates_config_and_reports_usage():
    original = ConversionConfig()

    updated, used = replace_cli_subs(original, ["conversion_label_types=click,follow,share"])

    assert updated.conversion_label_types == ("click", "follow", "share")
    assert original.conversion_label_types == ()
    assert used == {"conversion_label_types": ["conversion_label_types"]}


def test_variadic_tuple_rejects_invalid_element():
    with pytest.raises(ValueError):
        cast_type(tuple[int, ...], "7,invalid")


def test_fixed_tuple_keeps_heterogeneous_types():
    assert cast_type(tuple[int, str, bool], "7,ready,false") == (7, "ready", False)


def test_fixed_tuple_still_rejects_wrong_length():
    with pytest.raises(TypeError, match="different number of arguments"):
        cast_type(tuple[int, str], "7,ready,extra")
