# SPDX-License-Identifier: Apache-2.0
import typing

import pytest
from typing_extensions import TypeAliasType as BackportTypeAliasType
from xai_configlib import Config, cast_type, configclass, replace_cli_subs

TypeAliasType = getattr(typing, "TypeAliasType", BackportTypeAliasType)
RetryCount = TypeAliasType("RetryCount", int)
NestedRetryCount = TypeAliasType("NestedRetryCount", RetryCount)


@configclass
class RetryConfig(Config):
    retries: RetryCount = 3


@pytest.mark.parametrize("alias", [RetryCount, NestedRetryCount])
def test_cast_type_alias_preserves_underlying_value_type(alias):
    result = cast_type(alias, "7")

    assert result == 7
    assert type(result) is int


def test_type_alias_cli_override_round_trips():
    original = RetryConfig()

    updated, used = replace_cli_subs(original, ["retries=7"])

    assert original.retries == 3
    assert updated.retries == 7
    assert type(updated.retries) is int
    assert used == {"retries": ["retries"]}
    assert Config.from_json(updated.to_json()) == updated


def test_type_alias_rejects_invalid_cli_value():
    with pytest.raises(ValueError):
        replace_cli_subs(RetryConfig(), ["retries=invalid"])
