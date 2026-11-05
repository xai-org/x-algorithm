# SPDX-License-Identifier: Apache-2.0
import pytest
from xai_configlib import CONFIG_REGISTRY, Config, configclass, from_dict


@configclass
class RequiredConfig(Config):
    value: int = 7


def test_type_requirement_accepts_serialized_class_tag():
    config = RequiredConfig(value=11)

    assert Config.from_dict(config.to_dict(), ensure_class=RequiredConfig) == config
    assert Config.from_json(config.to_json(), ensure_class=RequiredConfig) == config


def test_type_requirement_accepts_legacy_class_tag():
    assert from_dict(
        {"__class": "RequiredConfig", "value": 11}, ensure_class=RequiredConfig
    ) == RequiredConfig(11)


def test_string_requirement_keeps_existing_behavior():
    assert from_dict({"value": 11}, ensure_class="RequiredConfig") == RequiredConfig(11)


def test_wrong_class_is_rejected():
    with pytest.raises(AssertionError, match="Inconsistent config class requirement"):
        from_dict(RequiredConfig().to_dict(), ensure_class=Config)


def test_type_requirement_uses_requested_class_when_names_collide():
    first = type(
        "SharedConfig",
        (Config,),
        {"__module__": "configlib_test_first", "__annotations__": {"value": int}, "value": 1},
    )
    second = type(
        "SharedConfig",
        (Config,),
        {"__module__": "configlib_test_second", "__annotations__": {"value": int}, "value": 2},
    )
    first = configclass(first)
    second = configclass(second)
    try:
        restored = from_dict({"value": 19}, ensure_class=second)

        assert type(restored) is second
        assert restored.value == 19
        assert from_dict(second(19).to_dict(), ensure_class=second) == second(19)
        with pytest.raises(AssertionError, match="Inconsistent config class requirement"):
            from_dict({"__class": "SharedConfig", "value": 19}, ensure_class=second)
        assert from_dict({"__class": "SharedConfig", "value": 19}, ensure_class=first) == first(19)
    finally:
        CONFIG_REGISTRY.pop("configlib_test_first.SharedConfig")
        CONFIG_REGISTRY.pop("configlib_test_second.SharedConfig")
        CONFIG_REGISTRY.pop("SharedConfig")
