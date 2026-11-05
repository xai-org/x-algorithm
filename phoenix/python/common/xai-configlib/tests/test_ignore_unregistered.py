# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
from xai_configlib import Config, configclass, from_dict


@configclass
class PluginOptions(Config):
    retries: int = 3


@configclass
class PluginBundle(Config):
    plugins: dict[str, Config | None] = dataclasses.field(default_factory=dict)


@pytest.mark.parametrize("wrap", [lambda x: x, lambda x: [x], lambda x: (x,)])
def test_ignore_flag_reaches_unknown_config_inside_plain_dict(wrap):
    value = wrap({"plugins": {"missing": {"__class": "missing_plugin.Options"}}})

    assert from_dict(value, ignore_unregistered_cls=True) == wrap({"plugins": {"missing": None}})


def test_ignore_flag_preserves_known_configs_in_bundle():
    bundle = PluginBundle(plugins={"known": PluginOptions(8)})
    payload = bundle.to_dict()
    payload["plugins"]["missing"] = {"__class": "missing_plugin.Options"}

    restored = Config.from_dict(payload, ignore_unregistered_cls=True)

    assert restored == PluginBundle(plugins={"known": PluginOptions(8), "missing": None})


def test_strict_mode_still_rejects_unknown_nested_config():
    with pytest.raises(ValueError, match="missing_plugin.Options is not registered"):
        from_dict({"plugins": {"missing": {"__class": "missing_plugin.Options"}}})
