# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import difflib
import importlib
import logging
import traceback
from typing import Any

import xai_configlib
from xai_configlib import Config

from xrex import settings

logger = logging.getLogger(__name__)

PRESETS: dict[str, list[str]] = {}


PRESETS["quiet"] = [
    "enable_wandb=False",
    "checkpoint_every_n=999999",
]

PRESETS["no_timeouts"] = [
    "timeout.boot=999999",
    "timeout.train_start=999999",
    "timeout.first_step=999999",
    "timeout.step=999999",
    "timeout.checkpoint=999999",
]

PRESETS["no_health"] = [
    "allow_unschedulable_nodes=True",
]

PRESETS["no_checkpointing"] = [
    "checkpoint_every_n=0",
    "save_final_checkpoint=False",
    "from_checkpoint=False",
]


def get_config_module(mod_name: str) -> dict[str, Config]:
    try:
        return importlib.import_module(f"xrex.configs.{mod_name}")
    except ImportError as e:
        raise ValueError(
            f"No config module found {mod_name!r} under 'xrex.configs', "
            f"or {mod_name!r} has another import issue: {traceback.format_exc()}"
        ) from e


def replace_cli_subs(config: Any, kvs: list[str]) -> tuple[Any, dict[str, list[str]]]:
    return xai_configlib.replace_cli_subs(config, kvs, PRESETS)


def replace_and_validate_cli_subs(config: Any, kvs: list[str]) -> Any:
    config, used = xai_configlib.replace_cli_subs(config, kvs, PRESETS)
    for k, v in used.items():
        if not v:
            raise ValueError(f"Unused KV pairs: {k}")
        elif len(v) > 1:
            logger.warning(f"{k} matches multiple paths: {v}")
    return config


_DEFAULT_AOT_CACHE_DIR = "/tmp/jax_aot_cache/"


def _apply_deployment_defaults(config: Config) -> Config:
    if settings.AOT_CACHE_DIR and getattr(config, "aot_cache_dir", None) == (
        _DEFAULT_AOT_CACHE_DIR
    ):
        config.aot_cache_dir = settings.AOT_CACHE_DIR
    return config


def get_named_config(name: str) -> Config:
    if ":" in name:
        mod_name, ver = name.rsplit(":", 1)
    else:
        parts = name.rsplit(".", 1)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid config {name!r}, expected format 'module.key' where key cannot contain '.'"
            )
        mod_name, ver = parts

    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        depth = mod_name.count(".")
        if depth > 1:
            raise ValueError(f"No absolute config module found {mod_name!r}")

        try:
            mod = importlib.import_module(f"xrex.configs.{mod_name}")
        except ModuleNotFoundError:
            raise ValueError(
                f"No config module {mod_name!r} found, either at the top level or under 'xrex.configs'"
            )

    if not hasattr(mod, "CONFIGS"):
        raise ValueError("Your config needs a global variable named CONFIGS")

    configs = mod.CONFIGS
    if ver not in configs:
        logger.warning(
            f"Couldn't find a valid config ({len(configs)=}), searching for closest matches..."
        )
        closest_matches = difflib.get_close_matches(ver, configs, n=5, cutoff=0)
        raise ValueError(
            f"Config module {mod_name!r} has no key {ver!r}. Closest matches: {closest_matches}"
        )
    return _apply_deployment_defaults(configs[ver])
