# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import dataclasses
import logging

from xai_configlib import Config

from xrex.configs.config_loader import replace_cli_subs
from xrex.driver.driver_local import LocalDriverConfig

logger = logging.getLogger(__name__)


def _sync_feature_prep(run_config: Config) -> None:
    model_config = getattr(run_config, "model_config", None)
    if model_config is None:
        return
    tower = getattr(model_config, "user_tower_config", model_config)
    if not getattr(tower, "feature_prep_enabled", False):
        return
    tower.feature_prep = dataclasses.replace(
        tower.feature_prep,
        emb_size=tower.model_config.emb_size,
        emb_table_width=tower.emb_table_width,
        scale_config=tower.model_config.scale_config,
        multimodal_embedding_dim=tower.multimodal_embedding_dim,
        search_query_embedding_dim=tower.search_query_embedding_dim,
        enable_user_embedding=tower.use_user_embedding,
        enable_post_embedding=tower.use_post_embedding,
        post_age_granularity_mins=tower.post_age_granularity_mins,
    )


def build_eval_configs(
    run_config: Config,
    overrides: list[str],
):
    driver_config = LocalDriverConfig()

    driver_config, used1 = replace_cli_subs(driver_config, overrides)
    run_config, used2 = replace_cli_subs(run_config, overrides)

    for k, v in used2.items():
        used1.setdefault(k, []).extend(v)

    for k, v in used1.items():
        if len(v) == 0:
            logger.error(f"Unused argument pairs: {k}")
            exit(1)
        elif len(v) > 1:
            logger.warning(f"Argument {k} is matches multiple paths: {v}")

    _sync_feature_prep(run_config)

    return driver_config, run_config
