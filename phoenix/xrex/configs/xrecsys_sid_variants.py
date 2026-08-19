# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from xrex.configs import config_registry


def build_streaming_model_cfgs(make_trainer, make_dataset) -> dict:
    return {
        "xrecsys_sid_gen_rec_kafka": make_trainer(
            name="xrecsys_sid_gen_rec_kafka",
            dataset=make_dataset("aggregated_kafka", "xrecsys_sid_gen_rec"),
            attn_impl="jax_attn",
        ),
    }


config_registry.SID_MODEL_CFG_BUILDERS.append(build_streaming_model_cfgs)

assert "aggregated_kafka" in config_registry.SID_DATASET_FACTORIES
assert build_streaming_model_cfgs in config_registry.SID_MODEL_CFG_BUILDERS
