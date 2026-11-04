# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import dataclasses

from xai_proto import recsys_pb2
from xrex import settings
from xrex.configs import xrecsys
from xrex.models.recsys_model import ContinuousActionLossConfig, NormConfig
from xrex.models.recsys_reply_two_tower_v2_model import RecsysReplyTwoTowerV2Config
from xrex.train.trainer_recsys import RecsysTrainer

MODEL_CFGS = {
    "reply_two_tower_v2_main": xrecsys._make_cfg(
        {
            "history_seq_len": 127,
            "candidate_seq_len": 64,
            "num_layers": 1,
            "emb_size": 256,
            "emb_table_width": 256,
            "query_heads": 1,
            "kv_heads": 1,
            "base_batch_size": 128,
            "bs_per_device": 128,
            "tp": 1,
            "ep": 16,
            "fsdp": 1,
            "dp": 1,
            "total_samples": 1e11,
            "learning_rate": 1e-4,
            "attn_impl": "pallas_ranker_attn",
            "log_q_correction": False,
            "use_product_surface": True,
            "num_negatives_per_example": 0,
            "metric_group": "reply",
            "use_post_sid": True,
            "sid_num_levels": 16,
            "sid_decoder_path": settings.SID_DECODER_PATH,
            "dwell_loss_weight": 0.02,
            "negfb_loss_weight": 10.0,
        },
        user_vocab_size=20_000_000,
        item_vocab_size=20_000_000,
        author_vocab_size=5_000_000,
    ),
}


def _build(
    config_name: str, mparams: dict, dataset_type: str = "aggregated_kafka"
) -> RecsysTrainer:
    trainer = xrecsys.make_trainer(config_name, mparams, dataset_type)
    base = trainer.model_config
    model_config = RecsysReplyTwoTowerV2Config(
        **{
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init},
            "negfb_loss_weight": mparams["negfb_loss_weight"],
            "continuous_action_losses": [
                ContinuousActionLossConfig(
                    action_index=recsys_pb2.ContinuousActionName.DWELL_TIME,
                    metric_name="dwell",
                    loss_weight=mparams["dwell_loss_weight"],
                    loss_type="tweedie",
                    tweedie_power=1.5,
                    norm_config=NormConfig(norm_scale=300.0, use_log=False),
                    mask_negatives=False,
                )
            ],
        }
    )
    return dataclasses.replace(
        trainer,
        model_config=model_config,
        smoothing_windows=[],
        dataset=dataclasses.replace(trainer.dataset, reply_parent_mode=True),
    )


CONFIGS = {
    f"{name}_{dataset_type}": _build(name, mparams, dataset_type)
    for name, mparams in MODEL_CFGS.items()
    for dataset_type in ("aggregated_kafka", "rust_kafka")
}
