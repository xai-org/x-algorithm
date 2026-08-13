# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import itertools
import os
from pathlib import Path

from xrex import settings
from xrex.configs import config_registry
from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.recsys.constants import continuous_action_type_map
from xrex.data.recsys.recsys_batch import CandidateNegativeFilter, CandidateNegativeMode
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.models.recsys_gen_recs_model import RecsysGenRecsModelConfig
from xrex.models.scaling import ScaleConfig
from xrex.models.transformer import FeedForwardConfig, RematType, TransformerConfig
from xrex.optimizers.optim import OptimConfig
from xrex.optimizers.schedule import ConstantSampleSchedule
from xrex.train.parallel_config import ParallelConfig
from xrex.train.trainer import CheckpointConfig
from xrex.train.trainer_gen_recs import GenRecsTrainer

PAD_TOKEN = 0
INPUT_VOCAB_K = 512
OUTPUT_VOCAB_K = 64
ACTION_TYPE_MAP_LEN = 60
CONTINUOUS_ACTION_TYPE_MAP_LEN = 8


def _round_up_to_multiple(n: int, k: int) -> int:
    return (n + k - 1) // k * k


DATASET_TYPES: list[str] = [
    "aggregated_kafka",
    "offline_kafka_dump_with_embeddings",
]


def _make_cfg(cfg, user_vocab_size: int, item_vocab_size: int, author_vocab_size: int):
    input_vocab_size = (
        user_vocab_size + item_vocab_size + author_vocab_size + ACTION_TYPE_MAP_LEN + 1
    )
    input_vocab_size = _round_up_to_multiple(input_vocab_size, INPUT_VOCAB_K)
    output_vocab_size = _round_up_to_multiple(ACTION_TYPE_MAP_LEN, OUTPUT_VOCAB_K)
    num_continuous_actions = _round_up_to_multiple(
        len(continuous_action_type_map), CONTINUOUS_ACTION_TYPE_MAP_LEN
    )

    cfg["user_vocab_size"] = user_vocab_size
    cfg["item_vocab_size"] = item_vocab_size
    cfg["author_vocab_size"] = author_vocab_size
    cfg["input_vocab_size"] = input_vocab_size
    cfg["output_vocab_size"] = output_vocab_size
    cfg["num_continuous_actions"] = num_continuous_actions

    cfg["user_hash_scales"] = [196_742_702, 1_852_108_266]
    cfg["user_biases"] = [193_5840_681, 16_7407_236]
    cfg["user_modulus"] = 2_859_568_897
    cfg["item_hash_scales"] = [2_161_410_491, 1_754_358_832]
    cfg["item_biases"] = [1_935_840_681, 167_407_236]
    cfg["item_modulus"] = 2_361_375_383
    cfg["author_hash_scales"] = [371_965_780, 328_930_218]
    cfg["author_biases"] = [139_686_260, 37_755_056]
    cfg["author_modulus"] = 631_860_353

    return cfg


def _make_dataset(
    mparams,
    dataset_type: str,
    hash_table: HashTable,
    config_name: str,
    num_global_negatives: int = 0,
    candidate_negative_filter=None,
    candidate_negative_mode=None,
) -> PhoenixDataset:
    del config_name
    factory = config_registry.GEN_RECS_DATASET_FACTORIES.get(dataset_type)
    if factory is not None:
        return factory(
            mparams,
            hash_table,
            {
                "pad_token": PAD_TOKEN,
                "num_global_negatives": num_global_negatives,
                "candidate_negative_filter": candidate_negative_filter,
                "candidate_negative_mode": candidate_negative_mode,
            },
        )

    match dataset_type:
        case "offline_kafka_dump_with_embeddings":
            offline_data_path = settings.GEN_RECS_OFFLINE_DATA_PATH
            offline_artifacts_dir = settings.GEN_RECS_OFFLINE_ARTIFACTS_DIR
            offline_date_range_str = os.environ.get("OFFLINE_DATE_RANGE")

            date_range: tuple[str, str] | None = None
            if offline_date_range_str:
                parts = offline_date_range_str.split(",", 1)
                date_range = (parts[0].strip(), parts[1].strip())

            offline_kwargs: dict = {}
            if num_global_negatives > 0:
                offline_kwargs["global_ids_file_path"] = Path(
                    os.path.join(offline_artifacts_dir, "global_ids.parquet")
                )

            return PhoenixDataset(
                hash_table=hash_table,
                path=offline_data_path,
                pad_token=PAD_TOKEN,
                history_seq_len=mparams["history_seq_len"],
                candidate_seq_len=mparams["candidate_seq_len"],
                input_vocab_size=mparams["input_vocab_size"],
                num_continuous_actions=mparams["num_continuous_actions"],
                num_kafka_partitions=mparams.get("num_kafka_partitions", 2048),
                output_vocab_size=mparams["output_vocab_size"],
                num_negatives_per_example=0,
                include_candidate_post_ids=True,
                multimodal_embedding_type="v5",
                offline_embedding_table_dir=offline_artifacts_dir,
                filter_candidates_require_embedding=True,
                date_range=date_range,
                num_global_negatives_per_example=num_global_negatives,
                candidate_negative_filter=candidate_negative_filter,
                candidate_negative_mode=candidate_negative_mode,
                **offline_kwargs,
            )
        case _:
            raise ValueError(f"Unknown {dataset_type=}, must be one of {DATASET_TYPES}")


MODEL_CFGS = {
    "xrecsys_gen_recs": _make_cfg(
        {
            "history_seq_len": 1023,
            "candidate_seq_len": 128,
            "num_layers": 8,
            "emb_size": 2560,
            "emb_table_width": 1024,
            "query_heads": 20,
            "kv_heads": 4,
            "base_batch_size": 32,
            "bs_per_device": 128,
            "ep": 512,
            "dp": 2,
            "total_samples": 1e11,
            "group_id": "gen_recs_xrecsys",
            "learning_rate": 2e-3,
            "use_product_surface": True,
            "mm_target_dim": 1024,
            "num_global_negatives": 128,
            "log_q_correction": True,
        },
        user_vocab_size=100_000_000,
        item_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
    ),
    "gen_recs_nano": _make_cfg(
        {
            "history_seq_len": 127,
            "candidate_seq_len": 64,
            "num_layers": 2,
            "emb_size": 256,
            "emb_table_width": 1024,
            "query_heads": 2,
            "kv_heads": 1,
            "base_batch_size": 32,
            "bs_per_device": 32,
            "ep": 1,
            "dp": 1,
            "total_samples": 1e9,
            "group_id": "gen_recs_nano",
            "learning_rate": 1e-3,
            "use_product_surface": True,
            "mm_target_dim": 1024,
            "num_global_negatives": 16,
            "log_q_correction": True,
            "attn_impl": "jax_attn",
            "num_kafka_partitions": 4,
            "max_posts": 8192,
        },
        user_vocab_size=16_384,
        item_vocab_size=65_536,
        author_vocab_size=4_096,
    ),
}


config_sweeps = {
    "config_name__mparams": MODEL_CFGS.items(),
    "dataset_type": DATASET_TYPES,
}
keys = list(config_sweeps.keys())
configs = [dict(zip(keys, values)) for values in itertools.product(*config_sweeps.values())]

CONFIGS: dict[str, GenRecsTrainer] = {}

for config in configs:
    config_name, mparams = config["config_name__mparams"]
    dataset_type = config["dataset_type"]
    config_name_gen = f"{config_name}_{dataset_type}"

    hash_table = HashTable(
        hash_keys=HashKeys(
            user_id_table_size=mparams["user_vocab_size"],
            user_hash_scales=mparams["user_hash_scales"],
            user_biases=mparams["user_biases"],
            user_modulus=mparams["user_modulus"],
            item_id_table_size=mparams["item_vocab_size"],
            item_hash_scales=mparams["item_hash_scales"],
            item_biases=mparams["item_biases"],
            item_modulus=mparams["item_modulus"],
            author_id_table_size=mparams["author_vocab_size"],
            author_hash_scales=mparams["author_hash_scales"],
            author_biases=mparams["author_biases"],
            author_modulus=mparams["author_modulus"],
        ),
        output_vocab_size=mparams["output_vocab_size"],
    )

    num_global_negatives = mparams.get("num_global_negatives", 0)
    neg_filter = (
        CandidateNegativeFilter.VQV_DWELL_10S
        if mparams.get("filter_impression_negatives", True)
        else CandidateNegativeFilter.NONE
    )
    neg_mode = (
        CandidateNegativeMode.FILTER
        if mparams.get("candidate_negative_mode") == "filter"
        else CandidateNegativeMode.PARTITION
    )
    dataset = _make_dataset(
        mparams, dataset_type, hash_table, config_name, num_global_negatives, neg_filter, neg_mode
    )

    assert mparams["emb_size"] % 128 == 0
    assert mparams["emb_table_width"] % 128 == 0
    hl = mparams["history_seq_len"]
    assert ((hl + 1) & (hl)) == 0, "history length must be of the form (2^n-1)"

    _default_eval_emb = settings.GEN_RECS_EVAL_POST_EMB_PATH
    if dataset_type == "offline_kafka_dump_with_embeddings":
        _offline_dir = os.environ.get("OFFLINE_ARTIFACTS_DIR")
        if _offline_dir:
            _default_eval_emb = os.path.join(_offline_dir, "inference_posts.parquet")
    _eval_post_emb_path = mparams.get("eval_post_emb_path", _default_eval_emb)

    CONFIGS[config_name_gen] = GenRecsTrainer(
        name=config_name_gen,
        precision_level=2,
        reuse_run_id=False,
        evals=[],
        eval_every_n=500,
        model_config=RecsysGenRecsModelConfig(
            use_product_surface=mparams["use_product_surface"],
            multimodal_embedding_type="v5",
            mm_target_dim=mparams["mm_target_dim"],
            candidate_embedding_mode=mparams.get("candidate_embedding_mode", "mm_only"),
            num_global_negatives=num_global_negatives,
            log_q_correction=mparams.get("log_q_correction", False),
            eval_post_emb_path=_eval_post_emb_path,
            max_posts=mparams.get("max_posts", 2_097_152),
            emb_table_width=mparams["emb_table_width"],
            history_seq_len=mparams["history_seq_len"],
            candidate_seq_len=mparams["candidate_seq_len"],
            pad_token=PAD_TOKEN,
            act_l2_weight=5e-7,
            num_continuous_actions=mparams["num_continuous_actions"],
            model_config=TransformerConfig(
                attn_config=RecsysAttentionConfig(
                    key_size=128,
                    num_q_heads=mparams["query_heads"],
                    num_kv_heads=mparams["kv_heads"],
                    attn_logit_cap=80.0,
                    max_num_segments_per_batch=1,
                    attn_impl=mparams.get("attn_impl", "jax_attn"),
                    fa_version="3",
                    qkv_merge=True,
                    attn_logit_cap_method="soft_sign",
                    rotary=True,
                    causal=True,
                    history_seq_len=mparams["history_seq_len"],
                ),
                ffn_config=FeedForwardConfig(
                    widening_factor=2,
                    gated_mlp=False,
                ),
                scale_config=ScaleConfig(),
                emb_size=mparams["emb_size"],
                num_layers=mparams["num_layers"],
                sequence_len=1 + mparams["history_seq_len"] + mparams["candidate_seq_len"],
                primer_norm=True,
                use_remat=True,
                remat_policy=RematType.WHOLE,
                use_layer_stack=True,
                output_vocab_size=mparams["output_vocab_size"],
            ),
            hash_table=hash_table,
        ),
        bs_per_device=mparams["bs_per_device"],
        dataset=dataset,
        parallel_config=ParallelConfig(
            ep=mparams["ep"],
            dp=mparams["dp"],
        ),
        optim_config=OptimConfig(
            optim="adam",
            weight_decay=1e-3,
            b1=0.95,
            b2=0.98,
        ),
        lr_schedule_in_samples_config=ConstantSampleSchedule(
            learning_rate=mparams["learning_rate"],
        ),
        max_steps=int(mparams["total_samples"] / mparams["base_batch_size"]) - 100,
        max_samples=None,
        checkpoint_config=CheckpointConfig(
            from_checkpoint=True,
            checkpoint_every_n=500,
            checkpoint_keep_every_nth=100,
            checkpoint_keep_last_n=10,
            save_final_checkpoint=True,
            verify_checksums=True,
            replica_axis="replica",
            checkpoint_chunked=False,
            checkpoint_compressed=False,
            copy_port=9988,
        ),
    )
