# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import importlib
import importlib.util
import itertools
from dataclasses import replace
from pathlib import Path

from xai_proto import recsys_pb2

from xrex.configs import config_registry, data_feeds
from xrex.data.parquet_recsys import (
    PhoenixDataset,
)
from xrex.data.recsys.constants import continuous_action_type_map
from xrex.data.recsys.feature_config import CategoricalFeature
from xrex.data.recsys.recsys_batch import EMBEDDING_CONFIG
from xrex.data.recsys.sequence_packing import BetaLengthDistribution
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.models.recsys_feature_prep import FeaturePrepConfig
from xrex.models.recsys_model import (
    CategoricalFeatureConfig,
    ContextFeaturesConfig,
    RecsysAggregatedModelConfig,
    UserFeaturesConfig,
)
from xrex.models.recsys_two_tower_model import (
    RecsysCandidateModelConfig,
    RecsysTwoTowerModelConfig,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.transformer import FeedForwardConfig, RematType, TransformerConfig
from xrex.optimizers.optim import OptimConfig
from xrex.optimizers.recsys.config import RecsysEmbeddingOptimConfig
from xrex.optimizers.recsys.rowwise_adagrad import RecsysRowwiseAdagradConfig
from xrex.optimizers.schedule import ConstantSampleSchedule
from xrex.train.parallel_config import ParallelConfig
from xrex.train.trainer import CheckpointConfig
from xrex.train.trainer_recsys import RecsysTrainer

PAD_TOKEN = 0
INPUT_VOCAB_K = 512
OUTPUT_VOCAB_K = 64
ACTION_TYPE_MAP_LEN = 60
CONTINUOUS_ACTION_TYPE_MAP_LEN = 8


def _round_up_to_multiple(n: int, k: int) -> int:
    return (n + k - 1) // k * k


def _has_user_features_token(mparams: dict) -> bool:
    return (
        mparams.get("enable_user_country_feature", False)
        or mparams.get("enable_user_language_feature", False)
        or mparams.get("enable_user_state_feature", False)
        or mparams.get("enable_user_dma_code_feature", False)
        or mparams.get("enable_user_location_feature", False)
        or mparams.get("enable_user_installed_apps", False)
        or mparams.get("enable_user_gender_feature", False)
        or mparams.get("enable_user_age_feature", False)
    )


def _default_recsys_scaling() -> ScaleConfig:
    return ScaleConfig()


def _make_feature_prep_config(mparams: dict, scale_config: ScaleConfig) -> FeaturePrepConfig:
    multimodal_type = mparams.get("multimodal_embedding_type")
    multimodal_dim = EMBEDDING_CONFIG[multimodal_type][1] if multimodal_type is not None else 0
    post_age_mins = mparams.get("post_age_granularity_mins", 60)
    use_post_emb = mparams.get("use_post_embedding", True)
    use_user_emb = mparams.get("use_user_embedding", True)
    structural = dict(
        emb_size=mparams["emb_size"],
        emb_table_width=mparams["emb_table_width"],
        scale_config=scale_config,
        multimodal_embedding_dim=multimodal_dim,
        search_query_embedding_dim=mparams.get("search_query_embedding_dim", 0),
        enable_user_embedding=use_user_emb,
        enable_post_embedding=use_post_emb,
        post_age_granularity_mins=post_age_mins,
    )

    if "feature_prep" in mparams:
        return replace(mparams["feature_prep"], **structural)

    enable_ctx = mparams.get("enable_context_features", True)
    use_post_sid = mparams.get("use_post_sid", False)
    return FeaturePrepConfig(
        **structural,
        enable_ip_address=mparams.get("ip_vocab_size", 0) > 0 and use_user_emb,
        enable_user_country=mparams.get("enable_user_country_feature", False),
        enable_user_language=mparams.get("enable_user_language_feature", False),
        enable_user_state=mparams.get("enable_user_state_feature", False),
        enable_user_dma_code=mparams.get("enable_user_dma_code_feature", False),
        enable_user_location=mparams.get("enable_user_location_feature", False),
        enable_user_gender=mparams.get("enable_user_gender_feature", False),
        enable_user_age=mparams.get("enable_user_age_feature", False),
        enable_user_installed_apps=mparams.get("enable_user_installed_apps", False),
        enable_post_sid=use_post_sid,
        sid_embed_dim=mparams.get("sid_embed_dim", 1024),
        sid_num_levels=mparams.get("sid_num_levels", 6),
        sid_codebook_size=mparams.get("sid_codebook_size", 1024),
        sid_hash_level=mparams.get("sid_hash_level", False),
        sid_cross_attn=mparams.get("sid_cross_attn", False),
        enable_product_surface=mparams.get("use_product_surface", False),
        enable_post_age=False,
        enable_timezone=enable_ctx,
        enable_dwell_time=mparams.get("enable_dwell_time", True),
        enable_time_of_day=False,
        enable_time_of_week=False,
        enable_hour_of_day=enable_ctx,
        hour_of_day_dither_fraction=0.0,
        enable_day_of_week=enable_ctx,
    )


def _num_user_prefix_tokens(
    mparams: dict, use_user_embedding: bool, scale_config: ScaleConfig
) -> int:
    if mparams.get("feature_prep_enabled", False):
        fp = _make_feature_prep_config(mparams, scale_config)
        return int(fp.enable_user_embedding) + int(fp.has_user_features)
    return int(use_user_embedding) + int(_has_user_features_token(mparams))


DATASET_TYPES: list[str] = [
    "aggregated_kafka",
]
DATASET_TYPES += config_registry.RETRIEVAL_EXTRA_DATASET_TYPES


def _resolve_global_ids_file_path(path: Path | None) -> Path | None:
    for _resolve in config_registry.GLOBAL_IDS_RESOLVERS:
        path = _resolve(path)
    return path


def _make_cfg(
    cfg,
    user_vocab_size: int,
    item_vocab_size: int,
    author_vocab_size: int,
    ip_vocab_size: int = 0,
    item_hash_vocab_size: int = 0,
):
    input_vocab_size = (
        user_vocab_size
        + item_vocab_size
        + author_vocab_size
        + ip_vocab_size
        + ACTION_TYPE_MAP_LEN
        + 1
    )
    input_vocab_size = _round_up_to_multiple(input_vocab_size, INPUT_VOCAB_K)

    item_hash_vocab_size = max(item_hash_vocab_size, item_vocab_size)
    hash_vocab_size = _round_up_to_multiple(
        user_vocab_size
        + item_hash_vocab_size
        + author_vocab_size
        + ip_vocab_size
        + ACTION_TYPE_MAP_LEN
        + 1,
        INPUT_VOCAB_K,
    )
    output_vocab_size = _round_up_to_multiple(ACTION_TYPE_MAP_LEN, OUTPUT_VOCAB_K)
    num_continuous_actions = _round_up_to_multiple(
        len(continuous_action_type_map), CONTINUOUS_ACTION_TYPE_MAP_LEN
    )

    cfg["user_vocab_size"] = user_vocab_size
    cfg["item_vocab_size"] = item_vocab_size
    cfg["item_hash_vocab_size"] = item_hash_vocab_size
    cfg["author_vocab_size"] = author_vocab_size
    cfg["ip_vocab_size"] = ip_vocab_size

    cfg["input_vocab_size"] = input_vocab_size
    cfg["hash_vocab_size"] = hash_vocab_size
    cfg["output_vocab_size"] = cfg.get("output_vocab_size", output_vocab_size)
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
    cfg["ip_hash_scales"] = [529_482_163, 1_327_604_891]
    cfg["ip_biases"] = [742_961_053, 318_205_477]
    cfg["ip_modulus"] = 1_073_741_789

    return cfg


def _make_dataset(
    mparams,
    dataset_type: str,
    hash_table: HashTable,
    config_name: str,
) -> PhoenixDataset:
    factory = config_registry.RETRIEVAL_DATASET_FACTORIES.get(dataset_type)
    if factory is not None:
        _gif = mparams.get("global_ids_file_path", None)
        return factory(
            mparams,
            hash_table,
            mparams.get("use_post_sid", False),
            mparams.get("sid_num_levels", 6),
            _resolve_global_ids_file_path(Path(_gif) if _gif is not None else None),
            config_name,
        )
    del config_name

    _use_post_sid = mparams.get("use_post_sid", False)
    _sid_num_levels = mparams.get("sid_num_levels", 6)
    _global_ids_file_path = mparams.get("global_ids_file_path", None)
    _global_ids_file_path = _resolve_global_ids_file_path(
        Path(_global_ids_file_path) if _global_ids_file_path is not None else None
    )

    match dataset_type:
        case "aggregated_kafka":
            return PhoenixDataset(
                hash_table=hash_table,
                path="/path/to/offline_kafka_dump",
                history_seq_len=mparams["history_seq_len"],
                candidate_seq_len=mparams["candidate_seq_len"],
                input_vocab_size=mparams["input_vocab_size"],
                hash_vocab_size=mparams["hash_vocab_size"],
                num_continuous_actions=mparams["num_continuous_actions"],
                num_negatives_per_example=mparams["num_negatives_per_example"],
                num_global_negatives_per_example=mparams["num_global_negatives_per_example"],
                num_kafka_partitions=1024,
                include_candidate_post_ids=True,
                date_range=mparams.get("date_range", None),
                global_ids_file_path=_global_ids_file_path,
                use_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
            )
        case _:
            raise ValueError(f"Uknown {dataset_type=}, must be one of {DATASET_TYPES}")


def _xrecsys_two_tower_combined_base() -> dict:
    return {
        "history_seq_len": 1023,
        "enable_user_country_feature": True,
        "enable_user_language_feature": True,
        "enable_user_location_feature": True,
        "enable_user_gender_feature": True,
        "enable_user_age_feature": True,
        "enable_user_installed_apps": True,
        "candidate_seq_len": 64,
        "num_negatives_per_example": 0,
        "num_global_negatives_per_example": 64,
        "num_layers": 8,
        "emb_size": 1024,
        "emb_table_width": 1024,
        "query_heads": 16,
        "kv_heads": 4,
        "base_batch_size": 32,
        "dp": 1,
        "total_samples": 1e11,
        "empty_history_user_dropout_rate": 0.1,
        "learning_rate": 2e-3,
        "emb_learning_rate": 0.1,
        "qk_norm": False,
        "attn_logit_cap": 80.0,
        "primer_norm": True,
        "feature_prep_enabled": True,
        "enable_candidate_tower_linear_proj": False,
        "apply_u2u_and_i2i_loss": False,
        "positive_actions": [
            recsys_pb2.ActionName.SERVER_TWEET_FAV,
        ],
        "immersive_positive_actions": [
            recsys_pb2.ActionName.SERVER_TWEET_FAV,
            recsys_pb2.ActionName.SERVER_TWEET_REPLY,
            recsys_pb2.ActionName.SERVER_TWEET_QUOTE,
            recsys_pb2.ActionName.SERVER_TWEET_RETWEET,
            recsys_pb2.ActionName.CLIENT_TWEET_VIDEO_QUALITY_VIEW,
            recsys_pb2.ActionName.CLIENT_TWEET_FOLLOW_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_BOOKMARK,
            recsys_pb2.ActionName.CLIENT_TWEET_SHARE,
        ],
        "num_candidate_heads": 2,
        "head_names": ["home", "immersive"],
        "head_dataset_mapping": {
            "HOME": 0,
            "IMMERSIVE4Day": 1,
            "IMMERSIVE2Day": 1,
            "IMMERSIVENSFW": 1,
        },
        "checkpoint_dataset_names": [
            "HOME",
            "IMMERSIVE4Day",
            "IMMERSIVE2Day",
            "IMMERSIVENSFW",
        ],
        "max_posts": 28_672_000,
        "use_user_embedding": False,
        "use_post_embedding": False,
        "use_post_sid": True,
        "sid_embed_dim": 1024,
        "sid_num_levels": 6,
        "sid_codebook_size": 256,
        "sid_hash_level": True,
        "sid_cross_attn": True,
        "global_ids_file_path": data_feeds.SID_GLOBAL_IDS_PLACEHOLDER,
        "use_seqpack": True,
        "right_anchored_rope": True,
        "effective_sequence_len": 513,
        "seqpack_distribution": BetaLengthDistribution(
            min_len=0,
            max_len=1023,
            mean_len=511,
            alpha=1.0,
            beta=1.0,
            block_size=128,
        ),
    }


_H100_OVERRIDES = {
    "bs_per_device": 480,
    "ep": 128,
    "attn_impl": "pallas_ranker_varlen_attn",
}

_GB300_OVERRIDES = {
    "bs_per_device": 768,
    "ep": 32,
    "attn_impl": "cutedsl_ranker_varlen_attn",
    "remat_policy": RematType.SAVE_GB300_RECSYS,
    "unroll_layer_stack": True,
}


MODEL_CFGS = {
    "xrecsys_two_tower": _make_cfg(
        {
            "history_seq_len": 1023,
            "enable_user_country_feature": True,
            "enable_user_language_feature": True,
            "enable_user_location_feature": False,
            "candidate_seq_len": 64,
            "num_negatives_per_example": 0,
            "num_global_negatives_per_example": 64,
            "num_layers": 8,
            "emb_size": 1024,
            "emb_table_width": 1024,
            "query_heads": 16,
            "kv_heads": 4,
            "base_batch_size": 32,
            "bs_per_device": 480,
            "ep": 16 * 8,
            "dp": 1,
            "total_samples": 1e11,
            "learning_rate": 2e-3,
            "attn_impl": "pallas_ranker_varlen_attn",
            "enable_candidate_tower_linear_proj": True,
            "apply_u2u_and_i2i_loss": False,
            "checkpoint_dataset_names": ["HOME"],
            "use_user_embedding": False,
            "use_post_embedding": False,
            "use_post_sid": True,
            "sid_embed_dim": 1024,
            "sid_num_levels": 6,
            "sid_codebook_size": 256,
            "sid_hash_level": True,
            "sid_cross_attn": True,
            "global_ids_file_path": data_feeds.SID_GLOBAL_IDS_PLACEHOLDER,
            "use_seqpack": True,
            "right_anchored_rope": True,
            "effective_sequence_len": 513,
            "seqpack_distribution": BetaLengthDistribution(
                min_len=0,
                max_len=1023,
                mean_len=511,
                alpha=1.0,
                beta=1.0,
                block_size=128,
            ),
        },
        user_vocab_size=0,
        item_vocab_size=0,
        item_hash_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
        ip_vocab_size=0,
    ),
    "xrecsys_two_tower_combined": _make_cfg(
        {**_xrecsys_two_tower_combined_base(), **_H100_OVERRIDES},
        user_vocab_size=0,
        item_vocab_size=0,
        item_hash_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
    ),
    "xrecsys_two_tower_combined_gb300": _make_cfg(
        {**_xrecsys_two_tower_combined_base(), **_GB300_OVERRIDES},
        user_vocab_size=0,
        item_vocab_size=0,
        item_hash_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
    ),
    "xrecsys_two_tower_nano": _make_cfg(
        {
            "history_seq_len": 1022,
            "candidate_seq_len": 64,
            "num_negatives_per_example": 0,
            "num_global_negatives_per_example": 64,
            "num_layers": 4,
            "emb_size": 512,
            "emb_table_width": 512,
            "query_heads": 4,
            "kv_heads": 2,
            "base_batch_size": 32,
            "bs_per_device": 64,
            "max_posts": 65_536,
            "ep": 1,
            "dp": 1,
            "total_samples": 1e11,
            "learning_rate": 2e-3,
            "attn_impl": "pallas_ranker_attn",
            "enable_candidate_tower_linear_proj": True,
            "apply_u2u_and_i2i_loss": False,
            "checkpoint_dataset_names": ["HOME"],
            "use_user_embedding": False,
            "use_post_embedding": False,
            "use_post_sid": True,
            "sid_embed_dim": 512,
            "sid_num_levels": 6,
            "sid_codebook_size": 256,
            "sid_hash_level": True,
            "sid_cross_attn": True,
        },
        user_vocab_size=0,
        item_vocab_size=0,
        item_hash_vocab_size=100_000,
        author_vocab_size=30_000,
    ),
}

for _preset_module in ("xrex.configs.xrecsys_two_tower_ads_retrieval",):
    if importlib.util.find_spec(_preset_module) is not None:
        importlib.import_module(_preset_module)

for _build_model_cfgs in config_registry.RETRIEVAL_MODEL_CFG_BUILDERS:
    MODEL_CFGS.update(_build_model_cfgs(_make_cfg))


config_sweeps = {
    "config_name__mparams": MODEL_CFGS.items(),
    "dataset_type": DATASET_TYPES,
}
keys = list(config_sweeps.keys())
configs = [dict(zip(keys, values)) for values in itertools.product(*config_sweeps.values())]

CONFIGS: dict[str, RecsysTrainer] = {}


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
            item_hash_vocab_size=mparams["item_hash_vocab_size"],
            item_hash_scales=mparams["item_hash_scales"],
            item_biases=mparams["item_biases"],
            item_modulus=mparams["item_modulus"],
            author_id_table_size=mparams["author_vocab_size"],
            author_hash_scales=mparams["author_hash_scales"],
            author_biases=mparams["author_biases"],
            author_modulus=mparams["author_modulus"],
            ip_id_table_size=mparams["ip_vocab_size"],
            ip_hash_scales=mparams["ip_hash_scales"],
            ip_biases=mparams["ip_biases"],
            ip_modulus=mparams["ip_modulus"],
        ),
        output_vocab_size=mparams["output_vocab_size"],
    )

    use_ip_address = mparams.get("ip_vocab_size", 0) > 0
    use_user_features = _has_user_features_token(mparams)

    dataset = _make_dataset(mparams, dataset_type, hash_table, config_name)
    evals = []
    for _build_evals in config_registry.RETRIEVAL_EVAL_BUILDERS:
        evals += _build_evals(mparams, dataset, _has_user_features_token)

    raw_checkpoint_datasets = mparams.get("checkpoint_dataset_names", None)
    if isinstance(raw_checkpoint_datasets, str):
        checkpoint_dataset_names = [
            s.strip() for s in raw_checkpoint_datasets.split(",") if s.strip()
        ]
    else:
        checkpoint_dataset_names = raw_checkpoint_datasets

    if mparams.get("enable_candidate_tower_linear_proj") and mparams.get(
        "feature_prep_enabled", False
    ):
        raise ValueError(
            "enable_candidate_tower_linear_proj and feature_prep_enabled "
            "(candidate project-then-sum) are mutually exclusive; enable at most one "
            "(or neither for mean-pool on the candidate tower)."
        )

    assert mparams["emb_size"] % 128 == 0
    assert mparams["emb_table_width"] % 128 == 0
    hl = mparams["history_seq_len"]
    use_user_embedding = mparams.get("use_user_embedding", True)
    scale_config = _default_recsys_scaling()
    num_user_prefix_tokens = _num_user_prefix_tokens(mparams, use_user_embedding, scale_config)
    total_seq = num_user_prefix_tokens + hl
    use_seqpack = mparams.get("use_seqpack", False)

    if num_user_prefix_tokens > 1:
        assert (total_seq & (total_seq - 1)) == 0, (
            f"total sequence length ({total_seq} = {num_user_prefix_tokens} + {hl}) must be a power of 2"
        )
    elif num_user_prefix_tokens == 1:
        assert ((hl + 1) & (hl)) == 0, "history length must be of the form (2^n-1)"

    positive_actions = mparams.get(
        "positive_actions", RecsysTwoTowerModelConfig.get_positive_actions()
    )
    hard_negative_actions = mparams.get(
        "hard_negative_actions", RecsysTwoTowerModelConfig.get_hard_negative_actions()
    )
    soft_negative_actions = mparams.get(
        "soft_negative_actions", RecsysTwoTowerModelConfig.get_soft_negative_actions()
    )
    user_features_config = UserFeaturesConfig(
        enable_user_country_feature=mparams.get("enable_user_country_feature", False),
        enable_user_language_feature=mparams.get("enable_user_language_feature", False),
        enable_user_state_feature=mparams.get("enable_user_state_feature", False),
        enable_user_dma_code_feature=mparams.get("enable_user_dma_code_feature", False),
        enable_user_location_feature=mparams.get("enable_user_location_feature", False),
        enable_user_installed_apps=mparams.get("enable_user_installed_apps", False),
        enable_user_gender_feature=mparams.get("enable_user_gender_feature", False),
        enable_user_age_feature=mparams.get("enable_user_age_feature", False),
        country_emb_dim=mparams.get("country_emb_dim", UserFeaturesConfig.country_emb_dim),
        language_emb_dim=mparams.get("language_emb_dim", UserFeaturesConfig.language_emb_dim),
        num_states=mparams.get("num_states", UserFeaturesConfig.num_states),
        state_emb_dim=mparams.get("state_emb_dim", UserFeaturesConfig.state_emb_dim),
        num_dma_codes=mparams.get("num_dma_codes", UserFeaturesConfig.num_dma_codes),
        dma_code_emb_dim=mparams.get("dma_code_emb_dim", UserFeaturesConfig.dma_code_emb_dim),
        location_emb_dim=mparams.get("location_emb_dim", UserFeaturesConfig.location_emb_dim),
        installed_apps_emb_dim=mparams.get(
            "installed_apps_emb_dim", UserFeaturesConfig.installed_apps_emb_dim
        ),
        gender_score_proj_dim=mparams.get(
            "gender_score_proj_dim", UserFeaturesConfig.gender_score_proj_dim
        ),
        age_bracket_emb_dim=mparams.get(
            "age_bracket_emb_dim", UserFeaturesConfig.age_bracket_emb_dim
        ),
        age_mlp_hidden_dim=mparams.get("age_mlp_hidden_dim", UserFeaturesConfig.age_mlp_hidden_dim),
        age_mlp_output_dim=mparams.get("age_mlp_output_dim", UserFeaturesConfig.age_mlp_output_dim),
        user_features_concat_dim=mparams.get(
            "user_features_concat_dim", UserFeaturesConfig.user_features_concat_dim
        ),
        user_features_concat_pad=mparams.get(
            "user_features_concat_pad", UserFeaturesConfig.user_features_concat_pad
        ),
        user_features_mlp=mparams.get("user_features_mlp", UserFeaturesConfig.user_features_mlp),
    )

    CONFIGS[config_name_gen] = RecsysTrainer(
        name=config_name_gen,
        precision_level=2,
        reuse_run_id=False,
        evals=evals,
        eval_every_n=mparams.get("eval_every_n", 1000),
        model_config=RecsysTwoTowerModelConfig(
            num_global_negatives_per_example=mparams["num_global_negatives_per_example"],
            debug_mode=False,
            apply_u2u_and_i2i_loss=mparams.get("apply_u2u_and_i2i_loss", False),
            positive_actions=positive_actions,
            hard_negative_actions=hard_negative_actions,
            soft_negative_actions=soft_negative_actions,
            logq_correction_scale=mparams.get(
                "logq_correction_scale", RecsysTwoTowerModelConfig.logq_correction_scale
            ),
            user_features=user_features_config,
            checkpoint_dataset_names=checkpoint_dataset_names,
            immersive_positive_actions=mparams.get(
                "immersive_positive_actions",
                RecsysTwoTowerModelConfig.__dataclass_fields__[
                    "immersive_positive_actions"
                ].default_factory(),
            ),
            immersive_hard_negative_actions=mparams.get(
                "immersive_hard_negative_actions",
                RecsysTwoTowerModelConfig.__dataclass_fields__[
                    "immersive_hard_negative_actions"
                ].default_factory(),
            ),
            immersive_soft_negative_actions=mparams.get(
                "immersive_soft_negative_actions",
                RecsysTwoTowerModelConfig.__dataclass_fields__[
                    "immersive_soft_negative_actions"
                ].default_factory(),
            ),
            head_dataset_mapping=mparams.get("head_dataset_mapping", None),
            head_names=mparams.get("head_names", ["home"]),
            user_tower_config=RecsysAggregatedModelConfig(
                candidate_seq_len=mparams["candidate_seq_len"],
                pad_token=PAD_TOKEN,
                emb_table_width=mparams["emb_table_width"],
                history_seq_len=mparams["history_seq_len"],
                right_anchored_rope=mparams.get("right_anchored_rope", use_seqpack),
                use_seqpack=use_seqpack,
                effective_sequence_len=mparams.get("effective_sequence_len", total_seq),
                act_l2_weight=5e-7,
                transformer_output_only=True,
                use_ip_address=use_ip_address,
                search_query_embedding_dim=mparams.get("search_query_embedding_dim", 0),
                multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
                user_features=user_features_config,
                feature_prep_enabled=mparams.get("feature_prep_enabled", False),
                feature_prep=_make_feature_prep_config(mparams, scale_config),
                num_continuous_actions=mparams["num_continuous_actions"],
                use_user_embedding=use_user_embedding,
                use_post_embedding=mparams.get("use_post_embedding", True),
                use_post_sid=mparams.get("use_post_sid", False),
                sid_embed_dim=mparams.get("sid_embed_dim", 1024),
                sid_num_levels=mparams.get("sid_num_levels", 6),
                sid_codebook_size=mparams.get("sid_codebook_size", 1024),
                sid_hash_level=mparams.get("sid_hash_level", False),
                sid_cross_attn=mparams.get("sid_cross_attn", False),
                context_features=ContextFeaturesConfig(
                    enabled=mparams.get("enable_context_features", True),
                    categorical_features=[
                        CategoricalFeatureConfig(
                            index=CategoricalFeature.timezoneSeq,
                            feature_name="timezone",
                            cardinality=32,
                            embedding_dim=16,
                        ),
                        CategoricalFeatureConfig(
                            index=CategoricalFeature.localHourOfDaySeq,
                            feature_name="local_hour_of_day",
                            cardinality=25,
                            embedding_dim=16,
                        ),
                        CategoricalFeatureConfig(
                            index=CategoricalFeature.localDayOfWeekSeq,
                            feature_name="local_day_of_week",
                            cardinality=8,
                            embedding_dim=8,
                        ),
                    ],
                ),
                model_config=TransformerConfig(
                    attn_config=RecsysAttentionConfig(
                        key_size=128,
                        num_q_heads=mparams["query_heads"],
                        num_kv_heads=mparams["kv_heads"],
                        attn_logit_cap=mparams.get("attn_logit_cap", 80.0),
                        max_num_segments_per_batch=1,
                        attn_impl=mparams["attn_impl"],
                        fa_version="3",
                        qkv_merge=True,
                        attn_logit_cap_method=mparams.get("attn_logit_cap_method", "soft_sign"),
                        qk_norm=mparams.get("qk_norm", False),
                        rotary=True,
                        causal=False,
                        history_seq_len=mparams["history_seq_len"],
                        sequence_len=mparams.get("attn_sequence_len", total_seq),
                        num_user_prefix_tokens=num_user_prefix_tokens,
                    ),
                    ffn_config=FeedForwardConfig(
                        widening_factor=2,
                        gated_mlp=False,
                    ),
                    scale_config=scale_config,
                    emb_size=mparams["emb_size"],
                    num_layers=mparams["num_layers"],
                    sequence_len=total_seq,
                    primer_norm=mparams.get("primer_norm", True),
                    use_remat=True,
                    remat_policy=mparams.get("remat_policy", RematType.WHOLE),
                    use_layer_stack=True,
                    unroll_layer_stack=mparams.get("unroll_layer_stack", False),
                    output_vocab_size=mparams["output_vocab_size"],
                ),
                hash_table=hash_table,
            ),
            candidate_tower_config=RecsysCandidateModelConfig(
                scale_config=scale_config,
                emb_table_width=mparams["emb_table_width"],
                enable_linear_proj=mparams["enable_candidate_tower_linear_proj"],
                hash_table=hash_table,
                max_posts=mparams.get("max_posts", 10_240_000),
                num_candidate_heads=mparams.get("num_candidate_heads", 1),
            ),
        ),
        bs_per_device=mparams["bs_per_device"],
        dataset=dataset,
        empty_history_user_dropout_rate=mparams.get("empty_history_user_dropout_rate", 0.0),
        seqpack_distribution=mparams.get("seqpack_distribution"),
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
        emb_optim_config=RecsysEmbeddingOptimConfig(
            rowwise_adagrad=RecsysRowwiseAdagradConfig(
                learning_rate=mparams.get("emb_learning_rate", 0.1),
            ),
        ),
        max_steps=int(mparams["total_samples"] / mparams["base_batch_size"]) - 100,
        max_samples=None,
        checkpoint_config=CheckpointConfig(
            from_checkpoint=True,
            checkpoint_every_n=300,
            checkpoint_keep_every_nth=100,
            checkpoint_keep_last_n=10,
            save_final_checkpoint=True,
            verify_checksums=True,
            replica_axis="replica",
            checkpoint_chunked=False,
            checkpoint_compressed=False,
            copy_port=9988,
            checkpoint_disk_every_s=600,
        ),
    )

for _apply_config_overrides in config_registry.RETRIEVAL_CONFIG_HOOKS:
    _apply_config_overrides(CONFIGS)
