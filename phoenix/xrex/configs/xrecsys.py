# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import itertools
import warnings
from dataclasses import replace

from xai_proto import recsys_pb2

from xrex.configs import config_registry
from xrex.data.parquet_recsys import (
    PhoenixDataset,
    PhoenixToyDataset,
)
from xrex.data.recsys.constants import continuous_action_type_map
from xrex.data.recsys.feature_config import CategoricalFeature
from xrex.data.recsys.recsys_batch import EMBEDDING_CONFIG
from xrex.data.recsys.sequence_packing import BetaLengthDistribution
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.models.recsys_feature_prep import FeaturePrepConfig
from xrex.models.recsys_model import (
    POST_AGE_MAX_MINUTES,
    CategoricalFeatureConfig,
    ContextFeaturesConfig,
    ContinuousActionLossConfig,
    NormConfig,
    RecsysAggregatedModelConfig,
    UserFeaturesConfig,
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


def _validate_params(mparams: dict, num_user_prefix_tokens: int) -> None:
    assert mparams["emb_size"] % 128 == 0
    assert mparams["emb_table_width"] % 128 == 0

    hl = mparams["history_seq_len"]
    total_user_seq = num_user_prefix_tokens + hl
    if num_user_prefix_tokens > 1:
        assert (total_user_seq & (total_user_seq - 1)) == 0, (
            "user prefix + history "
            f"({total_user_seq} = {num_user_prefix_tokens} + {hl}) must be a power of 2"
        )
    else:
        assert ((hl + 1) & (hl)) == 0, "history length must be of the form (2^n-1)"


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
        sid_embed_dim=mparams.get("sid_embed_dim", 1024),
        sid_num_levels=mparams.get("sid_num_levels", 6),
        sid_codebook_size=mparams.get("sid_codebook_size", 1024),
        sid_hash_level=mparams.get("sid_hash_level", False),
        sid_cross_attn=mparams.get("sid_cross_attn", False),
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
        enable_product_surface=mparams.get("use_product_surface", False),
        enable_post_age=True,
        enable_timezone=enable_ctx,
        enable_dwell_time=True,
        enable_time_of_day=False,
        enable_time_of_week=False,
        enable_hour_of_day=enable_ctx,
        hour_of_day_dither_fraction=0.0,
        enable_day_of_week=enable_ctx,
    )


DATASET_TYPES: list[str] = [
    "aggregated_kafka",
    "toy_dataset",
]
DATASET_TYPES += config_registry.RANKING_EXTRA_DATASET_TYPES


def _make_cfg(
    cfg,
    user_vocab_size: int,
    item_vocab_size: int,
    author_vocab_size: int,
    ip_vocab_size: int = 0,
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
    output_vocab_size = _round_up_to_multiple(ACTION_TYPE_MAP_LEN, OUTPUT_VOCAB_K)
    num_continuous_actions = _round_up_to_multiple(
        len(continuous_action_type_map), CONTINUOUS_ACTION_TYPE_MAP_LEN
    )

    cfg["user_vocab_size"] = user_vocab_size
    cfg["item_vocab_size"] = item_vocab_size
    cfg["author_vocab_size"] = author_vocab_size
    cfg["ip_vocab_size"] = ip_vocab_size
    cfg["use_seqpack"] = cfg.get("use_seqpack", False)
    cfg["seqpack_distribution"] = cfg.get("seqpack_distribution", None)

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
    factory = config_registry.RANKING_DATASET_FACTORIES.get(dataset_type)
    if factory is not None:
        return factory(
            mparams,
            hash_table,
            mparams.get("use_post_sid", False),
            mparams.get("sid_num_levels", 6),
            config_name,
        )
    del config_name

    _use_post_sid = mparams.get("use_post_sid", False)
    _sid_num_levels = mparams.get("sid_num_levels", 6)

    match dataset_type:
        case "aggregated_kafka":
            return PhoenixDataset(
                hash_table=hash_table,
                path="/path/to/offline_kafka_dump",
                history_seq_len=mparams["history_seq_len"],
                candidate_seq_len=mparams["candidate_seq_len"],
                input_vocab_size=mparams["input_vocab_size"],
                num_continuous_actions=mparams["num_continuous_actions"],
                num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
                num_kafka_partitions=1024,
                output_vocab_size=mparams["output_vocab_size"],
                multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
                use_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
            )
        case "toy_dataset":
            return PhoenixToyDataset(
                hash_table=hash_table,
                path=None,
                history_seq_len=mparams["history_seq_len"],
                candidate_seq_len=mparams["candidate_seq_len"],
                input_vocab_size=mparams["input_vocab_size"],
                num_continuous_actions=mparams["num_continuous_actions"],
                num_negatives_per_example=mparams.get("num_negatives_per_example", 1),
                multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
                use_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
            )
        case _:
            raise ValueError(f"Uknown {dataset_type=}, must be one of {DATASET_TYPES}")


def _sequence_len(mparams, dataset, num_user_prefix_tokens: int) -> int:
    return (
        num_user_prefix_tokens
        + mparams["history_seq_len"]
        + dataset.candidate_seq_len * (1 + getattr(dataset, "num_negatives_per_example", 0))
        + getattr(dataset, "num_global_negatives_per_example", 0)
    )


def _home_direct_packed_base() -> dict:
    return {
        "history_seq_len": 1022,
        "candidate_seq_len": 64,
        "num_layers": 8,
        "emb_size": 2560,
        "emb_table_width": 1024,
        "query_heads": 20,
        "kv_heads": 4,
        "base_batch_size": 32,
        "dp": 1,
        "total_samples": 1e11,
        "emb_learning_rate": 0.2,
        "log_q_correction": True,
        "continuous_metrics_mae_mean": True,
        "post_age_granularity_mins": 60,
        "compute_post_unexplored_label": True,
        "use_seqpack": True,
        "right_anchored_rope": True,
        "qk_norm": True,
        "attn_logit_cap": -1,
        "primer_norm": False,
        "multimodal_embedding_type": None,
        "use_post_sid": True,
        "sid_embed_dim": 1024,
        "sid_num_levels": 6,
        "sid_codebook_size": 256,
        "sid_hash_level": True,
        "sid_cross_attn": False,
        "feature_prep_enabled": True,
        "feature_prep": FeaturePrepConfig(
            enable_post_sid=True,
            enable_ip_address=True,
            enable_user_country=True,
            enable_user_language=True,
            enable_user_location=True,
            enable_user_gender=True,
            enable_user_age=True,
            enable_user_installed_apps=True,
            enable_product_surface=True,
            enable_post_age=True,
            enable_timezone=True,
            enable_dwell_time=True,
            enable_time_of_day=False,
            enable_hour_of_day=True,
            hour_of_day_dither_fraction=0.1,
        ),
        "seqpack_distribution": BetaLengthDistribution(
            min_len=126,
            max_len=1022,
            mean_len=510,
            alpha=1.0,
            beta=1.0,
            block_size=128,
        ),
    }


_H100_OVERRIDES = {
    "bs_per_device": 256,
    "ep": 256,
    "attn_impl": "pallas_ranker_varlen_attn",
    "learning_rate": 1e-3,
    "checkpoint_every_n": 150,
}

_GB300_OVERRIDES = {
    "bs_per_device": 512,
    "ep": 64,
    "attn_impl": "cutedsl_ranker_varlen_attn",
    "remat_policy": RematType.SAVE_GB300_RECSYS,
    "unroll_layer_stack": True,
    "learning_rate": 5e-4,
    "checkpoint_every_n": 300,
}


MODEL_CFGS = {
    "xrecsys_seqpack": _make_cfg(
        {
            "history_seq_len": 1022,
            "candidate_seq_len": 64,
            "enable_user_country_feature": True,
            "enable_user_language_feature": True,
            "enable_user_location_feature": True,
            "enable_user_gender_feature": True,
            "enable_user_age_feature": True,
            "enable_user_installed_apps": True,
            "num_layers": 8,
            "emb_size": 2560,
            "emb_table_width": 1024,
            "query_heads": 20,
            "kv_heads": 4,
            "base_batch_size": 32,
            "bs_per_device": 256,
            "ep": 512,
            "dp": 1,
            "total_samples": 1e11,
            "learning_rate": 2e-3,
            "attn_impl": "pallas_ranker_varlen_attn",
            "log_q_correction": True,
            "use_product_surface": True,
            "post_age_granularity_mins": 60,
            "use_seqpack": True,
            "right_anchored_rope": True,
            "compute_post_unexplored_label": True,
            "multimodal_embedding_type": "v5",
            "seqpack_distribution": BetaLengthDistribution(
                min_len=126,
                max_len=1022,
                mean_len=510,
                alpha=1.0,
                beta=1.0,
                block_size=128,
            ),
        },
        user_vocab_size=100_000_000,
        item_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
        ip_vocab_size=10_000_000,
    ),
    "home_direct_packed": _make_cfg(
        {**_home_direct_packed_base(), **_H100_OVERRIDES},
        user_vocab_size=100_000_000,
        item_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
        ip_vocab_size=10_000_000,
    ),
    "home_direct_packed_gb300": _make_cfg(
        {**_home_direct_packed_base(), **_GB300_OVERRIDES},
        user_vocab_size=100_000_000,
        item_vocab_size=100_000_000,
        author_vocab_size=30_000_000,
        ip_vocab_size=10_000_000,
    ),
    "home_direct_packed_nano": _make_cfg(
        {
            **_home_direct_packed_base(),
            "learning_rate": 2e-3,
            "bs_per_device": 64,
            "ep": 1,
            "dp": 1,
            "attn_impl": "pallas_ranker_varlen_attn",
            "emb_size": 512,
            "num_layers": 4,
            "query_heads": 4,
            "kv_heads": 2,
            "emb_table_width": 128,
            "multimodal_embedding_type": None,
            "log_q_num_bins": 100_000,
            "compute_post_unexplored_label": False,
        },
        user_vocab_size=100_000,
        item_vocab_size=100_000,
        author_vocab_size=30_000,
        ip_vocab_size=10_000,
    ),
}

for _build_model_cfgs in config_registry.RANKING_MODEL_CFG_BUILDERS:
    MODEL_CFGS.update(_build_model_cfgs(_make_cfg))


config_sweeps = {
    "config_name__mparams": MODEL_CFGS.items(),
    "dataset_type": DATASET_TYPES,
}
keys = list(config_sweeps.keys())
configs = [dict(zip(keys, values)) for values in itertools.product(*config_sweeps.values())]


class _ConfigRegistry(dict[str, RecsysTrainer]):
    _DEPRECATED_POSTFIX: str = "_mask_old_bidir"

    @classmethod
    def _normalize_key(cls, key: str) -> tuple[str, bool]:
        if key.endswith(cls._DEPRECATED_POSTFIX):
            return key[: -len(cls._DEPRECATED_POSTFIX)], True
        return key, False

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return super().__contains__(key)
        normalized, _ = self._normalize_key(key)
        return super().__contains__(normalized)

    def __getitem__(self, key: str) -> RecsysTrainer:
        normalized, deprecated = self._normalize_key(key)
        if deprecated:
            warnings.warn(
                f"Config '{key}' is deprecated and will be removed soon; use '{normalized}' instead.",
                FutureWarning,
                stacklevel=2,
            )
        return super().__getitem__(normalized)

    def get(self, key: str, default: RecsysTrainer | None = None) -> RecsysTrainer | None:
        normalized, deprecated = self._normalize_key(key)
        if deprecated:
            warnings.warn(
                f"Config '{key}' is deprecated and will be removed soon; use '{normalized}' instead.",
                FutureWarning,
                stacklevel=2,
            )
        return super().get(normalized, default)


CONFIGS: dict[str, RecsysTrainer] = _ConfigRegistry()


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
            ip_id_table_size=mparams["ip_vocab_size"],
            ip_hash_scales=mparams["ip_hash_scales"],
            ip_biases=mparams["ip_biases"],
            ip_modulus=mparams["ip_modulus"],
        ),
        output_vocab_size=mparams["output_vocab_size"],
    )

    dataset = _make_dataset(mparams, dataset_type, hash_table, config_name)
    evals = []

    user_features = UserFeaturesConfig(
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
        user_features_concat_pad=mparams.get("user_features_concat_pad", False),
        user_features_mlp=mparams.get("user_features_mlp", False),
    )

    scale_config = _default_recsys_scaling()
    feature_prep = _make_feature_prep_config(mparams, scale_config)
    use_user_embedding = mparams.get("use_user_embedding", True)
    use_post_embedding = mparams.get("use_post_embedding", True)
    if mparams.get("feature_prep_enabled", False):
        num_user_prefix_tokens = int(feature_prep.enable_user_embedding) + int(
            feature_prep.has_user_features
        )
    else:
        num_user_prefix_tokens = int(use_user_embedding) + int(user_features.has_user_features)
    sequence_len = _sequence_len(mparams, dataset, num_user_prefix_tokens)
    seqpack_distribution = mparams.get("seqpack_distribution", None)
    use_ip_address = mparams.get("ip_vocab_size", 0) > 0

    _validate_params(mparams, num_user_prefix_tokens)

    CONFIGS[config_name_gen] = RecsysTrainer(
        name=config_name_gen,
        precision_level=2,
        reuse_run_id=False,
        evals=evals,
        model_config=RecsysAggregatedModelConfig(
            multimodal_embedding_type=mparams.get("multimodal_embedding_type"),
            search_query_embedding_dim=mparams.get("search_query_embedding_dim", 0),
            use_ip_address=use_ip_address,
            use_user_embedding=use_user_embedding,
            use_post_embedding=use_post_embedding,
            feature_prep_enabled=mparams.get("feature_prep_enabled", False),
            feature_prep=feature_prep,
            use_post_sid=mparams.get("use_post_sid", False),
            sid_embed_dim=mparams.get("sid_embed_dim", 1024),
            sid_num_levels=mparams.get("sid_num_levels", 6),
            sid_codebook_size=mparams.get("sid_codebook_size", 1024),
            sid_hash_level=mparams.get("sid_hash_level", False),
            sid_cross_attn=mparams.get("sid_cross_attn", False),
            use_seqpack=mparams["use_seqpack"],
            right_anchored_rope=mparams.get("right_anchored_rope", False),
            user_features=user_features,
            post_age_granularity_mins=mparams.get("post_age_granularity_mins", 60),
            post_age_max_mins=mparams.get("post_age_max_mins", POST_AGE_MAX_MINUTES),
            post_age_bucket_strategy=mparams.get("post_age_bucket_strategy", "linear"),
            post_age_num_buckets=mparams.get("post_age_num_buckets", 80),
            log_q_correction=mparams["log_q_correction"],
            log_q_num_bins=mparams.get("log_q_num_bins", 100_000_000),
            mask_candidate_positive_when_negative_action_present=mparams.get(
                "mask_candidate_positive_when_negative_action_present", False
            ),
            metric_group=mparams.get("metric_group", "default"),
            continuous_metrics_mae_mean=mparams.get("continuous_metrics_mae_mean", False),
            emb_table_width=mparams["emb_table_width"],
            history_seq_len=mparams["history_seq_len"],
            candidate_seq_len=mparams["candidate_seq_len"],
            pad_token=PAD_TOKEN,
            act_l2_weight=5e-7,
            num_continuous_actions=mparams["num_continuous_actions"],
            continuous_action_losses=[
                ContinuousActionLossConfig(
                    action_index=recsys_pb2.ContinuousActionName.DWELL_TIME,
                    metric_name="dwell-time-tweedie",
                    loss_weight=0.1,
                    loss_type="tweedie",
                    tweedie_power=1.5,
                    output_cap=300.0,
                    product_surfaces=(recsys_pb2.ProductSurface.PRODUCT_SURFACE_GALLERY_PAGE,),
                    norm_config=NormConfig(norm_scale=300.0, use_log=False),
                ),
                ContinuousActionLossConfig(
                    action_index=recsys_pb2.ContinuousActionName.DWELL_TIME,
                    metric_name="dwell-time-mae",
                    loss_weight=0.1,
                    loss_type="mae",
                    exclude_product_surfaces=(
                        recsys_pb2.ProductSurface.PRODUCT_SURFACE_GALLERY_PAGE,
                    ),
                    norm_config=NormConfig(norm_scale=30.0, use_log=False),
                ),
            ],
            context_features=ContextFeaturesConfig(
                enabled=mparams.get("enable_context_features", True),
                enable_engagement_counts=mparams.get("enable_engagement_counts", False),
                enable_author_nsfw=mparams.get("enable_author_nsfw", False),
                categorical_features=[
                    CategoricalFeatureConfig(
                        feature_name="product_surface",
                        cardinality=16,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.postAgeBucketSeq,
                        feature_name="post_age",
                        cardinality=(
                            mparams.get("post_age_num_buckets", 80)
                            if mparams.get("post_age_bucket_strategy", "linear") == "log1p"
                            else mparams.get("post_age_max_mins", POST_AGE_MAX_MINUTES)
                            // mparams.get("post_age_granularity_mins", 60)
                        )
                        + 2,
                        embedding_dim=32,
                        embedding_name="post_age_embedding_table",
                    ),
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
                        embedding_dim=mparams.get("local_hour_of_day_dim", 16),
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.localDayOfWeekSeq,
                        feature_name="local_day_of_week",
                        cardinality=8,
                        embedding_dim=8,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.authorIsNsfwSeq,
                        feature_name="author_is_nsfw",
                        cardinality=2,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.favCountBucketSeq,
                        feature_name="fav_count_bucket",
                        cardinality=32,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.replyCountBucketSeq,
                        feature_name="reply_count_bucket",
                        cardinality=32,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.repostCountBucketSeq,
                        feature_name="repost_count_bucket",
                        cardinality=32,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.quoteCountBucketSeq,
                        feature_name="quote_count_bucket",
                        cardinality=32,
                        embedding_dim=16,
                    ),
                    CategoricalFeatureConfig(
                        index=CategoricalFeature.viewCountBucketSeq,
                        feature_name="view_count_bucket",
                        cardinality=32,
                        embedding_dim=16,
                    ),
                ],
            ),
            model_config=TransformerConfig(
                attn_config=RecsysAttentionConfig(
                    key_size=128,
                    num_q_heads=mparams["query_heads"],
                    num_kv_heads=mparams["kv_heads"],
                    attn_logit_cap=mparams.get("attn_logit_cap", 80.0),
                    attn_logit_cap_method=mparams.get("attn_logit_cap_method", "soft_sign"),
                    qk_norm=mparams.get("qk_norm", False),
                    max_num_segments_per_batch=1,
                    attn_impl=mparams["attn_impl"],
                    fa_version="3",
                    qkv_merge=True,
                    rotary=True,
                    causal=False,
                    history_seq_len=mparams["history_seq_len"],
                    sequence_len=sequence_len,
                    num_user_prefix_tokens=num_user_prefix_tokens,
                ),
                ffn_config=FeedForwardConfig(
                    widening_factor=2,
                    gated_mlp=False,
                ),
                scale_config=scale_config,
                emb_size=mparams["emb_size"],
                num_layers=mparams["num_layers"],
                sequence_len=sequence_len,
                primer_norm=mparams.get("primer_norm", True),
                use_remat=True,
                remat_policy=mparams.get("remat_policy", RematType.WHOLE),
                use_layer_stack=True,
                unroll_layer_stack=mparams.get("unroll_layer_stack", False),
                output_vocab_size=mparams["output_vocab_size"],
            ),
            hash_table=hash_table,
        ),
        bs_per_device=mparams["bs_per_device"],
        seqpack_distribution=seqpack_distribution,
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
        emb_optim_config=RecsysEmbeddingOptimConfig(
            rowwise_adagrad=RecsysRowwiseAdagradConfig(
                learning_rate=mparams.get("emb_learning_rate", 0.1),
            ),
        ),
        max_steps=int(mparams["total_samples"] / mparams["base_batch_size"]) - 100,
        max_samples=None,
        checkpoint_config=CheckpointConfig(
            from_checkpoint=True,
            checkpoint_every_n=mparams.get("checkpoint_every_n", 100),
            shm_max_entries=3,
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

for _apply_config_overrides in config_registry.RANKING_CONFIG_HOOKS:
    _apply_config_overrides(CONFIGS)
