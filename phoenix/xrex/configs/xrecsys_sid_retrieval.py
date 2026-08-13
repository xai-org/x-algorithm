# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from xrex.configs import config_registry
from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.recsys.feature_config import CategoricalFeature
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.models.recsys_model import (
    CategoricalFeatureConfig,
    ContextFeaturesConfig,
    UserFeaturesConfig,
)
from xrex.models.recsys_sid_retrieval_model import RecsysSIDRetrievalConfig
from xrex.models.scaling import ScaleConfig
from xrex.models.transformer import FeedForwardConfig, RematType, TransformerConfig
from xrex.optimizers.optim import OptimConfig
from xrex.optimizers.schedule import ConstantSampleSchedule
from xrex.train.parallel_config import ParallelConfig
from xrex.train.trainer import CheckpointConfig
from xrex.train.trainer_sid_retrieval import SidRetrievalTrainer

PAD_TOKEN = 0

HISTORY_SEQ_LEN = 1023
CANDIDATE_SEQ_LEN = 64
NUM_SID_TOKENS = 6
NUM_USER_PREFIX = 1
EMB_SIZE = 2560
EMB_TABLE_WIDTH = 1024
NUM_LAYERS = 8
QUERY_HEADS = 20
KV_HEADS = 4

USER_VOCAB_SIZE = 0
ITEM_VOCAB_SIZE = 0
ITEM_HASH_VOCAB_SIZE = 100_000_000
AUTHOR_VOCAB_SIZE = 30_000_000

USER_HASH_SCALES = [196_742_702, 1_852_108_266]
USER_BIASES = [193_5840_681, 16_7407_236]
USER_MODULUS = 2_859_568_897
ITEM_HASH_SCALES = [2_161_410_491, 1_754_358_832]
ITEM_BIASES = [1_935_840_681, 167_407_236]
ITEM_MODULUS = 2_361_375_383
AUTHOR_HASH_SCALES = [371_965_780, 328_930_218]
AUTHOR_BIASES = [139_686_260, 37_755_056]
AUTHOR_MODULUS = 631_860_353

INPUT_VOCAB_K = 512
OUTPUT_VOCAB_K = 64
ACTION_TYPE_MAP_LEN = 60
CONTINUOUS_ACTION_TYPE_MAP_LEN = 8


def _round_up(n: int, k: int) -> int:
    return (n + k - 1) // k * k


INPUT_VOCAB_SIZE = _round_up(
    USER_VOCAB_SIZE + ITEM_VOCAB_SIZE + AUTHOR_VOCAB_SIZE + ACTION_TYPE_MAP_LEN + 1,
    INPUT_VOCAB_K,
)
HASH_VOCAB_SIZE = _round_up(
    USER_VOCAB_SIZE + ITEM_HASH_VOCAB_SIZE + AUTHOR_VOCAB_SIZE + ACTION_TYPE_MAP_LEN + 1,
    INPUT_VOCAB_K,
)
OUTPUT_VOCAB_SIZE = _round_up(ACTION_TYPE_MAP_LEN, OUTPUT_VOCAB_K)
NUM_CONTINUOUS_ACTIONS = _round_up(8, CONTINUOUS_ACTION_TYPE_MAP_LEN)

hash_table = HashTable(
    hash_keys=HashKeys(
        user_id_table_size=USER_VOCAB_SIZE,
        user_hash_scales=USER_HASH_SCALES,
        user_biases=USER_BIASES,
        user_modulus=USER_MODULUS,
        item_id_table_size=ITEM_VOCAB_SIZE,
        item_hash_vocab_size=ITEM_HASH_VOCAB_SIZE,
        item_hash_scales=ITEM_HASH_SCALES,
        item_biases=ITEM_BIASES,
        item_modulus=ITEM_MODULUS,
        author_id_table_size=AUTHOR_VOCAB_SIZE,
        author_hash_scales=AUTHOR_HASH_SCALES,
        author_biases=AUTHOR_BIASES,
        author_modulus=AUTHOR_MODULUS,
    ),
    output_vocab_size=OUTPUT_VOCAB_SIZE,
)

DATASET_PARAMS: dict = {
    "history_seq_len": HISTORY_SEQ_LEN,
    "candidate_seq_len": CANDIDATE_SEQ_LEN,
    "input_vocab_size": INPUT_VOCAB_SIZE,
    "hash_vocab_size": HASH_VOCAB_SIZE,
    "num_continuous_actions": NUM_CONTINUOUS_ACTIONS,
    "pad_token": PAD_TOKEN,
    "sid_num_levels": NUM_SID_TOKENS,
}


def _make_dataset(dataset_type: str = "aggregated_kafka", config_name: str = "xrecsys_sid_gen_rec"):
    factory = config_registry.SID_DATASET_FACTORIES.get(dataset_type)
    if factory is not None:
        return factory(hash_table, DATASET_PARAMS, config_name)

    match dataset_type:
        case "aggregated_kafka":
            return PhoenixDataset(
                hash_table=hash_table,
                path="/path/to/offline_kafka_dump",
                history_seq_len=HISTORY_SEQ_LEN,
                candidate_seq_len=CANDIDATE_SEQ_LEN,
                input_vocab_size=INPUT_VOCAB_SIZE,
                hash_vocab_size=HASH_VOCAB_SIZE,
                num_continuous_actions=NUM_CONTINUOUS_ACTIONS,
                pad_token=PAD_TOKEN,
                num_negatives_per_example=0,
                num_kafka_partitions=1024,
                include_candidate_post_ids=True,
                multimodal_embedding_type=None,
                use_post_sid=True,
                sid_num_levels=NUM_SID_TOKENS,
            )
        case _:
            raise ValueError(f"Unknown {dataset_type=} for the SID retrieval family")


SEQUENCE_LEN = NUM_USER_PREFIX + HISTORY_SEQ_LEN + NUM_SID_TOKENS


def make_trainer(
    *,
    name: str,
    dataset,
    attn_impl: str = "jax_attn",
    fa_version: str = "3",
) -> SidRetrievalTrainer:
    return SidRetrievalTrainer(
        name=name,
        precision_level=2,
        reuse_run_id=False,
        evals=[],
        eval_every_n=0,
        model_config=RecsysSIDRetrievalConfig(
            multimodal_embedding_type=None,
            emb_table_width=EMB_TABLE_WIDTH,
            history_seq_len=HISTORY_SEQ_LEN,
            candidate_seq_len=CANDIDATE_SEQ_LEN,
            pad_token=PAD_TOKEN,
            use_post_embedding=False,
            use_post_sid=True,
            use_user_embedding=False,
            sid_num_levels=NUM_SID_TOKENS,
            sid_codebook_size=256,
            sid_embed_dim=EMB_TABLE_WIDTH,
            sid_hash_level=True,
            sid_cross_attn=True,
            act_l2_weight=5e-7,
            num_continuous_actions=NUM_CONTINUOUS_ACTIONS,
            user_features=UserFeaturesConfig(
                enable_user_country_feature=True,
                enable_user_language_feature=True,
            ),
            context_features=ContextFeaturesConfig(
                enabled=True,
                categorical_features=[
                    CategoricalFeatureConfig(
                        feature_name="product_surface",
                        cardinality=16,
                        embedding_dim=16,
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
                    num_q_heads=QUERY_HEADS,
                    num_kv_heads=KV_HEADS,
                    attn_logit_cap=80.0,
                    max_num_segments_per_batch=1,
                    attn_impl=attn_impl,
                    fa_version=fa_version,
                    qkv_merge=True,
                    attn_logit_cap_method="soft_sign",
                    rotary=True,
                    causal=True,
                    history_seq_len=HISTORY_SEQ_LEN,
                ),
                ffn_config=FeedForwardConfig(
                    widening_factor=2,
                    gated_mlp=False,
                ),
                scale_config=ScaleConfig(),
                emb_size=EMB_SIZE,
                num_layers=NUM_LAYERS,
                sequence_len=SEQUENCE_LEN,
                primer_norm=True,
                use_remat=True,
                remat_policy=RematType.WHOLE,
                use_layer_stack=False,
                output_vocab_size=OUTPUT_VOCAB_SIZE,
            ),
            hash_table=hash_table,
        ),
        bs_per_device=128,
        dataset=dataset,
        parallel_config=ParallelConfig(
            ep=128,
            dp=1,
        ),
        optim_config=OptimConfig(
            optim="adam",
            weight_decay=1e-3,
            b1=0.95,
            b2=0.98,
        ),
        lr_schedule_in_samples_config=ConstantSampleSchedule(
            learning_rate=2e-3,
        ),
        max_steps=int(1e11 / 32) - 100,
        max_samples=None,
        checkpoint_config=CheckpointConfig(
            from_checkpoint=True,
            checkpoint_every_n=100,
            checkpoint_keep_every_nth=1000,
            checkpoint_keep_last_n=10,
            save_final_checkpoint=True,
            verify_checksums=True,
            replica_axis="replica",
            checkpoint_chunked=False,
            checkpoint_compressed=False,
            copy_port=0,
        ),
    )


CONFIGS = {
    "xrecsys_sid_gen_rec_kafka": make_trainer(
        name="xrecsys_sid_gen_rec_kafka",
        dataset=_make_dataset(),
    ),
}
for _build_model_cfgs in config_registry.SID_MODEL_CFG_BUILDERS:
    CONFIGS.update(_build_model_cfgs(make_trainer, _make_dataset))
