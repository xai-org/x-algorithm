mod attention_rewards;
mod inputs;
mod phoenix_scores;
mod proto;
mod scoring;
mod weights;

pub use attention_rewards::AttentionRewardScaleError;
pub use inputs::{CandidateScoringInputs, QueryScoringContext};
pub use phoenix_scores::PhoenixScores;
pub use scoring::{
    PostFusionMultiplier, ValueScores, apply_cold_start_decisions, author_pool_counts,
    compute_value_scores, compute_value_scores_with_adjustment, compute_weighted_score,
    diversity_multiplier, fuse_heads, offset_score, post_fusion_multipliers,
    set_user_video_continuation,
};
pub use weights::{NEGATIVE_SCORES_OFFSET, ValueModelWeights};
