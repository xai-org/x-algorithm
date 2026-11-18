use crate::phoenix_scores::PhoenixScores;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandidateScoringInputs {
    pub phoenix_scores: PhoenixScores,
    pub author_id: u64,
    pub in_network: Option<bool>,
    pub is_reply: bool,
    pub is_retweet: bool,
    pub is_mutual_follow_author: bool,
    pub vqv_eligible: bool,
    pub quoted_vqv_eligible: bool,
    pub cold_start_lift_to_rank: Option<u32>,
    pub weighted_score: Option<f64>,
    pub author_exploration_bonus: f64,
    pub user_video_continuation_secs: Option<f64>,
}

impl CandidateScoringInputs {
    pub fn bidirectional_boost_eligible(&self) -> bool {
        !self.is_reply && !self.is_retweet && self.is_mutual_follow_author
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassFactors {
    pub engagement: f64,
    pub dwell_time: f64,
}

impl ClassFactors {
    pub fn new(engagement: f64, dwell_time: f64) -> Option<Self> {
        let valid = |f: f64| f.is_finite() && f >= 0.0;
        (valid(engagement) && valid(dwell_time)).then_some(Self {
            engagement,
            dwell_time,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryScoringContext {
    pub effective_oon_weight: f64,
    pub oon_factors: Option<ClassFactors>,
    pub in_network_factors: Option<ClassFactors>,
}

impl Default for QueryScoringContext {
    fn default() -> Self {
        Self {
            effective_oon_weight: 1.0,
            oon_factors: None,
            in_network_factors: None,
        }
    }
}
