use crate::inputs::CandidateScoringInputs;
use std::collections::HashMap;

pub const NEGATIVE_SCORES_OFFSET: f64 = 0.001;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValueModelWeights {
    pub favorite: f64,
    pub reply: f64,
    pub retweet: f64,
    pub photo_expand: f64,
    pub video_open: f64,
    pub click: f64,
    pub open_link: f64,
    pub profile_click: f64,
    pub vqv: f64,
    pub share: f64,
    pub share_via_dm: f64,
    pub share_via_copy_link: f64,
    pub dwell: f64,
    pub quote: f64,
    pub quoted_click: f64,
    pub quoted_vqv: f64,
    pub follow_author: f64,
    pub post_unexplored: f64,
    pub post_unexplored_include_out_of_network: bool,
    pub post_video: f64,
    pub grok_analyze: f64,
    pub not_interested: f64,
    pub block_author: f64,
    pub mute_author: f64,
    pub report: f64,
    pub not_dwelled: f64,
    pub cont_dwell_time: f64,
    pub cont_click_dwell_time: f64,
    pub video_continuation: f64,
    pub user_video_continuation: f64,
    pub profile_visit_secs: f64,
    pub min_video_duration_ms: i32,
    pub enable_quoted_vqv_duration_check: bool,
    pub bidirectional_follow_reply_weight_boost: f64,
    pub bidirectional_follow_dwell_weight_boost: f64,
    pub enable_author_diversity: bool,
    pub author_diversity_decay: f64,
    pub author_diversity_floor: f64,
    pub oon_rescore_in_network_replies_retweets: bool,
    pub multiplier_pre_offset: bool,
}

impl ValueModelWeights {
    fn positive_sum(&self) -> f64 {
        self.favorite
            + self.reply
            + self.retweet
            + self.photo_expand
            + self.video_open
            + self.click
            + self.open_link
            + self.profile_click
            + self.vqv
            + self.share
            + self.share_via_dm
            + self.share_via_copy_link
            + self.dwell
            + self.quote
            + self.quoted_click
            + self.quoted_vqv
            + self.follow_author
            + self.post_unexplored
            + self.post_video
            + self.grok_analyze
    }

    pub fn negative_sum(&self) -> f64 {
        -(self.not_interested
            + self.block_author
            + self.mute_author
            + self.report
            + self.not_dwelled)
    }

    pub fn total_sum(&self) -> f64 {
        self.positive_sum() + self.negative_sum()
    }

    pub fn reply_weight_for(&self, candidate: &CandidateScoringInputs) -> f64 {
        if self.bidirectional_follow_reply_weight_boost != 0.0
            && candidate.bidirectional_boost_eligible()
        {
            return self.reply + self.bidirectional_follow_reply_weight_boost;
        }
        self.reply
    }

    pub fn dwell_weight_for(&self, candidate: &CandidateScoringInputs) -> f64 {
        if self.bidirectional_follow_dwell_weight_boost != 0.0
            && candidate.bidirectional_boost_eligible()
        {
            return self.dwell + self.bidirectional_follow_dwell_weight_boost;
        }
        self.dwell
    }

    pub fn perturbed(mut self, sigma: f64, salt: &str, user_id: u64) -> Self {
        if sigma <= 0.0 {
            return self;
        }
        for (head, weight) in self.weights_mut() {
            *weight *= (sigma * perturbation_sign(salt, user_id, head)).exp();
        }
        self
    }

    fn weights_mut(&mut self) -> [(&'static str, &mut f64); 30] {
        [
            ("favorite", &mut self.favorite),
            ("reply", &mut self.reply),
            ("retweet", &mut self.retweet),
            ("photo_expand", &mut self.photo_expand),
            ("video_open", &mut self.video_open),
            ("click", &mut self.click),
            ("open_link", &mut self.open_link),
            ("profile_click", &mut self.profile_click),
            ("vqv", &mut self.vqv),
            ("share", &mut self.share),
            ("share_via_dm", &mut self.share_via_dm),
            ("share_via_copy_link", &mut self.share_via_copy_link),
            ("dwell", &mut self.dwell),
            ("quote", &mut self.quote),
            ("quoted_click", &mut self.quoted_click),
            ("quoted_vqv", &mut self.quoted_vqv),
            ("dwell_time", &mut self.cont_dwell_time),
            ("click_dwell_time", &mut self.cont_click_dwell_time),
            ("video_continuation", &mut self.video_continuation),
            ("user_video_continuation", &mut self.user_video_continuation),
            ("profile_visit_secs", &mut self.profile_visit_secs),
            ("follow_author", &mut self.follow_author),
            ("post_unexplored", &mut self.post_unexplored),
            ("post_video", &mut self.post_video),
            ("grok_analyze", &mut self.grok_analyze),
            ("not_interested", &mut self.not_interested),
            ("block_author", &mut self.block_author),
            ("mute_author", &mut self.mute_author),
            ("report", &mut self.report),
            ("not_dwelled", &mut self.not_dwelled),
        ]
    }

    pub fn applied_weights_map(&self) -> HashMap<String, f64> {
        HashMap::from(
            [
                ("favorite", self.favorite),
                ("reply", self.reply),
                ("retweet", self.retweet),
                ("photo_expand", self.photo_expand),
                ("video_open", self.video_open),
                ("click", self.click),
                ("open_link", self.open_link),
                ("profile_click", self.profile_click),
                ("vqv", self.vqv),
                ("share", self.share),
                ("share_via_dm", self.share_via_dm),
                ("share_via_copy_link", self.share_via_copy_link),
                ("dwell", self.dwell),
                ("quote", self.quote),
                ("quoted_click", self.quoted_click),
                ("quoted_vqv", self.quoted_vqv),
                ("follow_author", self.follow_author),
                ("post_unexplored", self.post_unexplored),
                ("pdwell", self.post_unexplored),
                ("post_video", self.post_video),
                ("grok_analyze", self.grok_analyze),
                ("not_interested", self.not_interested),
                ("block_author", self.block_author),
                ("mute_author", self.mute_author),
                ("report", self.report),
                ("not_dwelled", self.not_dwelled),
                ("dwell_time", self.cont_dwell_time),
                ("click_dwell_time", self.cont_click_dwell_time),
                (
                    "boost.bidirectional_follow_reply",
                    self.bidirectional_follow_reply_weight_boost,
                ),
                (
                    "boost.bidirectional_follow_dwell",
                    self.bidirectional_follow_dwell_weight_boost,
                ),
                (
                    "gate.quoted_vqv_duration_check",
                    self.enable_quoted_vqv_duration_check as u8 as f64,
                ),
                (
                    "gate.min_video_duration_ms",
                    self.min_video_duration_ms as f64,
                ),
            ]
            .map(|(k, v)| (k.to_string(), v)),
        )
    }
}

pub fn perturbation_sign(salt: &str, user_id: u64, head: &str) -> f64 {
    let digest = md5::compute(format!("{salt}:{user_id}:{head}"));
    if digest[0] & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}
