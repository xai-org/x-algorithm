use crate::models::candidate::{MpnParts, PhoenixScores, PostCandidate, SlateContext};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::scorers::author_cold_start::AuthorColdStart;
use crate::scorers::value_model_gate::GateModel;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::scorer::Scorer;

const DWELL_REGRET_SIGMOID_MODE: &str = "dwell_regret_sigmoid";
const GATED_DWELL_REGRET_MODE: &str = "gated_dwell_regret";
const DWELL_REGRET_MEAN_EPS: f64 = 1e-9;
const DWELL_REGRET_MIN_TEMPERATURE: f64 = 1e-6;
pub(crate) struct ScoringWeights {
    favorite: f64,
    reply: f64,
    retweet: f64,
    photo_expand: f64,
    video_open: f64,
    click: f64,
    open_link: f64,
    profile_click: f64,
    vqv: f64,
    share: f64,
    share_via_dm: f64,
    share_via_copy_link: f64,
    dwell: f64,
    quote: f64,
    quoted_click: f64,
    quoted_vqv: f64,
    cont_dwell_time: f64,
    cont_click_dwell_time: f64,
    enable_click_dwell_low_fav_rate_penalty: bool,
    click_dwell_low_fav_rate_penalty_baseline: f64,
    click_dwell_low_fav_rate_penalty_alpha: f64,
    click_dwell_low_fav_rate_penalty_floor: f64,
    click_dwell_low_fav_rate_penalty_cap: f64,
    cont_active_secs_5m_residual_norm: f64,
    follow_author: f64,
    post_unexplored: f64,
    enable_multiplicative_post_unexplored: bool,
    multiplicative_post_unexplored_alpha: f64,
    post_unexplored_in_network_only: bool,
    not_interested: f64,
    block_author: f64,
    mute_author: f64,
    report: f64,
    not_dwelled: f64,
    negative_sum: f64,
    total_sum: f64,
    min_video_duration_ms: i32,
    enable_quoted_vqv_duration_check: bool,
    bidirectional_follow_reply_weight_boost: f64,
    bidirectional_follow_dwell_weight_boost: f64,
}

impl ScoringWeights {
    pub(crate) fn from_params(params: &xai_feature_switches::Params) -> Self {
        let favorite = params.get(FavoriteWeight);
        let reply = params.get(ReplyWeight);
        let retweet = params.get(RetweetWeight);
        let photo_expand = params.get(PhotoExpandWeight);
        let video_open = params.get(VideoOpenWeight);
        let click = params.get(ClickWeight);
        let open_link = params.get(OpenLinkWeight);
        let profile_click = params.get(ProfileClickWeight);
        let vqv = params.get(VqvWeight);
        let share = params.get(ShareWeight);
        let share_via_dm = params.get(ShareViaDmWeight);
        let share_via_copy_link = params.get(ShareViaCopyLinkWeight);
        let dwell = params.get(DwellWeight);
        let quote = params.get(QuoteWeight);
        let quoted_click = params.get(QuotedClickWeight);
        let quoted_vqv = params.get(QuotedVqvWeight);
        let cont_dwell_time = params.get(ContDwellTimeWeight);
        let cont_click_dwell_time = params.get(ContClickDwellTimeWeight);
        let enable_click_dwell_low_fav_rate_penalty = params.get(EnableClickDwellLowFavRatePenalty);
        let click_dwell_low_fav_rate_penalty_baseline =
            params.get(ClickDwellLowFavRatePenaltyBaseline);
        let click_dwell_low_fav_rate_penalty_alpha = params.get(ClickDwellLowFavRatePenaltyAlpha);
        let click_dwell_low_fav_rate_penalty_floor = params.get(ClickDwellLowFavRatePenaltyFloor);
        let click_dwell_low_fav_rate_penalty_cap = params.get(ClickDwellLowFavRatePenaltyCap);
        let cont_active_secs_5m_residual_norm = params.get(ContActiveSecs5mResidualNormWeight);
        let follow_author = params.get(FollowAuthorWeight);
        let post_unexplored = params.get(PostUnexploredWeight);
        let enable_multiplicative_post_unexplored = params.get(EnableMultiplicativePostUnexplored);
        let multiplicative_post_unexplored_alpha = params.get(MultiplicativePostUnexploredAlpha);
        let post_unexplored_in_network_only = params.get(PostUnexploredWeightInNetworkOnly);
        let not_interested = params.get(NotInterestedWeight);
        let block_author = params.get(BlockAuthorWeight);
        let mute_author = params.get(MuteAuthorWeight);
        let report = params.get(ReportWeight);
        let not_dwelled = params.get(NotDwelledWeight);
        let min_video_duration_ms = params.get(MinVideoDurationMs);
        let enable_quoted_vqv_duration_check = params.get(EnableQuotedVqvDurationCheck);
        let bidirectional_follow_reply_weight_boost =
            params.get(BidirectionalFollowReplyWeightBoost);
        let bidirectional_follow_dwell_weight_boost =
            params.get(BidirectionalFollowDwellWeightBoost);

        let positive_sum = favorite
            + reply
            + retweet
            + photo_expand
            + video_open
            + click
            + open_link
            + profile_click
            + vqv
            + share
            + share_via_dm
            + share_via_copy_link
            + dwell
            + quote
            + quoted_click
            + quoted_vqv
            + follow_author
            + if enable_multiplicative_post_unexplored {
                0.0
            } else {
                post_unexplored
            };
        let negative_sum = -(not_interested + block_author + mute_author + report + not_dwelled);
        let total_sum = positive_sum + negative_sum;

        Self {
            favorite,
            reply,
            retweet,
            photo_expand,
            video_open,
            click,
            open_link,
            profile_click,
            vqv,
            share,
            share_via_dm,
            share_via_copy_link,
            dwell,
            quote,
            quoted_click,
            quoted_vqv,
            cont_dwell_time,
            cont_click_dwell_time,
            enable_click_dwell_low_fav_rate_penalty,
            click_dwell_low_fav_rate_penalty_baseline,
            click_dwell_low_fav_rate_penalty_alpha,
            click_dwell_low_fav_rate_penalty_floor,
            click_dwell_low_fav_rate_penalty_cap,
            cont_active_secs_5m_residual_norm,
            follow_author,
            post_unexplored,
            enable_multiplicative_post_unexplored,
            multiplicative_post_unexplored_alpha,
            post_unexplored_in_network_only,
            not_interested,
            block_author,
            mute_author,
            report,
            not_dwelled,
            negative_sum,
            total_sum,
            min_video_duration_ms,
            enable_quoted_vqv_duration_check,
            bidirectional_follow_reply_weight_boost,
            bidirectional_follow_dwell_weight_boost,
        }
    }
}

impl ScoringWeights {
    fn post_unexplored_active_for(&self, candidate: &PostCandidate) -> bool {
        !self.post_unexplored_in_network_only || candidate.in_network == Some(true)
    }

    fn bidirectional_boost_eligible(candidate: &PostCandidate) -> bool {
        candidate.in_reply_to_tweet_id.is_none()
            && candidate.retweeted_tweet_id.is_none()
            && candidate.is_mutual_follow_author == Some(true)
    }

    fn reply_weight_for(&self, candidate: &PostCandidate) -> f64 {
        if self.bidirectional_follow_reply_weight_boost != 0.0
            && Self::bidirectional_boost_eligible(candidate)
        {
            return self.reply + self.bidirectional_follow_reply_weight_boost;
        }
        self.reply
    }

    fn low_fav_penalized_click_dwell(&self, scores: &PhoenixScores) -> Option<f64> {
        if !self.enable_click_dwell_low_fav_rate_penalty {
            return scores.click_dwell_time;
        }
        match (scores.click_dwell_time, scores.favorite_score) {
            (Some(cd), Some(fav)) => {
                let baseline = self
                    .click_dwell_low_fav_rate_penalty_baseline
                    .max(f64::EPSILON);
                let multiplier = (fav / baseline)
                    .powf(self.click_dwell_low_fav_rate_penalty_alpha)
                    .max(self.click_dwell_low_fav_rate_penalty_floor)
                    .min(self.click_dwell_low_fav_rate_penalty_cap);
                Some(cd * multiplier)
            }
            (cd, None) => cd,
            (None, _) => None,
        }
    }

    fn dwell_weight_for(&self, candidate: &PostCandidate) -> f64 {
        if self.bidirectional_follow_dwell_weight_boost != 0.0
            && Self::bidirectional_boost_eligible(candidate)
        {
            return self.dwell + self.bidirectional_follow_dwell_weight_boost;
        }
        self.dwell
    }

    pub(crate) fn effective_head_weights(
        &self,
        query: &ScoredPostsQuery,
        candidate: &PostCandidate,
    ) -> xai_vm_ranker_proto::HeadWeights {
        let scores = &candidate.phoenix_scores;
        let vqv = crate::util::candidates_util::vqv_weight(
            query,
            candidate,
            self.min_video_duration_ms,
            self.vqv,
        );
        let dwell_time = match scores.post_unexplored_score {
            Some(post_unexplored)
                if self.enable_multiplicative_post_unexplored
                    && self.post_unexplored_active_for(candidate) =>
            {
                self.cont_dwell_time
                    * (1.0 + post_unexplored * self.multiplicative_post_unexplored_alpha)
            }
            _ => self.cont_dwell_time,
        };
        let click_dwell_time = if self.enable_click_dwell_low_fav_rate_penalty {
            match (scores.click_dwell_time, scores.favorite_score) {
                (Some(_), Some(fav)) => {
                    let baseline = self
                        .click_dwell_low_fav_rate_penalty_baseline
                        .max(f64::EPSILON);
                    let multiplier = (fav / baseline)
                        .powf(self.click_dwell_low_fav_rate_penalty_alpha)
                        .max(self.click_dwell_low_fav_rate_penalty_floor)
                        .min(self.click_dwell_low_fav_rate_penalty_cap);
                    self.cont_click_dwell_time * multiplier
                }
                _ => self.cont_click_dwell_time,
            }
        } else {
            self.cont_click_dwell_time
        };
        let quoted_vqv = crate::util::candidates_util::quoted_vqv_weight(
            candidate,
            self.min_video_duration_ms,
            self.quoted_vqv,
            self.enable_quoted_vqv_duration_check,
        );
        let post_unexplored = if !self.enable_multiplicative_post_unexplored
            && self.post_unexplored_active_for(candidate)
        {
            self.post_unexplored
        } else {
            0.0
        };
        xai_vm_ranker_proto::HeadWeights {
            favorite: Some(self.favorite),
            reply: Some(self.reply_weight_for(candidate)),
            retweet: Some(self.retweet),
            photo_expand: Some(self.photo_expand),
            click: Some(self.click),
            profile_click: Some(self.profile_click),
            vqv: Some(vqv),
            share: Some(self.share),
            share_via_dm: Some(self.share_via_dm),
            share_via_copy_link: Some(self.share_via_copy_link),
            dwell: Some(self.dwell_weight_for(candidate)),
            quote: Some(self.quote),
            quoted_click: Some(self.quoted_click),
            follow_author: Some(self.follow_author),
            not_interested: Some(self.not_interested),
            block_author: Some(self.block_author),
            mute_author: Some(self.mute_author),
            report: Some(self.report),
            dwell_time: Some(dwell_time),
            click_dwell_time: Some(click_dwell_time),
            not_dwelled: Some(self.not_dwelled),
            video_open: Some(self.video_open),
            open_link: Some(self.open_link),
            quoted_vqv: Some(quoted_vqv),
            post_unexplored: Some(post_unexplored),
            active_secs_5m_residual_norm: Some(self.cont_active_secs_5m_residual_norm),
        }
    }

    pub(crate) fn applied_weights_map(&self) -> HashMap<String, f64> {
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
                ("not_interested", self.not_interested),
                ("block_author", self.block_author),
                ("mute_author", self.mute_author),
                ("report", self.report),
                ("not_dwelled", self.not_dwelled),
                ("dwell_time", self.cont_dwell_time),
                ("click_dwell_time", self.cont_click_dwell_time),
                (
                    "active_secs_5m_residual_norm",
                    self.cont_active_secs_5m_residual_norm,
                ),
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
                (
                    "gate.multiplicative_post_unexplored",
                    self.enable_multiplicative_post_unexplored as u8 as f64,
                ),
                (
                    "boost.multiplicative_post_unexplored_alpha",
                    self.multiplicative_post_unexplored_alpha,
                ),
                (
                    "gate.post_unexplored_in_network_only",
                    self.post_unexplored_in_network_only as u8 as f64,
                ),
                (
                    "gate.click_dwell_low_fav_rate_penalty",
                    self.enable_click_dwell_low_fav_rate_penalty as u8 as f64,
                ),
            ]
            .map(|(k, v)| (k.to_string(), v)),
        )
    }
}

pub(crate) struct DwellRegretWeights {
    alpha_favorite: f64,
    alpha_reply: f64,
    alpha_retweet: f64,
    alpha_quote: f64,
    alpha_share: f64,
    alpha_share_via_dm: f64,
    alpha_share_via_copy_link: f64,
    neg_not_interested: f64,
    neg_block_author: f64,
    neg_mute_author: f64,
    neg_report: f64,
    temperature: f64,
    dwell_floor: f64,
}

impl DwellRegretWeights {
    pub(crate) fn from_params(params: &xai_feature_switches::Params) -> Self {
        Self {
            alpha_favorite: params.get(DwellRegretAlphaFavorite),
            alpha_reply: params.get(DwellRegretAlphaReply),
            alpha_retweet: params.get(DwellRegretAlphaRetweet),
            alpha_quote: params.get(DwellRegretAlphaQuote),
            alpha_share: params.get(DwellRegretAlphaShare),
            alpha_share_via_dm: params.get(DwellRegretAlphaShareViaDm),
            alpha_share_via_copy_link: params.get(DwellRegretAlphaShareViaCopyLink),
            neg_not_interested: params.get(DwellRegretNegNotInterested),
            neg_block_author: params.get(DwellRegretNegBlockAuthor),
            neg_mute_author: params.get(DwellRegretNegMuteAuthor),
            neg_report: params.get(DwellRegretNegReport),
            temperature: params.get(DwellRegretTemperature),
            dwell_floor: params.get(DwellRegretDwellFloor),
        }
    }
}

pub struct RankingScorer {
    pub author_cold_start: AuthorColdStart,
}

impl RankingScorer {
    // These weights reflect a combination of how much an action is
    // valued in ranking and typical propensities of these actions
    // across the X network (e.g. negative feedback is overall rare).

    // Each weight multiplies the *predicted* probability of that
    // action (P(favorite), P(repost), …) or a continuous value e.g.
    // watch time -- the weights do not multiply raw engagement counts.
    // One common misinterpretation is that you can read these weight
    // ratios as count equivalences, e.g. the incorrect statement that
    // "one report cancels 468 likes" -- this is incorrect because the
    // weights apply to the predicted probabilities rather than raw counts.

    // And the baseline probability of a Report is more than 1000x lower
    // than a Like, so it’s weighted more to allow the prediction to affect
    // the final ranking at all.

    // Related to the above is a misunderstanding that bad actors engaging
    // in mass blocking/reporting will significantly suppress reach. There
    // are multiple things inhibiting this:
    // 1. It’s predicting your likelihood of the action, not summing up
    // raw weights on counts. Also, recommendations are personalized, so
    // reports from bad actors will primarily affect recommendations for
    // users who are similar to the bad actors, rather than having the same
    // effect on the post's ranking to everyone.
    // 2. For an account to count in the algorithms recommendation system,
    // it must take place on a post served in Home Timeline. Directly
    // navigating to a post (i.e., coordinating via groupchat) has no
    // ranking impact. And users cannot manufacture a post to show up in
    // their Timeline in any consistently reproducible way.
    fn apply(score: Option<f64>, weight: f64) -> f64 {
        score.unwrap_or(0.0) * weight
    }

    pub(crate) fn compute_weighted_score(
        weights: &ScoringWeights,
        query: &ScoredPostsQuery,
        candidate: &PostCandidate,
    ) -> f64 {
        let (pos, neg) = Self::compute_weighted_parts(weights, query, candidate);
        Self::offset_score(pos - neg, weights)
    }

    pub(crate) fn compute_weighted_parts(
        weights: &ScoringWeights,
        query: &ScoredPostsQuery,
        candidate: &PostCandidate,
    ) -> (f64, f64) {
        let scores: &PhoenixScores = &candidate.phoenix_scores;

        let vqv_weight = crate::util::candidates_util::vqv_weight(
            query,
            candidate,
            weights.min_video_duration_ms,
            weights.vqv,
        );

        let quoted_vqv_weight = crate::util::candidates_util::quoted_vqv_weight(
            candidate,
            weights.min_video_duration_ms,
            weights.quoted_vqv,
            weights.enable_quoted_vqv_duration_check,
        );

        let post_unexplored_active = weights.post_unexplored_active_for(candidate);

        let base_dwell_time_term = Self::apply(scores.dwell_time, weights.cont_dwell_time);
        let dwell_time_term = match scores.post_unexplored_score {
            Some(post_unexplored)
                if weights.enable_multiplicative_post_unexplored && post_unexplored_active =>
            {
                base_dwell_time_term
                    * (1.0 + post_unexplored * weights.multiplicative_post_unexplored_alpha)
            }
            _ => base_dwell_time_term,
        };

        let post_unexplored_term = if post_unexplored_active {
            Self::apply(scores.post_unexplored_score, weights.post_unexplored)
        } else {
            0.0
        };

        let terms = [
            Self::apply(scores.favorite_score, weights.favorite),
            Self::apply(scores.reply_score, weights.reply_weight_for(candidate)),
            Self::apply(scores.retweet_score, weights.retweet),
            Self::apply(scores.photo_expand_score, weights.photo_expand),
            Self::apply(scores.video_open_score, weights.video_open),
            Self::apply(scores.click_score, weights.click),
            Self::apply(scores.open_link_score, weights.open_link),
            Self::apply(scores.profile_click_score, weights.profile_click),
            Self::apply(scores.vqv_score, vqv_weight),
            Self::apply(scores.share_score, weights.share),
            Self::apply(scores.share_via_dm_score, weights.share_via_dm),
            Self::apply(
                scores.share_via_copy_link_score,
                weights.share_via_copy_link,
            ),
            Self::apply(scores.dwell_score, weights.dwell_weight_for(candidate)),
            Self::apply(scores.quote_score, weights.quote),
            Self::apply(scores.quoted_click_score, weights.quoted_click),
            Self::apply(scores.quoted_vqv_score, quoted_vqv_weight),
            dwell_time_term,
            Self::apply(
                weights.low_fav_penalized_click_dwell(scores),
                weights.cont_click_dwell_time,
            ),
            Self::apply(
                scores.active_secs_5m_residual_norm,
                weights.cont_active_secs_5m_residual_norm,
            ),
            Self::apply(scores.follow_author_score, weights.follow_author),
            Self::apply(scores.not_interested_score, weights.not_interested),
            Self::apply(scores.block_author_score, weights.block_author),
            Self::apply(scores.mute_author_score, weights.mute_author),
            Self::apply(scores.report_score, weights.report),
            Self::apply(scores.not_dwelled_score, weights.not_dwelled),
            if weights.enable_multiplicative_post_unexplored {
                0.0
            } else {
                post_unexplored_term
            },
        ];

        let mut pos = 0.0;
        let mut neg = 0.0;
        for t in terms {
            if t >= 0.0 {
                pos += t;
            } else {
                neg -= t;
            }
        }
        (pos, neg)
    }

    pub(crate) fn offset_score(combined_score: f64, w: &ScoringWeights) -> f64 {
        if w.total_sum == 0.0 {
            combined_score.max(0.0)
        } else if combined_score < 0.0 {
            (combined_score + w.negative_sum) / w.total_sum * NEGATIVE_SCORES_OFFSET
        } else {
            combined_score + NEGATIVE_SCORES_OFFSET
        }
    }

    pub(crate) fn compute_dwell_regret_base_scores(
        w: &DwellRegretWeights,
        candidates: &[PostCandidate],
    ) -> Vec<f64> {
        let n = candidates.len();
        if n == 0 {
            return Vec::new();
        }
        let inv_n = 1.0 / n as f64;

        let mut mean_favorite = 0.0;
        let mut mean_reply = 0.0;
        let mut mean_retweet = 0.0;
        let mut mean_quote = 0.0;
        let mut mean_share = 0.0;
        let mut mean_share_via_dm = 0.0;
        let mut mean_share_via_copy_link = 0.0;
        for c in candidates {
            let ps = &c.phoenix_scores;
            mean_favorite += ps.favorite_score.unwrap_or(0.0);
            mean_reply += ps.reply_score.unwrap_or(0.0);
            mean_retweet += ps.retweet_score.unwrap_or(0.0);
            mean_quote += ps.quote_score.unwrap_or(0.0);
            mean_share += ps.share_score.unwrap_or(0.0);
            mean_share_via_dm += ps.share_via_dm_score.unwrap_or(0.0);
            mean_share_via_copy_link += ps.share_via_copy_link_score.unwrap_or(0.0);
        }
        mean_favorite *= inv_n;
        mean_reply *= inv_n;
        mean_retweet *= inv_n;
        mean_quote *= inv_n;
        mean_share *= inv_n;
        mean_share_via_dm *= inv_n;
        mean_share_via_copy_link *= inv_n;

        let temperature = w.temperature.max(DWELL_REGRET_MIN_TEMPERATURE);

        candidates
            .iter()
            .map(|c| {
                let ps = &c.phoenix_scores;
                let positive = w.alpha_favorite
                    * Self::centered_ratio(ps.favorite_score, mean_favorite)
                    + w.alpha_reply * Self::centered_ratio(ps.reply_score, mean_reply)
                    + w.alpha_retweet * Self::centered_ratio(ps.retweet_score, mean_retweet)
                    + w.alpha_quote * Self::centered_ratio(ps.quote_score, mean_quote)
                    + w.alpha_share * Self::centered_ratio(ps.share_score, mean_share)
                    + w.alpha_share_via_dm
                        * Self::centered_ratio(ps.share_via_dm_score, mean_share_via_dm)
                    + w.alpha_share_via_copy_link
                        * Self::centered_ratio(
                            ps.share_via_copy_link_score,
                            mean_share_via_copy_link,
                        );
                let negative = w.neg_not_interested * ps.not_interested_score.unwrap_or(0.0)
                    + w.neg_block_author * ps.block_author_score.unwrap_or(0.0)
                    + w.neg_mute_author * ps.mute_author_score.unwrap_or(0.0)
                    + w.neg_report * ps.report_score.unwrap_or(0.0);
                let modulation = 2.0
                    * Self::sigmoid(positive / temperature)
                    * (negative.min(0.0) / temperature).exp();
                let dwell = ps.dwell_time.unwrap_or(0.0).max(w.dwell_floor).max(0.0);
                dwell * modulation
            })
            .collect()
    }

    fn centered_ratio(p: Option<f64>, mean: f64) -> f64 {
        if mean < DWELL_REGRET_MEAN_EPS {
            0.0
        } else {
            p.unwrap_or(0.0) / mean - 1.0
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn diversity_multiplier(decay_factor: f64, floor: f64, exponent: f64) -> f64 {
        (1.0 - floor) * decay_factor.powf(exponent) + floor
    }

    fn compute_slate_contexts(
        candidates: &[PostCandidate],
        pre_diversity_scores: &[f64],
    ) -> Vec<SlateContext> {
        let mut indexed: Vec<(usize, f64)> = pre_diversity_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let mut contexts = vec![SlateContext::default(); candidates.len()];
        let mut author_counts: FxHashMap<u64, u32> = FxHashMap::default();
        let mut last_author_rank: FxHashMap<u64, u32> = FxHashMap::default();
        let mut sid_counts: [FxHashMap<u64, u32>; 3] = Default::default();
        let mut last_sid_rank: [FxHashMap<u64, u32>; 3] = Default::default();
        for (rank, (idx, score)) in indexed.into_iter().enumerate() {
            let rank = rank as u32;
            let author_id = candidates[idx].author_id;
            let k = author_counts.get(&author_id).copied().unwrap_or(0);
            let rank_gap = last_author_rank.get(&author_id).map(|last| rank - last);

            let mut sid_k = [0u32; 3];
            let mut sid_gap = [None; 3];
            let sids = candidates[idx].semantic_ids.as_deref().unwrap_or(&[]);
            let sid_known = !sids.is_empty();
            let mut prefix = 0u64;
            for (level, &code) in sids.iter().take(3).enumerate() {
                prefix = (prefix << 20) | (code as u32 as u64 & 0xFFFFF);
                sid_k[level] = sid_counts[level].get(&prefix).copied().unwrap_or(0);
                sid_gap[level] = last_sid_rank[level].get(&prefix).map(|last| rank - last);
                sid_counts[level].insert(prefix, sid_k[level] + 1);
                last_sid_rank[level].insert(prefix, rank);
            }

            contexts[idx] = SlateContext {
                k,
                pool_rank: rank,
                pool_rank_gap: rank_gap,
                fatigue: 0.0,
                pre_diversity_score: score,
                sid_known,
                sid_k_l1: sid_k[0],
                sid_k_l2: sid_k[1],
                sid_k_l3: sid_k[2],
                sid_gap_l1: sid_gap[0],
                sid_gap_l2: sid_gap[1],
                sid_gap_l3: sid_gap[2],
            };
            author_counts.insert(author_id, k + 1);
            last_author_rank.insert(author_id, rank);
        }

        contexts
    }

    fn stored_slate_contexts(candidates: &[PostCandidate]) -> Option<Vec<SlateContext>> {
        candidates.iter().map(|c| c.slate_context).collect()
    }

    fn author_diversity_multipliers(
        query: &ScoredPostsQuery,
        contexts: &[SlateContext],
    ) -> Vec<f64> {
        let decay_factor = query.params.get(AuthorDiversityDecay);
        let floor = query.params.get(AuthorDiversityFloor);

        contexts
            .iter()
            .map(|context| Self::diversity_multiplier(decay_factor, floor, f64::from(context.k)))
            .collect()
    }

    fn apply_author_diversity(
        query: &ScoredPostsQuery,
        contexts: &[SlateContext],
        pre_diversity_scores: &[f64],
    ) -> Vec<f64> {
        let multipliers = Self::author_diversity_multipliers(query, contexts);
        pre_diversity_scores
            .iter()
            .zip(multipliers)
            .map(|(&score, multiplier)| score * multiplier)
            .collect()
    }

    fn effective_oon_weight(query: &ScoredPostsQuery) -> f64 {
        if !query.topic_ids.is_empty() {
            return query.params.get(TopicOonWeightFactor);
        }

        let oon_weight_factor = query.params.get(OonWeightFactor);

        let new_user_age_threshold = Duration::from_secs(query.params.get(NewUserAgeThresholdSecs));

        let is_eligible_new_user = duration_since_creation_opt(query.user_id)
            .map(|age| age < new_user_age_threshold)
            .unwrap_or(false)
            && query.user_features.followed_user_ids.len() >= NEW_USER_MIN_FOLLOWING;

        if is_eligible_new_user {
            NEW_USER_OON_WEIGHT_FACTOR
        } else {
            oon_weight_factor
        }
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for RankingScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableRanking)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let weights = ScoringWeights::from_params(&query.params);
        let enable_author_diversity = query.params.get(EnableAuthorDiversity);

        let use_dwell_regret = match query.params.get(ValueModelMode).as_str() {
            DWELL_REGRET_SIGMOID_MODE => true,
            GATED_DWELL_REGRET_MODE => GateModel::from_params(&query.params)
                .is_some_and(|gate| gate.serve_new_scoring(query)),
            _ => false,
        };
        let weighted_parts: Vec<(f64, f64)> = if use_dwell_regret {
            Vec::new()
        } else {
            candidates
                .iter()
                .map(|c| Self::compute_weighted_parts(&weights, query, c))
                .collect()
        };
        let weighted_scores: Vec<f64> = if use_dwell_regret {
            let dr_weights = DwellRegretWeights::from_params(&query.params);
            Self::compute_dwell_regret_base_scores(&dr_weights, candidates)
        } else {
            weighted_parts
                .iter()
                .map(|&(pos, neg)| Self::offset_score(pos - neg, &weights))
                .collect()
        };

        let mpn_scoring = query.params.get(EnableMpnScoring) && !use_dwell_regret;

        let effective_oon = Self::effective_oon_weight(query);
        let deboost_in_network_replies_retweets = query
            .params
            .get(EnableOonRescoreForInNetworkRepliesRetweets);
        let oon_applies = |c: &PostCandidate| match c.in_network {
            Some(false) => true,
            Some(true) => {
                deboost_in_network_replies_retweets
                    && (c.in_reply_to_tweet_id.is_some() || c.retweeted_tweet_id.is_some())
            }
            None => false,
        };

        if mpn_scoring {
            let persisted_contexts: Option<Vec<SlateContext>> = if query.has_cached_posts {
                Self::stored_slate_contexts(candidates)
            } else {
                Some(Self::compute_slate_contexts(candidates, &weighted_scores))
            };

            let diversity_multipliers: Vec<f64> = if enable_author_diversity {
                let recomputed_contexts;
                let scoring_contexts: &[SlateContext] = match &persisted_contexts {
                    Some(contexts) if !query.has_cached_posts => contexts,
                    _ => {
                        recomputed_contexts =
                            Self::compute_slate_contexts(candidates, &weighted_scores);
                        &recomputed_contexts
                    }
                };
                Self::author_diversity_multipliers(query, scoring_contexts)
            } else {
                vec![1.0; candidates.len()]
            };

            let scalar_multipliers: Vec<f64> = candidates
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut m = diversity_multipliers[i];
                    if oon_applies(c) {
                        m *= effective_oon;
                    }
                    m
                })
                .collect();

            let mpn_scores: Vec<f64> = weighted_parts
                .iter()
                .zip(&scalar_multipliers)
                .map(|(&(pos, neg), &m)| {
                    let net = pos - neg;
                    let scaled = if net >= 0.0 { m * net } else { net };
                    Self::offset_score(scaled, &weights)
                })
                .collect();

            let final_scores = self.author_cold_start.apply(query, candidates, &mpn_scores);

            return weighted_scores
                .iter()
                .zip(final_scores)
                .enumerate()
                .map(|(i, (&weighted, score))| {
                    Ok(PostCandidate {
                        weighted_score: Some(weighted),
                        score: Some(score),
                        slate_context: persisted_contexts.as_ref().map(|contexts| contexts[i]),
                        mpn_parts: Some(MpnParts {
                            pos: weighted_parts[i].0,
                            neg: weighted_parts[i].1,
                            scalar_multiplier: scalar_multipliers[i],
                        }),
                        ..Default::default()
                    })
                })
                .collect();
        }

        let adjusted_scores = self
            .author_cold_start
            .apply(query, candidates, &weighted_scores);

        let persisted_contexts: Option<Vec<SlateContext>> = if query.has_cached_posts {
            Self::stored_slate_contexts(candidates)
        } else {
            Some(Self::compute_slate_contexts(candidates, &adjusted_scores))
        };

        let diversity_adjusted = if enable_author_diversity {
            let recomputed_contexts;
            let scoring_contexts: &[SlateContext] = match &persisted_contexts {
                Some(contexts) if !query.has_cached_posts => contexts,
                _ => {
                    recomputed_contexts =
                        Self::compute_slate_contexts(candidates, &adjusted_scores);
                    &recomputed_contexts
                }
            };
            Self::apply_author_diversity(query, scoring_contexts, &adjusted_scores)
        } else {
            adjusted_scores.clone()
        };

        let final_scores: Vec<f64> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let after_diversity = diversity_adjusted[i];
                if oon_applies(c) {
                    after_diversity * effective_oon
                } else {
                    after_diversity
                }
            })
            .collect();

        weighted_scores
            .iter()
            .zip(final_scores)
            .enumerate()
            .map(|(i, (&weighted, score))| {
                Ok(PostCandidate {
                    weighted_score: Some(weighted),
                    score: Some(score),
                    slate_context: persisted_contexts.as_ref().map(|contexts| contexts[i]),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
        candidate.score = scored.score;
        candidate.slate_context = scored.slate_context;
        candidate.mpn_parts = scored.mpn_parts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::author_rules::AuthorRulesEvaluator;
    use std::sync::Arc;

    const GATE_WEIGHTS_ALL_ZERO: &str = "seq_len:0,n_fav:0,n_reply:0,n_rt_quote:0,n_vqv:0,n_click:0,n_bm_share:0,n_profile_follow:0,n_photo:0,n_negfb:0,n_7d:0,n_1d:0,active_days:0,active_days_7d:0,days_since_last:0,span_days:0,followers:0,followings:0,account_age_years:0";

    fn test_scorer() -> RankingScorer {
        let fs = Arc::new(xai_feature_switches::FeatureSwitches::new(vec![]).unwrap());
        RankingScorer {
            author_cold_start: AuthorColdStart {
                author_rules: Arc::new(AuthorRulesEvaluator::new(fs)),
            },
        }
    }

    fn query_with_flags(flags: &[(&str, &str)]) -> ScoredPostsQuery {
        let mut query = ScoredPostsQuery::default();
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in flags {
            results.override_fs(key.to_string(), value);
        }
        query.params = results.into();
        query
    }

    fn candidate(author_id: u64, in_network: Option<bool>) -> PostCandidate {
        PostCandidate {
            author_id,
            in_network,
            ..Default::default()
        }
    }

    fn candidate_with_reply(author_id: u64, in_network: Option<bool>) -> PostCandidate {
        PostCandidate {
            author_id,
            in_network,
            in_reply_to_tweet_id: Some(42),
            ..Default::default()
        }
    }

    fn candidate_with_retweet(author_id: u64, in_network: Option<bool>) -> PostCandidate {
        PostCandidate {
            author_id,
            in_network,
            retweeted_tweet_id: Some(42),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn applies_author_diversity_decay_in_score_order() {
        let scorer = test_scorer();
        let candidates = vec![
            candidate(1, Some(true)),
            candidate(2, Some(true)),
            candidate(1, Some(true)),
        ];

        let query = query_with_flags(&[
            ("rust_home_mixer_enable_author_diversity", "true"),
            ("rust_home_mixer_author_diversity_decay", "0.5"),
            ("rust_home_mixer_author_diversity_floor", "0.25"),
            ("rust_home_mixer_value_model_mode", "weighted"),
            ("rust_home_mixer_enable_mpn_scoring", "false"),
        ]);
        let scored = scorer.score(&query, &candidates).await;

        let decay_factor = query.params.get(AuthorDiversityDecay);
        let floor = query.params.get(AuthorDiversityFloor);

        let first = scored[0].as_ref().unwrap().score.unwrap();
        let second = scored[1].as_ref().unwrap().score.unwrap();
        let third = scored[2].as_ref().unwrap().score.unwrap();

        assert!((first - second).abs() < 1e-9);
        let expected_multiplier = RankingScorer::diversity_multiplier(decay_factor, floor, 1.0);
        assert!((third - first * expected_multiplier).abs() < 1e-9);
    }

    #[tokio::test]
    async fn recomputes_contexts_for_scoring_on_cached_posts() {
        let scorer = test_scorer();
        let stored_context = SlateContext {
            k: 5,
            pool_rank: 9,
            pool_rank_gap: Some(3),
            fatigue: 0.0,
            pre_diversity_score: 0.5,
            ..Default::default()
        };
        let stored_repeat = PostCandidate {
            slate_context: Some(stored_context),
            ..candidate(1, Some(true))
        };
        let stored_fresh = PostCandidate {
            slate_context: Some(SlateContext::default()),
            ..candidate(2, Some(true))
        };

        let mut query = query_with_flags(&[
            ("rust_home_mixer_enable_author_diversity", "true"),
            ("rust_home_mixer_author_diversity_decay", "0.5"),
            ("rust_home_mixer_author_diversity_floor", "0.25"),
        ]);
        query.has_cached_posts = true;

        let scored = scorer.score(&query, &[stored_repeat, stored_fresh]).await;

        let repeat = scored[0].as_ref().unwrap();
        let fresh = scored[1].as_ref().unwrap();
        assert!((repeat.score.unwrap() - fresh.score.unwrap()).abs() < 1e-9);
        assert_eq!(repeat.slate_context, Some(stored_context));
    }

    #[tokio::test]
    async fn falls_back_to_pool_recompute_when_stored_missing() {
        let scorer = test_scorer();
        let candidates = vec![
            candidate(1, Some(true)),
            candidate(2, Some(true)),
            candidate(1, Some(true)),
        ];

        let mut query = query_with_flags(&[
            ("rust_home_mixer_enable_author_diversity", "true"),
            ("rust_home_mixer_author_diversity_decay", "0.5"),
            ("rust_home_mixer_author_diversity_floor", "0.25"),
            ("rust_home_mixer_value_model_mode", "weighted"),
            ("rust_home_mixer_enable_mpn_scoring", "false"),
        ]);
        query.has_cached_posts = true;

        let scored = scorer.score(&query, &candidates).await;

        let first = scored[0].as_ref().unwrap().score.unwrap();
        let third = scored[2].as_ref().unwrap().score.unwrap();
        let expected_multiplier = RankingScorer::diversity_multiplier(0.5, 0.25, 1.0);
        assert!((third - first * expected_multiplier).abs() < 1e-9);
        assert_eq!(scored[2].as_ref().unwrap().slate_context, None);
    }

    #[tokio::test]
    async fn applies_oon_discount_to_out_of_network() {
        let scorer = test_scorer();
        let candidates = vec![candidate(1, Some(true)), candidate(2, Some(false))];

        let query = query_with_flags(&[
            ("rust_home_mixer_oon_weight_factor", "0.75"),
            ("rust_home_mixer_value_model_mode", "weighted"),
            ("rust_home_mixer_enable_mpn_scoring", "false"),
        ]);
        let scored = scorer.score(&query, &candidates).await;

        let in_network_score = scored[0].as_ref().unwrap().score.unwrap();
        let oon_score = scored[1].as_ref().unwrap().score.unwrap();

        assert!((oon_score - in_network_score * 0.75).abs() < 1e-9);
    }

    #[test]
    fn video_open_head_is_weighted_into_score() {
        let zero_query = query_with_flags(&[("rust_home_mixer_video_open_weight", "0.0")]);
        let zero_weights = ScoringWeights::from_params(&zero_query.params);

        let with_video_open = PostCandidate {
            phoenix_scores: PhoenixScores {
                video_open_score: Some(0.4),
                ..Default::default()
            },
            ..candidate(1, Some(true))
        };
        let without_video_open = candidate(1, Some(true));

        let zero_with =
            RankingScorer::compute_weighted_score(&zero_weights, &zero_query, &with_video_open);
        let zero_without =
            RankingScorer::compute_weighted_score(&zero_weights, &zero_query, &without_video_open);
        assert!((zero_with - zero_without).abs() < 1e-9);

        let query = query_with_flags(&[("rust_home_mixer_video_open_weight", "0.3")]);
        let weights = ScoringWeights::from_params(&query.params);
        let scored = RankingScorer::compute_weighted_score(&weights, &query, &with_video_open);
        let baseline = RankingScorer::compute_weighted_score(&weights, &query, &without_video_open);
        assert!((scored - baseline - 0.3 * 0.4).abs() < 1e-9);
    }

    #[test]
    fn post_unexplored_additive_when_multiplicative_disabled() {
        let with_post_unexplored = PostCandidate {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.5),
                post_unexplored_score: Some(0.4),
                ..Default::default()
            },
            ..candidate(1, Some(true))
        };
        let without_post_unexplored = PostCandidate {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.5),
                ..Default::default()
            },
            ..candidate(1, Some(true))
        };

        let query = query_with_flags(&post_unexplored_test_flags(&[
            ("rust_home_mixer_post_unexplored_weight", "1.5"),
            (
                "rust_home_mixer_enable_multiplicative_post_unexplored",
                "false",
            ),
            (
                "rust_home_mixer_post_unexplored_weight_in_network_only",
                "false",
            ),
        ]));
        let weights = ScoringWeights::from_params(&query.params);
        let scored = RankingScorer::compute_weighted_score(&weights, &query, &with_post_unexplored);
        let baseline =
            RankingScorer::compute_weighted_score(&weights, &query, &without_post_unexplored);
        assert!((baseline - (1.0 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
        assert!((scored - (1.0 + 0.6 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
    }

    fn post_unexplored_test_flags<'a>(extra: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut flags = vec![
            ("rust_home_mixer_favorite_weight", "2.0"),
            ("rust_home_mixer_cont_dwell_time_weight", "0.0"),
            ("rust_home_mixer_not_interested_weight", "0.0"),
            ("rust_home_mixer_block_author_weight", "0.0"),
            ("rust_home_mixer_mute_author_weight", "0.0"),
            ("rust_home_mixer_report_weight", "0.0"),
            ("rust_home_mixer_not_dwelled_weight", "0.0"),
            ("rust_home_mixer_reply_weight", "0.0"),
            ("rust_home_mixer_retweet_weight", "0.0"),
            ("rust_home_mixer_photo_expand_weight", "0.0"),
            ("rust_home_mixer_video_open_weight", "0.0"),
            ("rust_home_mixer_click_weight", "0.0"),
            ("rust_home_mixer_open_link_weight", "0.0"),
            ("rust_home_mixer_profile_click_weight", "0.0"),
            ("rust_home_mixer_vqv_weight", "0.0"),
            ("rust_home_mixer_share_weight", "0.0"),
            ("rust_home_mixer_share_via_dm_weight", "0.0"),
            ("rust_home_mixer_share_via_copy_link_weight", "0.0"),
            ("rust_home_mixer_dwell_weight", "0.0"),
            ("rust_home_mixer_quote_weight", "0.0"),
            ("rust_home_mixer_quoted_click_weight", "0.0"),
            ("rust_home_mixer_quoted_vqv_weight", "0.0"),
            ("rust_home_mixer_cont_click_dwell_time_weight", "0.0"),
            (
                "rust_home_mixer_cont_active_secs_5m_residual_norm_weight",
                "0.0",
            ),
            ("rust_home_mixer_follow_author_weight", "0.0"),
            (
                "rust_home_mixer_enable_multiplicative_post_unexplored",
                "false",
            ),
            (
                "rust_home_mixer_post_unexplored_weight_in_network_only",
                "false",
            ),
        ];
        flags.extend_from_slice(extra);
        flags
    }

    fn post_unexplored_candidate(
        post_unexplored: Option<f64>,
        in_network: Option<bool>,
    ) -> PostCandidate {
        PostCandidate {
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.5),
                post_unexplored_score: post_unexplored,
                dwell_time: Some(10.0),
                ..Default::default()
            },
            ..candidate(1, in_network)
        }
    }

    #[test]
    fn multiplicative_post_unexplored_modulates_only_the_dwell_time_term() {
        let query = query_with_flags(&post_unexplored_test_flags(&[
            (
                "rust_home_mixer_enable_multiplicative_post_unexplored",
                "true",
            ),
            (
                "rust_home_mixer_multiplicative_post_unexplored_alpha",
                "0.25",
            ),
            ("rust_home_mixer_post_unexplored_weight", "1.5"),
            ("rust_home_mixer_cont_dwell_time_weight", "0.5"),
        ]));
        let weights = ScoringWeights::from_params(&query.params);

        let with_post_unexplored = post_unexplored_candidate(Some(0.4), Some(true));
        let without_post_unexplored = post_unexplored_candidate(None, Some(true));

        let scored = RankingScorer::compute_weighted_score(&weights, &query, &with_post_unexplored);
        let baseline =
            RankingScorer::compute_weighted_score(&weights, &query, &without_post_unexplored);
        assert!((scored - (1.0 + 5.5 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
        assert!((baseline - (1.0 + 5.0 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
    }

    #[test]
    fn post_unexplored_in_network_only_zeroes_additive_term_for_oon() {
        let query = query_with_flags(&post_unexplored_test_flags(&[
            ("rust_home_mixer_post_unexplored_weight", "1.5"),
            (
                "rust_home_mixer_post_unexplored_weight_in_network_only",
                "true",
            ),
        ]));
        let weights = ScoringWeights::from_params(&query.params);

        let in_network = post_unexplored_candidate(Some(0.4), Some(true));
        let oon = post_unexplored_candidate(Some(0.4), Some(false));

        let in_network_score = RankingScorer::compute_weighted_score(&weights, &query, &in_network);
        let oon_score = RankingScorer::compute_weighted_score(&weights, &query, &oon);
        assert!((in_network_score - (1.0 + 0.6 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
        assert!((oon_score - (1.0 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
    }

    #[test]
    fn post_unexplored_in_network_only_skips_multiplicative_modulation_for_oon() {
        let query = query_with_flags(&post_unexplored_test_flags(&[
            (
                "rust_home_mixer_enable_multiplicative_post_unexplored",
                "true",
            ),
            (
                "rust_home_mixer_multiplicative_post_unexplored_alpha",
                "0.25",
            ),
            (
                "rust_home_mixer_post_unexplored_weight_in_network_only",
                "true",
            ),
            ("rust_home_mixer_cont_dwell_time_weight", "0.5"),
        ]));
        let weights = ScoringWeights::from_params(&query.params);

        let in_network = post_unexplored_candidate(Some(0.4), Some(true));
        let oon = post_unexplored_candidate(Some(0.4), Some(false));

        let in_network_score = RankingScorer::compute_weighted_score(&weights, &query, &in_network);
        let oon_score = RankingScorer::compute_weighted_score(&weights, &query, &oon);
        assert!((in_network_score - (1.0 + 5.5 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
        assert!((oon_score - (1.0 + 5.0 + NEGATIVE_SCORES_OFFSET)).abs() < 1e-9);
    }

    #[test]
    fn bidirectional_weight_boosts_only_mutual_original_posts() {
        let query = query_with_flags(&[
            (
                "rust_home_mixer_bidirectional_follow_reply_weight_boost",
                "3.0",
            ),
            (
                "rust_home_mixer_bidirectional_follow_dwell_weight_boost",
                "2.0",
            ),
        ]);
        let weights = ScoringWeights::from_params(&query.params);
        let base_reply = query.params.get(ReplyWeight);
        let base_dwell = query.params.get(DwellWeight);

        let mutual_original = PostCandidate {
            is_mutual_follow_author: Some(true),
            ..candidate(20, Some(true))
        };
        assert!((weights.reply_weight_for(&mutual_original) - (base_reply + 3.0)).abs() < 1e-9);
        assert!((weights.dwell_weight_for(&mutual_original) - (base_dwell + 2.0)).abs() < 1e-9);

        let mutual_reply = PostCandidate {
            is_mutual_follow_author: Some(true),
            ..candidate_with_reply(20, Some(true))
        };
        assert!((weights.reply_weight_for(&mutual_reply) - base_reply).abs() < 1e-9);
        assert!((weights.dwell_weight_for(&mutual_reply) - base_dwell).abs() < 1e-9);
        let mutual_retweet = PostCandidate {
            is_mutual_follow_author: Some(true),
            ..candidate_with_retweet(20, Some(true))
        };
        assert!((weights.reply_weight_for(&mutual_retweet) - base_reply).abs() < 1e-9);

        let non_mutual = PostCandidate {
            is_mutual_follow_author: Some(false),
            ..candidate(999, Some(true))
        };
        assert!((weights.reply_weight_for(&non_mutual) - base_reply).abs() < 1e-9);
        assert!((weights.dwell_weight_for(&non_mutual) - base_dwell).abs() < 1e-9);
    }

    #[test]
    fn bidirectional_weight_zero_boost_is_noop() {
        let query = query_with_flags(&[
            (
                "rust_home_mixer_bidirectional_follow_reply_weight_boost",
                "0.0",
            ),
            (
                "rust_home_mixer_bidirectional_follow_dwell_weight_boost",
                "0.0",
            ),
        ]);
        let weights = ScoringWeights::from_params(&query.params);
        let base_reply = query.params.get(ReplyWeight);
        let base_dwell = query.params.get(DwellWeight);

        let mutual_original = PostCandidate {
            is_mutual_follow_author: Some(true),
            ..candidate(20, Some(true))
        };
        assert!((weights.reply_weight_for(&mutual_original) - base_reply).abs() < 1e-9);
        assert!((weights.dwell_weight_for(&mutual_original) - base_dwell).abs() < 1e-9);
    }

    #[tokio::test]
    async fn fs_off_does_not_discount_in_network_replies_or_retweets() {
        let scorer = test_scorer();
        let candidates = vec![
            candidate(1, Some(true)),
            candidate_with_reply(2, Some(true)),
            candidate_with_retweet(3, Some(true)),
        ];

        let query = query_with_flags(&[(
            "rust_home_mixer_enable_oon_rescore_for_in_network_replies_retweets",
            "false",
        )]);

        let scored = scorer.score(&query, &candidates).await;

        let original = scored[0].as_ref().unwrap().score.unwrap();
        let reply = scored[1].as_ref().unwrap().score.unwrap();
        let retweet = scored[2].as_ref().unwrap().score.unwrap();

        assert!((reply - original).abs() < 1e-9);
        assert!((retweet - original).abs() < 1e-9);
    }

    #[tokio::test]
    async fn fs_on_applies_oon_discount_to_in_network_replies_and_retweets() {
        let scorer = test_scorer();
        let candidates = vec![
            candidate(1, Some(true)),
            candidate_with_reply(2, Some(true)),
            candidate_with_retweet(3, Some(true)),
            candidate(4, Some(false)),
        ];

        let query = query_with_flags(&[
            (
                "rust_home_mixer_enable_oon_rescore_for_in_network_replies_retweets",
                "true",
            ),
            ("rust_home_mixer_oon_weight_factor", "0.75"),
            ("rust_home_mixer_value_model_mode", "weighted"),
            ("rust_home_mixer_enable_mpn_scoring", "false"),
        ]);

        let scored = scorer.score(&query, &candidates).await;

        let original = scored[0].as_ref().unwrap().score.unwrap();
        let reply = scored[1].as_ref().unwrap().score.unwrap();
        let retweet = scored[2].as_ref().unwrap().score.unwrap();
        let oon = scored[3].as_ref().unwrap().score.unwrap();

        let expected_oon = query.params.get(OonWeightFactor);

        assert!((reply - original * expected_oon).abs() < 1e-9);
        assert!((retweet - original * expected_oon).abs() < 1e-9);
        assert!((oon - original * expected_oon).abs() < 1e-9);
    }

    fn dr_weights() -> DwellRegretWeights {
        DwellRegretWeights {
            alpha_favorite: 1.0,
            alpha_reply: 1.0,
            alpha_retweet: 1.0,
            alpha_quote: 1.0,
            alpha_share: 1.0,
            alpha_share_via_dm: 1.0,
            alpha_share_via_copy_link: 1.0,
            neg_not_interested: -50.0,
            neg_block_author: -50.0,
            neg_mute_author: -80.0,
            neg_report: -1000.0,
            temperature: 1.0,
            dwell_floor: 1.0,
        }
    }

    fn dr_candidate(author_id: u64, scores: PhoenixScores) -> PostCandidate {
        PostCandidate {
            author_id,
            in_network: Some(true),
            phoenix_scores: scores,
            ..Default::default()
        }
    }

    fn dwell_regret_query() -> ScoredPostsQuery {
        let mut query = ScoredPostsQuery::default();
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_value_model_mode".to_string(),
            "dwell_regret_sigmoid",
        );
        results.override_fs(
            "rust_home_mixer_enable_author_diversity".to_string(),
            "false",
        );
        query.params = results.into();
        query
    }

    #[test]
    fn dwell_regret_neutral_post_scores_dwell() {
        let s = PhoenixScores {
            favorite_score: Some(0.1),
            reply_score: Some(0.02),
            dwell_time: Some(20.0),
            ..Default::default()
        };
        let candidates = vec![dr_candidate(1, s.clone()), dr_candidate(2, s)];
        let scores = RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &candidates);
        assert!((scores[0] - 20.0).abs() < 1e-9, "score={}", scores[0]);
        assert!((scores[1] - 20.0).abs() < 1e-9, "score={}", scores[1]);
    }

    #[test]
    fn dwell_regret_above_average_likeable_is_boosted() {
        let a = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.3),
                dwell_time: Some(10.0),
                ..Default::default()
            },
        );
        let b = dr_candidate(
            2,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: Some(10.0),
                ..Default::default()
            },
        );
        let scores = RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &[a, b]);
        assert!((scores[0] - 12.449_186_63).abs() < 1e-6, "a={}", scores[0]);
        assert!((scores[1] - 7.550_813_37).abs() < 1e-6, "b={}", scores[1]);
        assert!(scores[0] > 10.0 && scores[1] < 10.0);
    }

    #[test]
    fn dwell_regret_report_sinks_candidate() {
        let clean = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: Some(10.0),
                ..Default::default()
            },
        );
        let reported = dr_candidate(
            2,
            PhoenixScores {
                favorite_score: Some(0.1),
                report_score: Some(0.01),
                dwell_time: Some(10.0),
                ..Default::default()
            },
        );
        let scores =
            RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &[clean, reported]);
        assert!((scores[0] - 10.0).abs() < 1e-9, "clean={}", scores[0]);
        assert!(scores[1] < 0.01, "reported={}", scores[1]);
        assert!(scores[1] < scores[0]);
    }

    #[test]
    fn dwell_regret_floor_applies_to_low_dwell() {
        let none_dwell = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: None,
                ..Default::default()
            },
        );
        let zero_dwell = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: Some(0.0),
                ..Default::default()
            },
        );
        let none_scores =
            RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &[none_dwell]);
        let zero_scores =
            RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &[zero_dwell]);
        assert!(
            (none_scores[0] - 1.0).abs() < 1e-9,
            "none={}",
            none_scores[0]
        );
        assert!(
            (zero_scores[0] - 1.0).abs() < 1e-9,
            "zero={}",
            zero_scores[0]
        );
    }

    #[test]
    fn dwell_regret_per_request_normalization_amplifies_for_low_engagement_set() {
        let target = || {
            dr_candidate(
                1,
                PhoenixScores {
                    favorite_score: Some(0.03),
                    dwell_time: Some(10.0),
                    ..Default::default()
                },
            )
        };
        let low_set = vec![
            target(),
            dr_candidate(
                2,
                PhoenixScores {
                    favorite_score: Some(0.003),
                    dwell_time: Some(10.0),
                    ..Default::default()
                },
            ),
        ];
        let high_set = vec![
            target(),
            dr_candidate(
                2,
                PhoenixScores {
                    favorite_score: Some(0.09),
                    dwell_time: Some(10.0),
                    ..Default::default()
                },
            ),
        ];
        let low = RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &low_set);
        let high = RankingScorer::compute_dwell_regret_base_scores(&dr_weights(), &high_set);
        assert!(
            low[0] > high[0],
            "low-set target {} should beat high-set target {}",
            low[0],
            high[0]
        );
        assert!(
            low[0] > 10.0,
            "stands out → boosted above dwell: {}",
            low[0]
        );
        assert!(high[0] < 10.0, "below mean → damped: {}", high[0]);
    }

    #[tokio::test]
    async fn score_uses_dwell_regret_mode_when_selected() {
        let scorer = test_scorer();
        let s = PhoenixScores {
            favorite_score: Some(0.1),
            dwell_time: Some(20.0),
            ..Default::default()
        };
        let candidates = vec![dr_candidate(1, s.clone()), dr_candidate(2, s)];
        let scored = scorer.score(&dwell_regret_query(), &candidates).await;
        let s0 = scored[0].as_ref().unwrap().score.unwrap();
        assert!((s0 - 20.0).abs() < 1e-9, "score={s0}");
    }

    #[tokio::test]
    async fn gated_mode_routes_by_gate_decision() {
        let scorer = test_scorer();
        let s = PhoenixScores {
            favorite_score: Some(0.1),
            dwell_time: Some(20.0),
            ..Default::default()
        };
        let candidates = vec![dr_candidate(1, s.clone()), dr_candidate(2, s)];

        let mut query = query_with_flags(&[
            ("rust_home_mixer_value_model_mode", "gated_dwell_regret"),
            ("rust_home_mixer_enable_author_diversity", "false"),
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                GATE_WEIGHTS_ALL_ZERO,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_hysteresis_band", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_threshold", "-1000.0"),
            ("rust_home_mixer_dwell_regret_dwell_floor", "1.0"),
        ]);
        query.user_id = 7;
        let scored = scorer.score(&query, &candidates).await;
        let s0 = scored[0].as_ref().unwrap().score.unwrap();
        assert!(
            (s0 - 20.0).abs() < 1e-9,
            "gate-open should use dwell-regret: {s0}"
        );

        let mut query = query_with_flags(&[
            ("rust_home_mixer_value_model_mode", "gated_dwell_regret"),
            ("rust_home_mixer_enable_author_diversity", "false"),
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                GATE_WEIGHTS_ALL_ZERO,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_hysteresis_band", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_threshold", "1000.0"),
            ("rust_home_mixer_favorite_weight", "1.0"),
            ("rust_home_mixer_cont_dwell_time_weight", "0.0"),
        ]);
        query.user_id = 7;
        let scored = scorer.score(&query, &candidates).await;
        let w0 = scored[0].as_ref().unwrap().weighted_score.unwrap();
        assert!(w0 < 1.0, "gate-closed should use weighted: {w0}");
    }

    #[tokio::test]
    async fn gated_mode_with_invalid_gate_config_falls_back_to_weighted() {
        let scorer = test_scorer();
        let candidate = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: Some(20.0),
                ..Default::default()
            },
        );
        let query = query_with_flags(&[
            ("rust_home_mixer_value_model_mode", "gated_dwell_regret"),
            ("rust_home_mixer_enable_author_diversity", "false"),
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                "seq_len:not_a_number",
            ),
            ("rust_home_mixer_favorite_weight", "1.0"),
            ("rust_home_mixer_cont_dwell_time_weight", "0.0"),
        ]);
        let scored = scorer.score(&query, std::slice::from_ref(&candidate)).await;
        let w0 = scored[0].as_ref().unwrap().weighted_score.unwrap();
        assert!(
            w0 < 1.0,
            "invalid gate config must fall back to weighted: {w0}"
        );
    }

    #[tokio::test]
    async fn weighted_mode_uses_weighted_scorer() {
        let scorer = test_scorer();
        let candidate = dr_candidate(
            1,
            PhoenixScores {
                favorite_score: Some(0.1),
                dwell_time: Some(20.0),
                ..Default::default()
            },
        );
        let scored = scorer
            .score(
                &query_with_flags(&[
                    ("rust_home_mixer_value_model_mode", "weighted"),
                    ("rust_home_mixer_favorite_weight", "1.0"),
                    ("rust_home_mixer_cont_dwell_time_weight", "0.0"),
                ]),
                std::slice::from_ref(&candidate),
            )
            .await;
        let weighted = scored[0].as_ref().unwrap().weighted_score.unwrap();
        assert!(
            weighted < 1.0,
            "weighted-mode score should be small: {weighted}"
        );
    }
}
