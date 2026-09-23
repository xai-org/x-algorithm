use crate::models::candidate::{PhoenixScores, PostCandidate, SlateContext};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::scorers::author_cold_start::AuthorColdStart;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::scorer::Scorer;

pub(crate) const WEIGHTED_VALUE_MODEL_MODE: &str = "weighted";

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
    enable_cdwell_on_impr: bool,
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
        let enable_cdwell_on_impr = params.get(EnableCdwellOnImpr);
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

        let mut weights = Self {
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
            enable_cdwell_on_impr,
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
            negative_sum: 0.0,
            total_sum: 0.0,
            min_video_duration_ms,
            enable_quoted_vqv_duration_check,
            bidirectional_follow_reply_weight_boost,
            bidirectional_follow_dwell_weight_boost,
        };
        weights.recompute_sums();
        weights
    }

    fn recompute_sums(&mut self) {
        // Negative action weights are configured as negative floats (e.g., -43.2).
        // Asserting non-positive magnitude prevents sign inversion misconfigurations.
        debug_assert!(
            self.not_interested <= 0.0
                && self.block_author <= 0.0
                && self.mute_author <= 0.0
                && self.report <= 0.0
                && self.not_dwelled <= 0.0,
            "Negative action weights must be non-positive values"
        );
        let positive_sum = self.favorite
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
            + if self.enable_multiplicative_post_unexplored {
                0.0
            } else {
                self.post_unexplored
            };
        self.negative_sum = -(self.not_interested
            + self.block_author
            + self.mute_author
            + self.report
            + self.not_dwelled);
        self.total_sum = positive_sum + self.negative_sum;
    }

    pub(crate) fn perturbed(mut self, query: &ScoredPostsQuery) -> Self {
        let sigma = query.params.get(WeightPerturbationSigma);
        if sigma <= 0.0 {
            return self;
        }
        let salt = query.params.get(WeightPerturbationSalt);
        for (head, weight) in self.weights_mut() {
            *weight *= (sigma * perturbation_sign(&salt, query.user_id, head)).exp();
        }
        self.recompute_sums();
        self
    }

    fn weights_mut(&mut self) -> [(&'static str, &mut f64); 26] {
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
            (
                "active_secs_5m_residual_norm",
                &mut self.cont_active_secs_5m_residual_norm,
            ),
            ("follow_author", &mut self.follow_author),
            ("post_unexplored", &mut self.post_unexplored),
            ("not_interested", &mut self.not_interested),
            ("block_author", &mut self.block_author),
            ("mute_author", &mut self.mute_author),
            ("report", &mut self.report),
            ("not_dwelled", &mut self.not_dwelled),
        ]
    }
}

pub(crate) fn perturbation_sign(salt: &str, user_id: u64, head: &str) -> f64 {
    let digest = md5::compute(format!("{salt}:{user_id}:{head}"));
    if digest[0] & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

impl ScoringWeights {
    fn post_unexplored_active_for(&self, candidate: &PostCandidate) -> bool {
        !self.post_unexplored_in_network_only || candidate.in_network == Some(true)
    }

    /// Determines if a post candidate is eligible for the mutual follow boost.
    /// Excludes replies (`in_reply_to_tweet_id`) and retweets (`retweeted_tweet_id`).
    /// Note: Quote tweets are intentionally eligible because they contain new original commentary
    /// authored by the mutual-follow user.
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

    fn click_dwell_term(&self, scores: &PhoenixScores) -> Option<f64> {
        if !self.enable_cdwell_on_impr {
            return scores.click_dwell_time;
        }
        match (scores.click_dwell_time, scores.click_score) {
            (Some(cd), Some(click)) => Some(cd * click),
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
                    "gate.cdwell_on_impr",
                    self.enable_cdwell_on_impr as u8 as f64,
                ),
            ]
            .map(|(k, v)| (k.to_string(), v)),
        )
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
                weights.click_dwell_term(scores),
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

    pub(crate) fn unoffset_score(weighted_score: f64, w: &ScoringWeights) -> f64 {
        if w.total_sum == 0.0 {
            weighted_score
        } else if weighted_score < NEGATIVE_SCORES_OFFSET {
            weighted_score / NEGATIVE_SCORES_OFFSET * w.total_sum - w.negative_sum
        } else {
            weighted_score - NEGATIVE_SCORES_OFFSET
        }
    }

    pub(crate) fn reuses_cached_weighted_score(query: &ScoredPostsQuery) -> bool {
        query.has_cached_posts && query.params.get(CachedPostsReuseWeightedScore)
    }

    fn diversity_multiplier(decay_factor: f64, floor: f64, exponent: f64) -> f64 {
        (1.0 - floor) * decay_factor.powf(exponent) + floor
    }

    fn author_pool_counts(candidates: &[PostCandidate], pre_diversity_scores: &[f64]) -> Vec<u32> {
        let mut indexed: Vec<(usize, f64)> = pre_diversity_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let mut counts = vec![0u32; candidates.len()];
        let mut author_counts: FxHashMap<u64, u32> = FxHashMap::default();
        for (idx, _) in indexed {
            let author_id = candidates[idx].author_id;
            let k = author_counts.get(&author_id).copied().unwrap_or(0);
            counts[idx] = k;
            author_counts.insert(author_id, k + 1);
        }
        counts
    }

    fn served_slate_contexts(candidates: &[PostCandidate]) -> Option<Vec<SlateContext>> {
        candidates.iter().map(|c| c.served_slate_context).collect()
    }

    fn stored_slate_contexts(candidates: &[PostCandidate]) -> Option<Vec<SlateContext>> {
        candidates.iter().map(|c| c.slate_context).collect()
    }

    fn author_diversity_multipliers(query: &ScoredPostsQuery, counts: &[u32]) -> Vec<f64> {
        let decay_factor = query.params.get(AuthorDiversityDecay);
        let floor = query.params.get(AuthorDiversityFloor);

        counts
            .iter()
            .map(|&k| Self::diversity_multiplier(decay_factor, floor, f64::from(k)))
            .collect()
    }

    fn apply_author_diversity(
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        pre_diversity_scores: &[f64],
    ) -> Vec<f64> {
        let counts = Self::author_pool_counts(candidates, pre_diversity_scores);
        let multipliers = Self::author_diversity_multipliers(query, &counts);
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
            query.params.get(NewUserOonWeightFactor)
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
        let weights = ScoringWeights::from_params(&query.params).perturbed(query);
        let enable_author_diversity = query.params.get(EnableAuthorDiversity);

        let reuse_cached = Self::reuses_cached_weighted_score(query);
        let weighted_scores: Vec<f64> = candidates
            .iter()
            .map(|c| match c.weighted_score.filter(|_| reuse_cached) {
                Some(weighted) => weighted,
                None => Self::compute_weighted_score(&weights, query, c),
            })
            .collect();

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

        let persisted_contexts: Option<Vec<SlateContext>> = Self::served_slate_contexts(candidates)
            .or_else(|| {
                query
                    .has_cached_posts
                    .then(|| Self::stored_slate_contexts(candidates))
                    .flatten()
            });

        if query.params.get(MultiplierPreOffset) {
            let diversity_multipliers: Vec<f64> = if enable_author_diversity {
                let counts = Self::author_pool_counts(candidates, &weighted_scores);
                Self::author_diversity_multipliers(query, &counts)
            } else {
                vec![1.0; candidates.len()]
            };
            let scores: Vec<f64> = weighted_scores
                .iter()
                .enumerate()
                .map(|(i, &weighted)| {
                    let mut m = diversity_multipliers[i];
                    if oon_applies(&candidates[i]) {
                        m *= effective_oon;
                    }
                    let net = Self::unoffset_score(weighted, &weights);
                    let scaled = if net >= 0.0 { m * net } else { net };
                    Self::offset_score(scaled, &weights)
                })
                .collect();
            let cold_start = self
                .author_cold_start
                .apply_with_decisions(query, candidates, &scores);
            return weighted_scores
                .iter()
                .zip(&cold_start.scores)
                .enumerate()
                .map(|(i, (&weighted, &score))| {
                    Ok(PostCandidate {
                        weighted_score: Some(weighted),
                        score: Some(score),
                        author_policy_zeroed: cold_start.author_policy_zeroed[i],
                        cold_start_lift_to_rank: cold_start.lift_to_rank(i),
                        slate_context: persisted_contexts.as_ref().map(|contexts| contexts[i]),
                        ..Default::default()
                    })
                })
                .collect();
        }

        let cold_start =
            self.author_cold_start
                .apply_with_decisions(query, candidates, &weighted_scores);
        let adjusted_scores = cold_start.scores.clone();

        let diversity_adjusted = if enable_author_diversity {
            Self::apply_author_diversity(query, candidates, &adjusted_scores)
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
                    author_policy_zeroed: cold_start.author_policy_zeroed[i],
                    cold_start_lift_to_rank: cold_start.lift_to_rank(i),
                    slate_context: persisted_contexts.as_ref().map(|contexts| contexts[i]),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
        candidate.score = scored.score;
        candidate.author_policy_zeroed = scored.author_policy_zeroed;
        candidate.cold_start_lift_to_rank = scored.cold_start_lift_to_rank;
        candidate.slate_context = scored.slate_context;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::author_rules::AuthorRulesEvaluator;
    use std::sync::Arc;

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

        let query = query_with_flags(&[("rust_home_mixer_oon_weight_factor", "0.75")]);
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

    #[tokio::test]
    async fn new_user_applies_new_user_oon_weight() {
        let scorer = test_scorer();
        let candidates = vec![candidate(1, Some(true)), candidate(2, Some(false))];

        let mut query = query_with_flags(&[
            ("rust_home_mixer_new_user_age_threshold_secs", "3600"),
            ("rust_home_mixer_new_user_oon_weight_factor", "0.5"),
            ("rust_home_mixer_oon_weight_factor", "0.75"),
            (
                "rust_home_mixer_enable_oon_rescore_for_in_network_replies_retweets",
                "false",
            ),
        ]);
        query.user_id =
            xai_candidate_pipeline::component_library::utils::current_time_to_id() as u64;
        query.user_features.followed_user_ids = vec![1, 2, 3, 4, 5];

        let scored = scorer.score(&query, &candidates).await;
        let inn = scored[0].as_ref().unwrap().score.unwrap();
        let oon = scored[1].as_ref().unwrap().score.unwrap();
        assert!((oon - inn * 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn new_user_oon_weight_skipped_when_threshold_zero() {
        let scorer = test_scorer();
        let candidates = vec![candidate(1, Some(true)), candidate(2, Some(false))];

        let mut query = query_with_flags(&[
            ("rust_home_mixer_new_user_age_threshold_secs", "0"),
            ("rust_home_mixer_new_user_oon_weight_factor", "0.5"),
            ("rust_home_mixer_oon_weight_factor", "0.75"),
            (
                "rust_home_mixer_enable_oon_rescore_for_in_network_replies_retweets",
                "false",
            ),
        ]);
        query.user_id =
            xai_candidate_pipeline::component_library::utils::current_time_to_id() as u64;
        query.user_features.followed_user_ids = vec![1, 2, 3, 4, 5];

        let scored = scorer.score(&query, &candidates).await;
        let inn = scored[0].as_ref().unwrap().score.unwrap();
        let oon = scored[1].as_ref().unwrap().score.unwrap();
        assert!((oon - inn * 0.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn new_user_oon_weight_skipped_below_min_following() {
        let scorer = test_scorer();
        let candidates = vec![candidate(1, Some(true)), candidate(2, Some(false))];

        let mut query = query_with_flags(&[
            ("rust_home_mixer_new_user_age_threshold_secs", "3600"),
            ("rust_home_mixer_new_user_oon_weight_factor", "0.5"),
            ("rust_home_mixer_oon_weight_factor", "0.75"),
            (
                "rust_home_mixer_enable_oon_rescore_for_in_network_replies_retweets",
                "false",
            ),
        ]);
        query.user_id =
            xai_candidate_pipeline::component_library::utils::current_time_to_id() as u64;
        query.user_features.followed_user_ids = vec![1, 2, 3, 4];

        let scored = scorer.score(&query, &candidates).await;
        let inn = scored[0].as_ref().unwrap().score.unwrap();
        let oon = scored[1].as_ref().unwrap().score.unwrap();
        assert!((oon - inn * 0.75).abs() < 1e-9);
    }
}
