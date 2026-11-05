use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use xai_feature_switches::{AuthorRulesEvaluator, Params};
use xai_value_model::{
    compute_value_scores, set_user_video_continuation, weighted_terms, CandidateScoringInputs,
    PhoenixScores, QueryScoringContext, ValueModelWeights, ValueScores,
};
use xai_vm_ranker_proto::{HeadContribution, RankRequest};

use crate::metrics::{
    AUTHOR_EXPLORATION_CANDIDATES, CANDIDATE_SCORE, HEAD_PREDICTION_SUM, SAMPLED_CANDIDATES,
    VALUE_MODEL_FALLBACK, VALUE_MODEL_REQUESTS, VALUE_MODEL_STAGE,
};
use crate::params::*;
use crate::ranking_config::snowflake_creation_ms;
use crate::scoring::{debias, DebiasModels};

const NEW_USER_MIN_FOLLOWING: u32 = 5;
const FALLBACK_WARN_INTERVAL_MS: u64 = 10_000;
const SCORE_METRICS_SAMPLE_RATE: f64 = 0.01;

static LAST_FALLBACK_WARN_MS: AtomicU64 = AtomicU64::new(0);

pub struct ValueModelOutput {
    pub weighted: Vec<f64>,
    pub scores: Vec<f64>,
    pub debiased: bool,
    pub top_heads: Option<Vec<Vec<HeadContribution>>>,
}

struct Fallback {
    reason: &'static str,
    detail: String,
}

impl Fallback {
    fn new(reason: &'static str, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

pub fn compute(
    req: &RankRequest,
    params: Option<&Params>,
    author_rules: Option<&AuthorRulesEvaluator>,
    compute_value_model: bool,
    debias: &DebiasModels,
) -> Option<ValueModelOutput> {
    if !compute_value_model {
        record_request("passthrough", false);
        return None;
    }
    match compute_or_fallback(req, params, author_rules, debias) {
        Ok(output) => {
            record_request("value_model", output.debiased);
            Some(output)
        }
        Err(fallback) => {
            record_request("fallback", false);
            record_fallback(&fallback, req.viewer_id);
            None
        }
    }
}

pub fn weights_from_params(params: &Params, viewer_id: u64) -> ValueModelWeights {
    ValueModelWeights {
        favorite: params.get(FavoriteWeight),
        reply: params.get(ReplyWeight),
        retweet: params.get(RetweetWeight),
        photo_expand: params.get(PhotoExpandWeight),
        video_open: params.get(VideoOpenWeight),
        click: params.get(ClickWeight),
        open_link: params.get(OpenLinkWeight),
        profile_click: params.get(ProfileClickWeight),
        vqv: params.get(VqvWeight),
        share: params.get(ShareWeight),
        share_via_dm: params.get(ShareViaDmWeight),
        share_via_copy_link: params.get(ShareViaCopyLinkWeight),
        dwell: params.get(DwellWeight),
        quote: params.get(QuoteWeight),
        quoted_click: params.get(QuotedClickWeight),
        quoted_vqv: params.get(QuotedVqvWeight),
        follow_author: params.get(FollowAuthorWeight),
        post_unexplored: params.get(PostUnexploredWeight),
        post_unexplored_include_out_of_network: params.get(PostUnexploredIncludeOutOfNetwork),
        post_video: params.get(PostVideoWeight),
        grok_analyze: params.get(GrokAnalyzeWeight),
        not_interested: params.get(NotInterestedWeight),
        block_author: params.get(BlockAuthorWeight),
        mute_author: params.get(MuteAuthorWeight),
        report: params.get(ReportWeight),
        not_dwelled: params.get(NotDwelledWeight),
        cont_dwell_time: params.get(ContDwellTimeWeight),
        cont_click_dwell_time: params.get(ContClickDwellTimeWeight),
        video_continuation: params.get(VideoContinuationWeight),
        user_video_continuation: params.get(UserVideoContinuationWeight),
        profile_visit_secs: params.get(ProfileVisitSecsWeight),
        min_video_duration_ms: params.get(MinVideoDurationMs),
        enable_quoted_vqv_duration_check: params.get(EnableQuotedVqvDurationCheck),
        bidirectional_follow_reply_weight_boost: params.get(BidirectionalFollowReplyWeightBoost),
        bidirectional_follow_dwell_weight_boost: params.get(BidirectionalFollowDwellWeightBoost),
        enable_author_diversity: params.get(EnableAuthorDiversity),
        author_diversity_decay: params.get(AuthorDiversityDecay),
        author_diversity_floor: params.get(AuthorDiversityFloor),
        oon_rescore_in_network_replies_retweets: params
            .get(EnableOonRescoreForInNetworkRepliesRetweets),
        multiplier_pre_offset: params.get(MultiplierPreOffset),
    }
    .perturbed(
        params.get(WeightPerturbationSigma),
        &params.get(WeightPerturbationSalt),
        viewer_id,
    )
}

pub fn scoring_context(req: &RankRequest, params: &Params) -> QueryScoringContext {
    if req.topic_request {
        return QueryScoringContext {
            effective_oon_weight: params.get(TopicOonWeightFactor),
        };
    }
    let now_ms = req
        .viewer
        .as_ref()
        .map_or_else(|| now_ms() as i64, |v| v.now_ms);
    let account_age_secs = snowflake_creation_ms(req.viewer_id)
        .map(|created| now_ms - created)
        .filter(|&age| age >= 0)
        .map(|age| (age / 1000) as u64);
    let is_eligible_new_user = account_age_secs
        .is_some_and(|age| age < params.get(NewUserAgeThresholdSecs))
        && req.viewer_following_count >= NEW_USER_MIN_FOLLOWING;
    QueryScoringContext {
        effective_oon_weight: if is_eligible_new_user {
            params.get(NewUserOonWeightFactor)
        } else {
            params.get(OonWeightFactor)
        },
    }
}

pub fn candidate_inputs(
    req: &RankRequest,
    weights: &ValueModelWeights,
) -> Vec<CandidateScoringInputs> {
    req.candidates
        .iter()
        .map(|c| CandidateScoringInputs::from_rank_candidate(c, weights.min_video_duration_ms))
        .collect()
}

fn compute_or_fallback(
    req: &RankRequest,
    params: Option<&Params>,
    author_rules: Option<&AuthorRulesEvaluator>,
    debias: &DebiasModels,
) -> Result<ValueModelOutput, Fallback> {
    let params = params.ok_or_else(|| Fallback::new("no_config", ""))?;
    let weights = weights_from_params(params, req.viewer_id);
    let ctx = scoring_context(req, params);
    let mut inputs = candidate_inputs(req, &weights);
    if weights.user_video_continuation != 0.0 {
        set_user_video_continuation(&mut inputs);
    }
    if let Some(author_rules) = author_rules {
        set_author_exploration_bonuses(author_rules, &mut inputs);
    }

    let raw = compute_value_scores(&weights, &ctx, &inputs);
    if rand::random_bool(SCORE_METRICS_SAMPLE_RATE) {
        record_score_metrics(&inputs, &raw);
    }
    let cached_weighted = !inputs.is_empty() && inputs.iter().all(|c| c.weighted_score.is_some());
    let stage = if cached_weighted {
        "cached_weighted"
    } else {
        "heads"
    };
    VALUE_MODEL_STAGE.with_label_values(&[stage]).inc();
    if cached_weighted && !req.include_top_heads {
        return Ok(ValueModelOutput {
            weighted: raw.weighted,
            scores: raw.scores,
            debiased: false,
            top_heads: None,
        });
    }

    let recentered = debias::recenter(req, params, debias.user.as_deref(), &inputs);
    let scored_inputs = recentered.as_deref().unwrap_or(&inputs);
    let top_heads = req.include_top_heads.then(|| {
        scored_inputs
            .iter()
            .map(|c| top_heads(&weights, c))
            .collect()
    });
    if cached_weighted || recentered.is_none() {
        return Ok(ValueModelOutput {
            weighted: raw.weighted,
            scores: raw.scores,
            debiased: false,
            top_heads,
        });
    }
    let debiased = compute_value_scores(&weights, &ctx, scored_inputs);
    Ok(ValueModelOutput {
        weighted: debiased.weighted,
        scores: debiased.scores,
        debiased: true,
        top_heads,
    })
}

const TOP_HEADS: usize = 3;
const MIN_HEAD_SHARE: f64 = 0.10;

fn top_heads(
    weights: &ValueModelWeights,
    candidate: &CandidateScoringInputs,
) -> Vec<HeadContribution> {
    let mut terms = weighted_terms(weights, candidate);
    let total: f64 = terms.iter().map(|(_, c)| c.abs()).sum();
    terms.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    terms
        .iter()
        .take(TOP_HEADS)
        .map(|&(head, contribution)| HeadContribution {
            head: head.to_string(),
            contribution,
            share: contribution.abs() / total,
        })
        .take_while(|h| h.share >= MIN_HEAD_SHARE)
        .collect()
}

fn set_author_exploration_bonuses(
    author_rules: &AuthorRulesEvaluator,
    inputs: &mut [CandidateScoringInputs],
) {
    let mut by_author = HashMap::new();
    let mut with_bonus = 0;
    for c in inputs.iter_mut() {
        let bonus = *by_author
            .entry(c.author_id)
            .or_insert_with(|| author_rules.get(c.author_id, AuthorExplorationBonus));
        c.author_exploration_bonus = if bonus.is_finite() { bonus } else { 0.0 };
        with_bonus += u64::from(c.author_exploration_bonus != 0.0);
    }
    AUTHOR_EXPLORATION_CANDIDATES
        .with_label_values(&["non_zero"])
        .inc_by(with_bonus);
    AUTHOR_EXPLORATION_CANDIDATES
        .with_label_values(&["zero"])
        .inc_by(inputs.len() as u64 - with_bonus);
}

fn head_predictions(s: &PhoenixScores) -> [(&'static str, Option<f64>); 27] {
    [
        ("favorite", s.favorite_score),
        ("reply", s.reply_score),
        ("retweet", s.retweet_score),
        ("photo_expand", s.photo_expand_score),
        ("video_open", s.video_open_score),
        ("click", s.click_score),
        ("open_link", s.open_link_score),
        ("profile_click", s.profile_click_score),
        ("vqv", s.vqv_score),
        ("share", s.share_score),
        ("share_via_dm", s.share_via_dm_score),
        ("share_via_copy_link", s.share_via_copy_link_score),
        ("dwell", s.dwell_score),
        ("quote", s.quote_score),
        ("quoted_click", s.quoted_click_score),
        ("quoted_vqv", s.quoted_vqv_score),
        ("dwell_time", s.dwell_time),
        ("click_dwell_time", s.click_dwell_time),
        (
            "home_video_continuation_secs",
            s.home_video_continuation_secs,
        ),
        ("home_profile_visit_secs", s.home_profile_visit_secs),
        ("follow_author", s.follow_author_score),
        ("not_interested", s.not_interested_score),
        ("block_author", s.block_author_score),
        ("mute_author", s.mute_author_score),
        ("report", s.report_score),
        ("not_dwelled", s.not_dwelled_score),
        ("post_unexplored", s.post_unexplored_score),
    ]
}

fn record_score_metrics(inputs: &[CandidateScoringInputs], raw: &ValueScores) {
    let mut candidates = 0;
    let mut prediction = [0.0; 27];
    for c in inputs.iter().filter(|c| c.weighted_score.is_none()) {
        candidates += 1;
        for (i, (_, score)) in head_predictions(&c.phoenix_scores).into_iter().enumerate() {
            prediction[i] += score.unwrap_or(0.0);
        }
    }
    if candidates > 0 {
        SAMPLED_CANDIDATES.inc_by(candidates);
        let heads = head_predictions(&PhoenixScores::default());
        for ((head, _), sum) in heads.iter().zip(prediction) {
            HEAD_PREDICTION_SUM
                .with_label_values(&[head])
                .inc_by(sum.max(0.0));
        }
    }
    for (stage, scores) in [("weighted", &raw.weighted), ("ranked", &raw.scores)] {
        let histogram = CANDIDATE_SCORE.with_label_values(&[stage]);
        for &score in scores {
            histogram.observe(score);
        }
    }
}

fn record_request(mode: &str, debias_applied: bool) {
    VALUE_MODEL_REQUESTS
        .with_label_values(&[mode, if debias_applied { "on" } else { "off" }])
        .inc();
}

fn record_fallback(fallback: &Fallback, viewer_id: u64) {
    VALUE_MODEL_FALLBACK
        .with_label_values(&[fallback.reason])
        .inc();
    let now = now_ms();
    let last = LAST_FALLBACK_WARN_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= FALLBACK_WARN_INTERVAL_MS
        && LAST_FALLBACK_WARN_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        warn!(
            "value model fell back to upstream scores: viewer={viewer_id} reason={} {}",
            fallback.reason, fallback.detail
        );
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
