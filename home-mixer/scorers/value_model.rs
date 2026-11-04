use crate::models::candidate::{PhoenixScores, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::util::candidates_util::{quoted_vqv_eligible, vqv_eligible};
use std::collections::HashMap;
use xai_value_model::{fuse_heads, CandidateScoringInputs, ValueModelWeights};

pub(crate) fn weights_for(query: &ScoredPostsQuery) -> ValueModelWeights {
    let params = &query.params;
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
        post_unexplored_include_out_of_network: false,
        post_video: params.get(PostVideoWeight),
        grok_analyze: params.get(GrokAnalyzeWeight),
        not_interested: params.get(NotInterestedWeight),
        block_author: params.get(BlockAuthorWeight),
        mute_author: params.get(MuteAuthorWeight),
        report: params.get(ReportWeight),
        not_dwelled: params.get(NotDwelledWeight),
        cont_dwell_time: params.get(ContDwellTimeWeight),
        cont_click_dwell_time: params.get(ContClickDwellTimeWeight),
        video_continuation: 0.0,
        user_video_continuation: 0.0,
        profile_visit_secs: 0.0,
        min_video_duration_ms: params.get(MinVideoDurationMs),
        enable_quoted_vqv_duration_check: params.get(EnableQuotedVqvDurationCheck),
        bidirectional_follow_reply_weight_boost: params.get(BidirectionalFollowReplyWeightBoost),
        bidirectional_follow_dwell_weight_boost: params.get(BidirectionalFollowDwellWeightBoost),
        enable_author_diversity: false,
        author_diversity_decay: 1.0,
        author_diversity_floor: 1.0,
        oon_rescore_in_network_replies_retweets: false,
        multiplier_pre_offset: false,
    }
}

pub(crate) fn applied_weights(query: &ScoredPostsQuery) -> HashMap<String, f64> {
    weights_for(query).applied_weights_map()
}

pub(crate) fn scoring_inputs(
    query: &ScoredPostsQuery,
    weights: &ValueModelWeights,
    candidate: &PostCandidate,
) -> CandidateScoringInputs {
    CandidateScoringInputs {
        phoenix_scores: shared_phoenix_scores(&candidate.phoenix_scores),
        author_id: candidate.author_id,
        in_network: candidate.in_network,
        is_reply: candidate.in_reply_to_tweet_id.is_some(),
        is_retweet: candidate.retweeted_tweet_id.is_some(),
        is_mutual_follow_author: candidate.is_mutual_follow_author == Some(true),
        vqv_eligible: vqv_eligible(query, candidate, weights.min_video_duration_ms),
        quoted_vqv_eligible: quoted_vqv_eligible(
            candidate,
            weights.min_video_duration_ms,
            weights.enable_quoted_vqv_duration_check,
        ),
        cold_start_lift_to_rank: candidate.cold_start_lift_to_rank,
        weighted_score: None,
        author_exploration_bonus: 0.0,
        user_video_continuation_secs: None,
    }
}

pub(crate) fn weighted_score(
    query: &ScoredPostsQuery,
    weights: &ValueModelWeights,
    candidate: &PostCandidate,
) -> f64 {
    fuse_heads(weights, &scoring_inputs(query, weights, candidate))
}

fn shared_phoenix_scores(s: &PhoenixScores) -> xai_value_model::PhoenixScores {
    xai_value_model::PhoenixScores {
        favorite_score: s.favorite_score,
        reply_score: s.reply_score,
        retweet_score: s.retweet_score,
        photo_expand_score: s.photo_expand_score,
        video_open_score: s.video_open_score,
        click_score: s.click_score,
        open_link_score: s.open_link_score,
        profile_click_score: s.profile_click_score,
        vqv_score: s.vqv_score,
        share_score: s.share_score,
        share_via_dm_score: s.share_via_dm_score,
        share_via_copy_link_score: s.share_via_copy_link_score,
        dwell_score: s.dwell_score,
        quote_score: s.quote_score,
        quoted_click_score: s.quoted_click_score,
        quoted_vqv_score: s.quoted_vqv_score,
        follow_author_score: s.follow_author_score,
        not_interested_score: s.not_interested_score,
        block_author_score: s.block_author_score,
        mute_author_score: s.mute_author_score,
        report_score: s.report_score,
        not_dwelled_score: s.not_dwelled_score,
        post_unexplored_score: s.post_unexplored_score,
        post_video_score: s.post_video_score,
        grok_analyze_score: s.grok_analyze_score,
        dwell_time: s.dwell_time,
        click_dwell_time: s.click_dwell_time,
        home_video_continuation_secs: s.home_video_continuation_secs,
        home_profile_visit_secs: s.home_profile_visit_secs,
    }
}
