use crate::models::candidate::{PhoenixScores, PostCandidate};
use crate::models::fs_recipient::FsRecipientInputs;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::scorers::vm_ranker_debias_payload::{candidate_payload, request_payload};
use xai_vm_ranker_proto as pb;

const DPP_VALUE_MODEL_ID: &str = "dpp";

pub(crate) struct RequestShape {
    debias: bool,
    pacing: bool,
    history_topics: bool,
}

impl RequestShape {
    pub(crate) fn from_query(query: &ScoredPostsQuery) -> Self {
        Self {
            debias: (!query.has_cached_posts || query.is_under_the_hood_request)
                && query.params.get(VMRankerSendDebiasInputs),
            pacing: query.params.get(VMRankerSendPacingInputs),
            history_topics: query.params.get(VMRankerSendHistoryTopics)
                && query.history_topics.is_some(),
        }
    }

    pub(crate) fn build(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        local: &[PostCandidate],
    ) -> pb::RankRequest {
        let proto_candidates = candidates
            .iter()
            .zip(local)
            .map(|(c, local)| self.candidate(c, local))
            .collect();
        let mut request = pb::RankRequest {
            viewer_id: query.user_id,
            candidates: proto_candidates,
            value_model_id: DPP_VALUE_MODEL_ID.to_string(),
            compute_value_model: true,
            viewer: query.fs_recipient_inputs.as_ref().map(viewer_context_proto),
            topic_request: query.is_topic_request(),
            include_top_heads: query.is_under_the_hood_request,
            ..Default::default()
        };
        if self.debias || self.history_topics {
            request.experiment_payload =
                request_payload(query, candidates, self.debias, self.history_topics).into();
        }
        request
    }

    fn candidate(&self, c: &PostCandidate, local: &PostCandidate) -> pb::RankCandidate {
        let mut out = pb::RankCandidate {
            tweet_id: c.tweet_id,
            author_id: c.author_id,
            retweeted_tweet_id: c.retweeted_tweet_id.unwrap_or(0),
            score: local.score,
            in_network: c.in_network.unwrap_or(false),
            is_retweet: c.retweeted_tweet_id.is_some(),
            is_reply: c.in_reply_to_tweet_id.is_some(),
            is_mutual_follow_author: c.is_mutual_follow_author == Some(true),
            cold_start_lift_to_rank: local.cold_start_lift_to_rank,
            min_video_duration_ms: c.min_video_duration_ms,
            phoenix_scores: Some(phoenix_scores_proto(&c.phoenix_scores)),
            weighted_score: c.weighted_score,
            semantic_ids: c.semantic_ids.clone().unwrap_or_default(),
            ..Default::default()
        };
        if self.debias || self.pacing {
            out.experiment_payload = candidate_payload(c, self.debias, self.pacing).into();
        }
        out
    }
}

fn viewer_context_proto(inputs: &FsRecipientInputs) -> pb::ViewerContext {
    pb::ViewerContext {
        user_id: inputs.user_id,
        country_code: inputs.country_code.clone(),
        language_code: inputs.language_code.clone(),
        client_app_id: inputs.client_app_id,
        client_version: inputs.client_version.clone(),
        user_roles: inputs.user_roles.clone(),
        datacenter: inputs.datacenter.clone(),
        has_phone_number: inputs.has_phone_number,
        resurrection_time_ms: inputs.resurrection_time_ms,
        product: inputs.product.clone(),
        now_ms: inputs.now_ms,
        fs_overrides: inputs.fs_overrides.clone(),
    }
}

fn phoenix_scores_proto(s: &PhoenixScores) -> pb::PhoenixScores {
    pb::PhoenixScores {
        favorite_score: s.favorite_score,
        reply_score: s.reply_score,
        retweet_score: s.retweet_score,
        photo_expand_score: s.photo_expand_score,
        click_score: s.click_score,
        profile_click_score: s.profile_click_score,
        vqv_score: s.vqv_score,
        share_score: s.share_score,
        share_via_dm_score: s.share_via_dm_score,
        share_via_copy_link_score: s.share_via_copy_link_score,
        dwell_score: s.dwell_score,
        quote_score: s.quote_score,
        quoted_click_score: s.quoted_click_score,
        follow_author_score: s.follow_author_score,
        not_interested_score: s.not_interested_score,
        block_author_score: s.block_author_score,
        mute_author_score: s.mute_author_score,
        report_score: s.report_score,
        dwell_time: s.dwell_time,
        click_dwell_time: s.click_dwell_time,
        not_dwelled_score: s.not_dwelled_score,
        video_open_score: s.video_open_score,
        open_link_score: s.open_link_score,
        quoted_vqv_score: s.quoted_vqv_score,
        post_unexplored_score: s.post_unexplored_score,
        post_video_score: s.post_video_score,
        grok_analyze_score: s.grok_analyze_score,
        home_video_continuation_secs: s.home_video_continuation_secs,
        home_profile_visit_secs: s.home_profile_visit_secs,
    }
}
