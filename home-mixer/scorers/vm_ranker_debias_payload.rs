use crate::models::candidate::PostCandidate;
use crate::models::content_features;
use crate::models::query::ScoredPostsQuery;
use prost::Message;
use xai_recsys_proto::{ContentFeatures, ProductSurface};
use xai_vm_ranker_proto as pb;
use xai_vm_ranker_proto::experiment::{CandidatePayload, PostTopics, RequestPayload};

pub(crate) fn request_payload(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
    debias: bool,
    history_topics: bool,
    post_topics: bool,
) -> Vec<u8> {
    let mut payload = RequestPayload::default();
    if debias {
        payload.engine_time_hint_ms = phoenix_scored_at_ms(candidates);
        payload.user_sequence_arrow_ipc =
            query.columnar_scoring_sequence.clone().unwrap_or_default();
        payload.product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        } as i32;
        payload.client_app_id = i64::from(query.client_app_id);
    }
    if history_topics {
        payload.history_topics = query.history_topics.clone();
    }
    if post_topics {
        payload.recent_views = query.recent_views.clone();
        payload.recent_view_topics = query.recent_view_topics.clone();
    }
    payload.encode_to_vec()
}

pub(crate) fn candidate_payload(
    candidate: &PostCandidate,
    debias: bool,
    pacing: bool,
    topics: Option<&PostTopics>,
) -> Vec<u8> {
    let mut payload = CandidatePayload::default();
    if debias {
        payload.content_features =
            Some(content_features_proto(&content_features::build(candidate)));
        payload.quoted_content_features =
            content_features::build_quoted(candidate).map(|f| content_features_proto(&f));
        payload.quoted_tweet_id = candidate.quoted_tweet_id.unwrap_or(0);
    }
    if pacing {
        payload.view_count = candidate.view_count;
        payload.author_followers_count = candidate
            .author_followers_count
            .and_then(|n| u64::try_from(n).ok());
    }
    payload.topics = topics.cloned();
    payload.encode_to_vec()
}

fn phoenix_scored_at_ms(candidates: &[PostCandidate]) -> Option<i64> {
    candidates
        .iter()
        .filter_map(|c| c.last_scored_at_ms)
        .max()
        .map(|ms| ms as i64)
}

fn content_features_proto(f: &ContentFeatures) -> pb::experiment::ContentFeatures {
    pb::experiment::ContentFeatures {
        has_video: f.has_video,
        max_video_duration_ms: f.max_video_duration_ms,
        has_photo: f.has_photo,
        media_count: f.media_count,
        weighted_text_len: f.weighted_text_len,
        newline_count: f.newline_count,
        has_url: f.has_url,
    }
}
