use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;

const MAX_FOLLOWERS_THRESHOLD: i64 = 10_000;

pub fn get_related_post_ids(candidate: &PostCandidate) -> Vec<u64> {
    let mut ids = vec![candidate.tweet_id];
    ids.extend(candidate.retweeted_tweet_id);
    ids.extend(candidate.in_reply_to_tweet_id);
    ids
}

pub fn related_post_ids_iter(candidate: &PostCandidate) -> impl Iterator<Item = u64> {
    std::iter::once(candidate.tweet_id)
        .chain(candidate.retweeted_tweet_id)
        .chain(candidate.in_reply_to_tweet_id)
}

/// PR-F5 attribution key: RTs and quotes of the same original share one K bucket.
/// IncludeQuotes is fixed true (HoE) — quoted_tweet_id always participates.
pub fn source_key(candidate: &PostCandidate) -> u64 {
    if let Some(rt) = candidate.retweeted_tweet_id {
        return rt;
    }
    if let Some(q) = candidate.quoted_tweet_id {
        return q;
    }
    candidate.tweet_id
}

pub fn vqv_weight(
    query: &ScoredPostsQuery,
    candidate: &PostCandidate,
    min_video_duration_ms: i32,
    vqv_weight_value: f64,
) -> f64 {
    let exceeds_followers = query
        .user_features
        .follower_count
        .map(|count| count >= MAX_FOLLOWERS_THRESHOLD)
        .unwrap_or(false);

    if !exceeds_followers
        && candidate
            .min_video_duration_ms
            .is_some_and(|ms| ms > min_video_duration_ms)
    {
        vqv_weight_value
    } else {
        0.0
    }
}

pub fn quoted_vqv_weight(
    candidate: &PostCandidate,
    min_video_duration_ms: i32,
    quoted_vqv_weight_value: f64,
    enable_duration_check: bool,
) -> f64 {
    if !enable_duration_check {
        return quoted_vqv_weight_value;
    }

    if candidate
        .quoted_video_duration_ms
        .is_some_and(|ms| ms > min_video_duration_ms)
    {
        quoted_vqv_weight_value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_vqv_returns_weight_when_check_disabled() {
        let candidate = PostCandidate {
            quoted_video_duration_ms: None,
            ..Default::default()
        };
        assert!((quoted_vqv_weight(&candidate, 10_000, 0.5, false) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn quoted_vqv_returns_weight_when_duration_exceeds_threshold() {
        let candidate = PostCandidate {
            quoted_video_duration_ms: Some(15_000),
            ..Default::default()
        };
        assert!((quoted_vqv_weight(&candidate, 10_000, 0.5, true) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn quoted_vqv_returns_zero_when_duration_below_threshold() {
        let candidate = PostCandidate {
            quoted_video_duration_ms: Some(5_000),
            ..Default::default()
        };
        assert!((quoted_vqv_weight(&candidate, 10_000, 0.5, true)).abs() < 1e-9);
    }

    #[test]
    fn quoted_vqv_returns_zero_when_duration_equals_threshold() {
        let candidate = PostCandidate {
            quoted_video_duration_ms: Some(10_000),
            ..Default::default()
        };
        assert!((quoted_vqv_weight(&candidate, 10_000, 0.5, true)).abs() < 1e-9);
    }

    #[test]
    fn quoted_vqv_returns_zero_when_duration_is_none() {
        let candidate = PostCandidate {
            quoted_video_duration_ms: None,
            ..Default::default()
        };
        assert!((quoted_vqv_weight(&candidate, 10_000, 0.5, true)).abs() < 1e-9);
    }

    #[test]
    fn source_key_prefers_retweeted_then_quoted_then_tweet() {
        let rt = PostCandidate {
            tweet_id: 1,
            retweeted_tweet_id: Some(100),
            quoted_tweet_id: Some(200),
            ..Default::default()
        };
        assert_eq!(source_key(&rt), 100);

        let quote = PostCandidate {
            tweet_id: 2,
            quoted_tweet_id: Some(100),
            ..Default::default()
        };
        assert_eq!(source_key(&quote), 100);

        let orig = PostCandidate {
            tweet_id: 100,
            ..Default::default()
        };
        assert_eq!(source_key(&orig), 100);
    }
}
