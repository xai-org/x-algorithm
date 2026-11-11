use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableInventoryHoldout, InventoryHoldoutOriginalsPercent, InventoryHoldoutRepliesPercent,
    InventoryHoldoutRetweetsPercent,
};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostKind {
    Original,
    Reply,
    Retweet,
}

impl PostKind {
    fn of(candidate: &PostCandidate) -> Self {
        if candidate.retweeted_tweet_id.is_some() {
            PostKind::Retweet
        } else if candidate.in_reply_to_tweet_id.is_some() {
            PostKind::Reply
        } else {
            PostKind::Original
        }
    }
}

pub struct InventoryHoldoutFilter;

impl InventoryHoldoutFilter {
    pub(crate) fn holdout_bucket(post_id: u64, viewer_id: u64) -> u64 {
        let mut z = post_id
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(
                viewer_id
                    .rotate_left(32)
                    .wrapping_mul(0xD1B5_4A32_D192_ED03),
            )
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z % 100
    }

    pub(crate) fn is_held_out(post_id: u64, viewer_id: u64, percent: u32) -> bool {
        Self::holdout_bucket(post_id, viewer_id) < percent as u64
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for InventoryHoldoutFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableInventoryHoldout)
            && (query.params.get(InventoryHoldoutOriginalsPercent) > 0
                || query.params.get(InventoryHoldoutRepliesPercent) > 0
                || query.params.get(InventoryHoldoutRetweetsPercent) > 0)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_id = query.user_id;
        let originals_pct = query.params.get(InventoryHoldoutOriginalsPercent).min(100);
        let replies_pct = query.params.get(InventoryHoldoutRepliesPercent).min(100);
        let retweets_pct = query.params.get(InventoryHoldoutRetweetsPercent).min(100);

        FilterResult::partition(candidates, |c| {
            let percent = match PostKind::of(c) {
                PostKind::Original => originals_pct,
                PostKind::Reply => replies_pct,
                PostKind::Retweet => retweets_pct,
            };
            !Self::is_held_out(c.get_original_tweet_id(), viewer_id, percent)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    fn original(tweet_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            ..Default::default()
        }
    }

    fn reply(tweet_id: u64, in_reply_to: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            in_reply_to_tweet_id: Some(in_reply_to),
            ..Default::default()
        }
    }

    fn retweet(retweet_id: u64, original_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id: retweet_id,
            retweeted_tweet_id: Some(original_id),
            ..Default::default()
        }
    }

    fn params(enable: bool, originals: u32, replies: u32, retweets: u32) -> Params {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_inventory_holdout".to_string(),
            if enable { "true" } else { "false" },
        );
        results.override_fs(
            "rust_home_mixer_inventory_holdout_originals_percent".to_string(),
            &originals.to_string(),
        );
        results.override_fs(
            "rust_home_mixer_inventory_holdout_replies_percent".to_string(),
            &replies.to_string(),
        );
        results.override_fs(
            "rust_home_mixer_inventory_holdout_retweets_percent".to_string(),
            &retweets.to_string(),
        );
        results.into()
    }

    fn query(viewer_id: u64, params: Params) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id: viewer_id,
            params,
            ..Default::default()
        }
    }

    #[test]
    fn post_kind_classification() {
        assert_eq!(PostKind::of(&original(1)), PostKind::Original);
        assert_eq!(PostKind::of(&reply(2, 1)), PostKind::Reply);
        assert_eq!(PostKind::of(&retweet(3, 1)), PostKind::Retweet);
        let quote = PostCandidate {
            tweet_id: 4,
            quoted_tweet_id: Some(99),
            ..Default::default()
        };
        assert_eq!(PostKind::of(&quote), PostKind::Original);
    }

    #[test]
    fn holdout_is_deterministic_for_same_pair() {
        let a = InventoryHoldoutFilter::holdout_bucket(123, 456);
        let b = InventoryHoldoutFilter::holdout_bucket(123, 456);
        assert_eq!(a, b);
    }

    #[test]
    fn different_viewers_get_different_held_out_sets() {
        let held_out_for = |viewer: u64| -> Vec<u64> {
            (0..5_000u64)
                .filter(|&post| InventoryHoldoutFilter::is_held_out(post, viewer, 10))
                .collect()
        };
        let viewer_a = held_out_for(11);
        let viewer_b = held_out_for(22);
        assert_ne!(viewer_a, viewer_b);
        assert!(!viewer_a.is_empty() && !viewer_b.is_empty());
    }

    #[test]
    fn holdout_rate_is_approximately_the_configured_percent() {
        let n = 200_000u64;
        let viewer = 987_654_321u64;
        let held = (0..n)
            .filter(|&post| InventoryHoldoutFilter::is_held_out(post, viewer, 1))
            .count();
        let rate = held as f64 / n as f64;
        assert!(
            (0.007..=0.013).contains(&rate),
            "holdout rate {rate} not within tolerance of 1%"
        );
    }

    #[test]
    fn percent_zero_holds_out_nothing_and_hundred_holds_out_everything() {
        let viewer = 42u64;
        let none = (0..1_000u64)
            .filter(|&post| InventoryHoldoutFilter::is_held_out(post, viewer, 0))
            .count();
        assert_eq!(none, 0);
        let all = (0..1_000u64)
            .filter(|&post| InventoryHoldoutFilter::is_held_out(post, viewer, 100))
            .count();
        assert_eq!(all, 1_000);
    }

    #[test]
    fn retweet_shares_holdout_decision_with_its_original_at_equal_rate() {
        let viewer = 555u64;
        let original_id = 9_000u64;
        let rt = retweet(1u64, original_id);
        assert_eq!(
            InventoryHoldoutFilter::is_held_out(
                original(original_id).get_original_tweet_id(),
                viewer,
                50
            ),
            InventoryHoldoutFilter::is_held_out(rt.get_original_tweet_id(), viewer, 50),
        );
    }

    #[test]
    fn filter_targets_only_the_enabled_kind() {
        let viewer = 12_345u64;
        let q = query(viewer, params(true, 100, 0, 0));
        let candidates = vec![
            original(1),
            reply(2, 100),
            retweet(3, 200),
            original(4),
            reply(5, 101),
        ];
        let result = InventoryHoldoutFilter.filter(&q, candidates);

        assert_eq!(result.removed.len(), 2);
        assert!(result
            .removed
            .iter()
            .all(|c| PostKind::of(c) == PostKind::Original));
        assert_eq!(result.kept.len(), 3);
        assert!(result
            .kept
            .iter()
            .all(|c| PostKind::of(c) != PostKind::Original));
    }

    #[test]
    fn filter_can_target_replies_and_retweets_independently() {
        let viewer = 7u64;
        let q = query(viewer, params(true, 0, 100, 100));
        let candidates = vec![original(1), reply(2, 100), retweet(3, 200)];
        let result = InventoryHoldoutFilter.filter(&q, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(PostKind::of(&result.kept[0]), PostKind::Original);
        assert_eq!(result.removed.len(), 2);
    }

    #[test]
    fn disabled_master_switch_disables_the_filter() {
        let q = query(1, params(false, 100, 100, 100));
        assert!(!InventoryHoldoutFilter.enable(&q));

        let q_zero = query(1, params(true, 0, 0, 0));
        assert!(!InventoryHoldoutFilter.enable(&q_zero));

        let q_on = query(1, params(true, 0, 1, 0));
        assert!(InventoryHoldoutFilter.enable(&q_on));
    }
}
