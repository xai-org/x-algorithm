use crate::filters::inventory_holdout_filter::InventoryHoldoutFilter;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableFavHoldout;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct FavHoldoutFilter;

impl FavHoldoutFilter {
    const STEPS: &'static [(i64, u32)] = &[
        (10000, 15),
        (5000, 15),
        (3000, 15),
        (2000, 14),
        (1500, 13),
        (1000, 13),
        (750, 12),
        (500, 11),
        (300, 10),
        (200, 9),
        (150, 8),
        (100, 7),
        (75, 7),
        (50, 6),
        (30, 6),
        (20, 5),
        (15, 5),
        (10, 5),
        (7, 4),
        (5, 4),
        (3, 3),
        (2, 3),
        (1, 2),
    ];

    pub(crate) fn holdout_percent(fav_count: Option<i64>) -> u32 {
        let favs = fav_count.unwrap_or(0);
        if favs <= 1 {
            return 0;
        }
        for &(k, pct) in Self::STEPS {
            if favs > k {
                return pct;
            }
        }
        0
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for FavHoldoutFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableFavHoldout)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_id = query.user_id;
        FilterResult::partition(candidates, |c| {
            let percent = Self::holdout_percent(c.fav_count);
            !InventoryHoldoutFilter::is_held_out(c.get_original_tweet_id(), viewer_id, percent)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    fn post(tweet_id: u64, fav_count: Option<i64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            fav_count,
            ..Default::default()
        }
    }

    fn params(enable: bool) -> Params {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_fav_holdout".to_string(),
            if enable { "true" } else { "false" },
        );
        results.into()
    }

    fn query(viewer_id: u64, enable: bool) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id: viewer_id,
            params: params(enable),
            ..Default::default()
        }
    }

    #[test]
    fn percent_is_zero_at_or_below_moe_eligibility() {
        assert_eq!(FavHoldoutFilter::holdout_percent(None), 0);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(0)), 0);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(1)), 0);
    }

    #[test]
    fn percent_follows_tc_table() {
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(2)), 2);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(3)), 3);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(10)), 4);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(11)), 5);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(100)), 7);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(101)), 7);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(1000)), 12);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(1001)), 13);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(5000)), 15);
        assert_eq!(FavHoldoutFilter::holdout_percent(Some(20_000)), 15);
    }

    #[test]
    fn disabled_master_switch_disables_the_filter() {
        assert!(!FavHoldoutFilter.enable(&query(1, false)));
        assert!(FavHoldoutFilter.enable(&query(1, true)));
    }

    #[test]
    fn low_fav_posts_are_never_held_out() {
        let q = query(42, true);
        let candidates = vec![post(1, None), post(2, Some(0)), post(3, Some(1))];
        let result = FavHoldoutFilter.filter(&q, candidates);
        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.kept.len(), 3);
    }

    #[test]
    fn high_fav_holdout_is_approximately_the_table_percent() {
        let n = 200_000u64;
        let viewer = 987_654_321u64;
        let percent = FavHoldoutFilter::holdout_percent(Some(5000));
        assert_eq!(percent, 15);
        let held = (0..n)
            .filter(|&tweet_id| InventoryHoldoutFilter::is_held_out(tweet_id, viewer, percent))
            .count();
        let rate = held as f64 / n as f64;
        assert!(
            (0.14..=0.16).contains(&rate),
            "holdout rate {rate} not within tolerance of 15%"
        );
    }

    #[test]
    fn filter_drops_only_hashed_high_fav_posts() {
        let viewer = 12_345u64;
        let q = query(viewer, true);
        let candidates: Vec<_> = (0..200)
            .map(|i| post(i, Some(if i % 2 == 0 { 5000 } else { 1 })))
            .collect();
        let result = FavHoldoutFilter.filter(&q, candidates);
        assert!(result.removed.iter().all(|c| c.fav_count == Some(5000)));
        assert!(result.kept.iter().any(|c| c.fav_count == Some(5000)));
        assert!(result.kept.iter().any(|c| c.fav_count == Some(1)));
        assert!(!result.removed.is_empty());
    }
}
