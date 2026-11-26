use crate::ads::drops::{blocked_ad, dropped_ad};
use crate::ads::util::{has_avoid, should_drop_bsr_low, should_drop_handle, should_drop_keyword};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAdAdjacentServedFilter;
use std::collections::HashSet;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_home_mixer_proto::{feed_item, FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;
use xai_stats_receiver::global_stats_receiver;

const ACTION_METRIC: &str = "AdAdjacentServed.action";
const OUTCOME_METRIC: &str = "AdAdjacentServed.ad_outcome";
const AD_STATUS_METRIC: &str = "AdAdjacentServed.ad_status";
const FEED_STATUS_METRIC: &str = "AdAdjacentServed.feed_status";
const FEED_POSTS_METRIC: &str = "AdAdjacentServed.feed_posts";
const SWAP_FAILURE_METRIC: &str = "AdAdjacentServed.swap_failure";
const UNSERVED_BUCKET_METRIC: &str = "AdAdjacentServed.unserved_posts_bucket";

pub struct AdAdjacentServedFilter;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Above,
    Below,
}

impl Filter<ScoredPostsQuery, FeedItem> for AdAdjacentServedFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        if !query.params.get(EnableAdAdjacentServedFilter) {
            return false;
        }
        if query.is_bottom_request {
            return false;
        }
        let ctx = RequestContext::parse(&query.request_context);
        !matches!(
            ctx,
            RequestContext::PullToRefresh
                | RequestContext::Launch
                | RequestContext::Signup
                | RequestContext::ForegroundTruncate
                | RequestContext::Gap
        )
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<FeedItem>,
    ) -> FilterResult<FeedItem> {
        if candidates.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }
        let served: HashSet<u64> = query.served_ids.iter().copied().collect();

        let (served_posts, unserved_posts) = count_served_posts(&candidates, &served);
        emit_feed_posts(served_posts, unserved_posts);
        emit_unserved_bucket(unserved_posts);

        if served.is_empty() {
            emit_ad_status(candidates.iter().filter_map(as_ad).count() as u64, 0);
            emit_feed_status(false);
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        let total_ads = candidates.iter().filter_map(as_ad).count() as u64;
        let mut items = candidates;
        let mut removed: Vec<FeedItem> = Vec::new();
        let mut drops: Vec<DroppedAd> = Vec::new();
        let mut insert_count: u64 = 0;
        let mut drop_count: u64 = 0;
        let mut acted_ad_ids: HashSet<i64> = HashSet::new();

        while let Some((ad_idx, side)) = first_served_adjacency(&items, &served) {
            if let Some(ad) = as_ad(&items[ad_idx]) {
                acted_ad_ids.insert(ad.post_id);
            }
            match find_swap_in_candidate(&items, ad_idx, side, &served) {
                Some(c_idx) => {
                    insert_eligible_neighbour(&mut items, ad_idx, side, c_idx);
                    insert_count += 1;
                }
                None => {
                    emit_swap_failure(classify_swap_failure(&items, ad_idx, &served));
                    let neighbour_idx = match side {
                        Side::Above => ad_idx - 1,
                        Side::Below => ad_idx + 1,
                    };
                    if let (Some(ad), Some(post)) =
                        (as_ad(&items[ad_idx]), as_post(&items[neighbour_idx]))
                    {
                        drops.push(blocked_ad(ad, AdDropReason::ServedNeighbour, post.tweet_id));
                    }
                    removed.push(items.remove(ad_idx));
                    drop_count += 1;
                }
            }
        }

        if drop_count > 0 {
            while items.last().is_some_and(is_ad) {
                removed.push(items.pop().unwrap());
            }
            while items.first().is_some_and(is_ad) {
                removed.push(items.remove(0));
            }
            let trimmed = removed[drop_count as usize..].iter().filter_map(as_ad);
            drops.extend(trimmed.map(|ad| dropped_ad(ad, AdDropReason::Truncated)));
            query.ad_drops.extend(drops);
        }

        if insert_count > 0 || !removed.is_empty() {
            for (i, item) in items.iter_mut().enumerate() {
                item.position = i as i32;
            }
        }

        if !acted_ad_ids.is_empty() {
            let surviving: HashSet<i64> =
                items.iter().filter_map(as_ad).map(|a| a.post_id).collect();
            let kept_ads = acted_ad_ids
                .iter()
                .filter(|id| surviving.contains(*id))
                .count() as u64;
            let dropped_ads = acted_ad_ids.len() as u64 - kept_ads;
            emit_ad_outcome(kept_ads, dropped_ads);
        }

        let acted_count = acted_ad_ids.len() as u64;
        emit_ad_status(total_ads.saturating_sub(acted_count), acted_count);
        emit_feed_status(acted_count > 0);

        emit_action_metrics(insert_count, drop_count);
        FilterResult {
            kept: items,
            removed,
        }
    }
}

fn is_ad(item: &FeedItem) -> bool {
    matches!(item.item, Some(feed_item::Item::Ad(_)))
}

fn as_ad(item: &FeedItem) -> Option<&AdIndexInfo> {
    match &item.item {
        Some(feed_item::Item::Ad(ad)) => Some(ad),
        _ => None,
    }
}

fn as_post(item: &FeedItem) -> Option<&ScoredPost> {
    match &item.item {
        Some(feed_item::Item::Post(post)) => Some(post),
        _ => None,
    }
}

fn post_tweet_id(item: &FeedItem) -> Option<u64> {
    as_post(item).map(|post| post.tweet_id)
}

fn first_served_adjacency(items: &[FeedItem], served: &HashSet<u64>) -> Option<(usize, Side)> {
    for (i, item) in items.iter().enumerate() {
        if !is_ad(item) {
            continue;
        }
        let above_served = i
            .checked_sub(1)
            .and_then(|j| post_tweet_id(&items[j]))
            .is_some_and(|id| served.contains(&id));
        if above_served {
            return Some((i, Side::Above));
        }
        let below_served = items
            .get(i + 1)
            .and_then(post_tweet_id)
            .is_some_and(|id| served.contains(&id));
        if below_served {
            return Some((i, Side::Below));
        }
    }
    None
}

fn is_eligible_neighbour(ad: &AdIndexInfo, post: &ScoredPost, served: &HashSet<u64>) -> bool {
    !served.contains(&post.tweet_id)
        && !has_avoid(post)
        && !should_drop_bsr_low(ad, Some(post), None)
        && !should_drop_handle(ad, Some(post), None)
        && !should_drop_keyword(ad, Some(post), None)
}

fn is_ad_neighbour(items: &[FeedItem], idx: usize) -> bool {
    let above_is_ad = idx.checked_sub(1).is_some_and(|j| is_ad(&items[j]));
    let below_is_ad = items.get(idx + 1).is_some_and(is_ad);
    above_is_ad || below_is_ad
}

fn find_swap_in_candidate(
    items: &[FeedItem],
    ad_idx: usize,
    side: Side,
    served: &HashSet<u64>,
) -> Option<usize> {
    let ad = as_ad(&items[ad_idx])?;
    let len = items.len();

    let left: Vec<usize> = if ad_idx >= 2 {
        (0..=ad_idx - 2).rev().collect()
    } else {
        Vec::new()
    };
    let right: Vec<usize> = ((ad_idx + 2)..len).collect();

    let order: Vec<usize> = match side {
        Side::Above => left.into_iter().chain(right).collect(),
        Side::Below => right.into_iter().chain(left).collect(),
    };

    order.into_iter().find(|&c| {
        !is_ad_neighbour(items, c)
            && as_post(&items[c]).is_some_and(|p| is_eligible_neighbour(ad, p, served))
    })
}

fn insert_eligible_neighbour(items: &mut Vec<FeedItem>, ad_idx: usize, side: Side, c_idx: usize) {
    let cand = items.remove(c_idx);
    let insert_at = match (c_idx < ad_idx, side) {
        (true, Side::Above) => ad_idx - 1,
        (true, Side::Below) => ad_idx,
        (false, Side::Above) => ad_idx,
        (false, Side::Below) => ad_idx + 1,
    };
    items.insert(insert_at, cand);
}

fn classify_swap_failure(items: &[FeedItem], ad_idx: usize, served: &HashSet<u64>) -> &'static str {
    let len = items.len();
    let mut any_unserved = false;
    let mut any_unserved_non_neighbour = false;
    for c in 0..len {
        if c == ad_idx || c + 1 == ad_idx || c == ad_idx + 1 {
            continue;
        }
        let Some(post) = as_post(&items[c]) else {
            continue;
        };
        if served.contains(&post.tweet_id) {
            continue;
        }
        any_unserved = true;
        if !is_ad_neighbour(items, c) {
            any_unserved_non_neighbour = true;
        }
    }
    if !any_unserved {
        "no_unserved"
    } else if !any_unserved_non_neighbour {
        "all_ad_neighbours"
    } else {
        "brand_unsafe"
    }
}

fn emit_action_metrics(inserts: u64, drops: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if inserts > 0 {
        receiver.incr(ACTION_METRIC, &[("action", "insert")], inserts);
    }
    if drops > 0 {
        receiver.incr(ACTION_METRIC, &[("action", "fallback_drop")], drops);
    }
}

fn emit_swap_failure(reason: &'static str) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    receiver.incr(SWAP_FAILURE_METRIC, &[("reason", reason)], 1);
}

fn emit_ad_outcome(kept: u64, dropped: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if kept > 0 {
        receiver.incr(OUTCOME_METRIC, &[("outcome", "kept")], kept);
    }
    if dropped > 0 {
        receiver.incr(OUTCOME_METRIC, &[("outcome", "dropped")], dropped);
    }
}

fn emit_ad_status(clean: u64, needs_action: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if clean > 0 {
        receiver.incr(AD_STATUS_METRIC, &[("status", "clean")], clean);
    }
    if needs_action > 0 {
        receiver.incr(
            AD_STATUS_METRIC,
            &[("status", "needs_action")],
            needs_action,
        );
    }
}

fn emit_feed_status(needs_action: bool) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    let status = if needs_action {
        "needs_action"
    } else {
        "clean"
    };
    receiver.incr(FEED_STATUS_METRIC, &[("status", status)], 1);
}

fn count_served_posts(items: &[FeedItem], served: &HashSet<u64>) -> (u64, u64) {
    let mut served_count = 0u64;
    let mut unserved_count = 0u64;
    for item in items {
        if let Some(id) = post_tweet_id(item) {
            if served.contains(&id) {
                served_count += 1;
            } else {
                unserved_count += 1;
            }
        }
    }
    (served_count, unserved_count)
}

fn emit_feed_posts(served: u64, unserved: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if served > 0 {
        receiver.incr(FEED_POSTS_METRIC, &[("status", "served")], served);
    }
    if unserved > 0 {
        receiver.incr(FEED_POSTS_METRIC, &[("status", "unserved")], unserved);
    }
}

fn unserved_bucket(n: u64) -> &'static str {
    match n {
        0 => "00",
        1 => "01",
        2 => "02",
        3 => "03",
        4 => "04",
        5 => "05",
        6..=8 => "06-08",
        9..=12 => "09-12",
        13..=16 => "13-16",
        17..=20 => "17-20",
        21..=25 => "21-25",
        _ => "26+",
    }
}

fn emit_unserved_bucket(unserved: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    receiver.incr(
        UNSERVED_BUCKET_METRIC,
        &[("bucket", unserved_bucket(unserved))],
        1,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_home_mixer_proto::{BrandSafetyVerdict, ScoredPost};
    use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

    fn post(tweet_id: u64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                ..Default::default()
            })),
        }
    }

    fn post_with_verdict(tweet_id: u64, verdict: BrandSafetyVerdict) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                brand_safety_verdict: verdict.into(),
                ..Default::default()
            })),
        }
    }

    fn post_by_author(tweet_id: u64, author_id: u64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                author_id,
                ..Default::default()
            })),
        }
    }

    fn post_with_text(tweet_id: u64, text: &str) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                tweet_text: text.to_string(),
                ..Default::default()
            })),
        }
    }

    #[test]
    fn count_served_posts_splits_posts_and_ignores_ads() {
        let feed = vec![post(1), ad(10), post(2), post(3), ad(11)];
        let served: HashSet<u64> = [1, 3].into_iter().collect();
        let (served_count, unserved_count) = count_served_posts(&feed, &served);
        assert_eq!(served_count, 2);
        assert_eq!(unserved_count, 1);
    }

    #[test]
    fn unserved_bucket_boundaries() {
        assert_eq!(unserved_bucket(0), "00");
        assert_eq!(unserved_bucket(1), "01");
        assert_eq!(unserved_bucket(5), "05");
        assert_eq!(unserved_bucket(6), "06-08");
        assert_eq!(unserved_bucket(8), "06-08");
        assert_eq!(unserved_bucket(9), "09-12");
        assert_eq!(unserved_bucket(12), "09-12");
        assert_eq!(unserved_bucket(16), "13-16");
        assert_eq!(unserved_bucket(20), "17-20");
        assert_eq!(unserved_bucket(25), "21-25");
        assert_eq!(unserved_bucket(26), "26+");
    }

    #[test]
    fn classify_swap_failure_no_unserved() {
        let feed = vec![ad(10), post(1), post(2), post(3)];
        let served: HashSet<u64> = [1, 2, 3].into_iter().collect();
        assert_eq!(classify_swap_failure(&feed, 0, &served), "no_unserved");
    }

    #[test]
    fn classify_swap_failure_all_ad_neighbours() {
        let feed = vec![ad(10), post(1), ad(11), post(2)];
        let served: HashSet<u64> = [1].into_iter().collect();
        assert_eq!(
            classify_swap_failure(&feed, 0, &served),
            "all_ad_neighbours"
        );
    }

    fn ad(post_id: i64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo {
                post_id,
                ..Default::default()
            })),
        }
    }

    fn sensitive_ad(post_id: i64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo {
                post_id,
                ad_adjacency_control: Some(AdAdjacencyControl {
                    brand_safety_risk: BrandSafetyRiskLevel::BsrLow.into(),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        }
    }

    fn ad_excluding_handle(post_id: i64, handle: i64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo {
                post_id,
                ad_adjacency_control: Some(AdAdjacencyControl {
                    handles: vec![handle],
                    ..Default::default()
                }),
                ..Default::default()
            })),
        }
    }

    fn ad_excluding_keyword(post_id: i64, keyword: &str) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo {
                post_id,
                ad_adjacency_control: Some(AdAdjacencyControl {
                    keywords: vec![keyword.to_string()],
                    ..Default::default()
                }),
                ..Default::default()
            })),
        }
    }

    fn query_with_served(served_ids: Vec<u64>) -> ScoredPostsQuery {
        ScoredPostsQuery {
            request_context: "foreground".to_string(),
            served_ids,
            ..Default::default()
        }
    }

    fn kept_post_ids(items: &[FeedItem]) -> Vec<u64> {
        items.iter().filter_map(post_tweet_id).collect()
    }

    fn ad_neighbours_clean(items: &[FeedItem], served: &HashSet<u64>) -> bool {
        for (i, item) in items.iter().enumerate() {
            if !is_ad(item) {
                continue;
            }
            let above_bad = i
                .checked_sub(1)
                .and_then(|j| post_tweet_id(&items[j]))
                .is_some_and(|id| served.contains(&id));
            let below_bad = items
                .get(i + 1)
                .and_then(post_tweet_id)
                .is_some_and(|id| served.contains(&id));
            if above_bad || below_bad {
                return false;
            }
        }
        true
    }

    fn ad_above_id(items: &[FeedItem]) -> Option<u64> {
        let i = items.iter().position(is_ad)?;
        post_tweet_id(&items[i.checked_sub(1)?])
    }

    fn ad_below_id(items: &[FeedItem]) -> Option<u64> {
        let i = items.iter().position(is_ad)?;
        post_tweet_id(items.get(i + 1)?)
    }

    #[test]
    fn no_op_when_served_ids_empty() {
        let query = query_with_served(vec![]);
        let candidates = vec![post(1), ad(100), post(2)];

        let result = AdAdjacentServedFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 3);
        assert!(result.removed.is_empty());
    }

    #[test]
    fn no_op_when_no_neighbor_in_served() {
        let query = query_with_served(vec![999]);
        let candidates = vec![post(1), ad(100), post(2)];

        let result = AdAdjacentServedFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 3);
        assert!(result.removed.is_empty());
        assert_eq!(kept_post_ids(&result.kept), vec![1, 2]);
    }

    #[test]
    fn inserts_from_near_side_when_above_served() {
        let served = vec![3];
        let query = query_with_served(served.clone());
        let candidates = vec![post(1), post(2), post(3), ad(100), post(4)];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty(), "insert should not drop anything");
        assert_eq!(kept_post_ids(&result.kept), vec![1, 3, 2, 4]);
        assert_eq!(ad_above_id(&result.kept), Some(2));
        assert_eq!(ad_below_id(&result.kept), Some(4));
        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
    }

    #[test]
    fn inserts_from_other_side_when_near_side_exhausted() {
        let served = vec![1];
        let query = query_with_served(served.clone());
        let candidates = vec![post(1), ad(100), post(2), post(3)];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(kept_post_ids(&result.kept), vec![1, 3, 2]);
        assert_eq!(ad_above_id(&result.kept), Some(3));
        assert_eq!(ad_below_id(&result.kept), Some(2));
        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
    }

    #[test]
    fn is_ad_neighbour_flags_posts_next_to_ads() {
        let feed = vec![post(1), ad(10), post(2), post(3)];
        assert!(is_ad_neighbour(&feed, 0));
        assert!(is_ad_neighbour(&feed, 2));
        assert!(!is_ad_neighbour(&feed, 3));
    }

    #[test]
    fn widens_search_beyond_adjacent_ad_to_save_ad() {
        let served = vec![1];
        let query = query_with_served(served.clone());
        let candidates = vec![
            post(1),
            ad(100),
            post(2),
            ad(101),
            post(3),
            post(4),
            post(5),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(
            result.removed.is_empty(),
            "widened search should save the ad instead of dropping it"
        );
        assert_eq!(result.kept.iter().filter(|it| is_ad(it)).count(), 2);
        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
        let mut ids = kept_post_ids(&result.kept);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4, 5], "no organic dropped");
    }

    #[test]
    fn skips_medium_risk_candidate_and_uses_safe_one() {
        let served = vec![1];
        let query = query_with_served(served.clone());
        let candidates = vec![
            post(1),
            ad(100),
            post(2),
            post_with_verdict(3, BrandSafetyVerdict::MediumRisk),
            post(4),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(ad_above_id(&result.kept), Some(4));
        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
    }

    #[test]
    fn sensitive_ad_skips_low_risk_candidate() {
        let query = query_with_served(vec![1]);
        let candidates = vec![
            post(1),
            sensitive_ad(100),
            post(2),
            post_with_verdict(3, BrandSafetyVerdict::LowRisk),
            post(4),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(ad_above_id(&result.kept), Some(4));
    }

    #[test]
    fn skips_excluded_handle_candidate() {
        let query = query_with_served(vec![1]);
        let candidates = vec![
            post(1),
            ad_excluding_handle(100, 7),
            post(2),
            post_by_author(3, 7),
            post(4),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(ad_above_id(&result.kept), Some(4));
    }

    #[test]
    fn skips_excluded_keyword_candidate() {
        let query = query_with_served(vec![1]);
        let candidates = vec![
            post(1),
            ad_excluding_keyword(100, "casino"),
            post(2),
            post_with_text(3, "Big casino night downtown"),
            post(4),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(ad_above_id(&result.kept), Some(4));
    }

    #[test]
    fn falls_back_to_drop_ad_only_when_no_eligible_candidate() {
        let query = query_with_served(vec![1]);
        let candidates = vec![
            post(1),
            ad(100),
            post(2),
            post_with_verdict(3, BrandSafetyVerdict::MediumRisk),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 1, "only the ad is dropped");
        assert!(
            result.removed.iter().all(is_ad),
            "fallback must never drop an organic post"
        );
        assert_eq!(kept_post_ids(&result.kept), vec![1, 2, 3]);
        let drops = query.ad_drops.read(<[DroppedAd]>::to_vec);
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].drop_reason(), AdDropReason::ServedNeighbour);
        assert_eq!(drops[0].blocking_tweet_id, 1);
    }

    #[test]
    fn never_drops_organic_even_when_both_sides_served() {
        let query = query_with_served(vec![1, 2]);
        let candidates = vec![
            post(1),
            ad(100),
            post(2),
            post_with_verdict(3, BrandSafetyVerdict::MediumRisk),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 1);
        assert!(result.removed.iter().all(is_ad));
        let mut ids = kept_post_ids(&result.kept);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3], "no organic dropped");
    }

    #[test]
    fn rewrites_positions_after_insert() {
        let query = query_with_served(vec![3]);
        let mut candidates = vec![post(1), post(2), post(3), ad(100), post(4)];
        for (i, item) in candidates.iter_mut().enumerate() {
            item.position = i as i32;
        }

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        let positions: Vec<i32> = result.kept.iter().map(|i| i.position).collect();
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn handles_multiple_ads_independently() {
        let served = vec![1, 4];
        let query = query_with_served(served.clone());
        let candidates = vec![
            post(1),
            ad(100),
            post(2),
            post(3),
            post(4),
            ad(200),
            post(5),
            post(6),
        ];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        assert!(result.removed.is_empty());
        assert_eq!(result.kept.iter().filter(|it| is_ad(it)).count(), 2);
        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
        let mut ids = kept_post_ids(&result.kept);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn fixes_both_sides_of_one_ad() {
        let served = vec![2, 3];
        let query = query_with_served(served.clone());
        let candidates = vec![post(1), post(2), ad(100), post(3), post(4)];

        let result = AdAdjacentServedFilter.filter(&query, candidates);

        let served_set: HashSet<u64> = served.into_iter().collect();
        assert!(ad_neighbours_clean(&result.kept, &served_set));
        let mut ids = kept_post_ids(&result.kept);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    fn params_with_filter_on() -> xai_feature_switches::Params {
        let fs = xai_feature_switches::FeatureSwitches::load_string(
            r#"
ad_adjacent_test_feature:
  description: "Force-enable AdAdjacentServedFilter for unit tests."
  owner: "test"
  defaults:
    rust_home_mixer_enable_ad_adjacent_served_filter: true
"#,
        )
        .expect("test FS YAML must parse");
        let recipient = xai_feature_switches::SimpleRecipient::default();
        fs.match_recipient(&recipient).into()
    }

    #[test]
    fn enable_skips_bottom_request() {
        let q = ScoredPostsQuery {
            params: params_with_filter_on(),
            is_bottom_request: true,
            request_context: "foreground".to_string(),
            ..Default::default()
        };
        assert!(!AdAdjacentServedFilter.enable(&q));
    }

    #[test]
    fn enable_skips_pull_to_refresh() {
        let q = ScoredPostsQuery {
            params: params_with_filter_on(),
            request_context: "ptr".to_string(),
            ..Default::default()
        };
        assert!(!AdAdjacentServedFilter.enable(&q));
    }

    #[test]
    fn enable_skips_all_denylist_contexts() {
        for ctx in ["ptr", "launch", "signup", "foreground_truncate", "gap"] {
            let q = ScoredPostsQuery {
                params: params_with_filter_on(),
                request_context: ctx.to_string(),
                ..Default::default()
            };
            assert!(
                !AdAdjacentServedFilter.enable(&q),
                "context {ctx} must be in the denylist"
            );
        }
    }

    #[test]
    fn enable_runs_for_blue_pill_top_request() {
        let q = ScoredPostsQuery {
            params: params_with_filter_on(),
            is_top_request: true,
            request_context: "auto".to_string(),
            ..Default::default()
        };
        assert!(AdAdjacentServedFilter.enable(&q));
    }

    #[test]
    fn enable_runs_for_silent_polling() {
        let q = ScoredPostsQuery {
            params: params_with_filter_on(),
            is_polling: true,
            request_context: "polling".to_string(),
            ..Default::default()
        };
        assert!(AdAdjacentServedFilter.enable(&q));
    }

    #[test]
    fn enable_runs_for_default_foreground() {
        let q = ScoredPostsQuery {
            params: params_with_filter_on(),
            request_context: "foreground".to_string(),
            ..Default::default()
        };
        assert!(AdAdjacentServedFilter.enable(&q));
    }

    fn params_with_filter_off() -> xai_feature_switches::Params {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let recipient = xai_feature_switches::SimpleRecipient::default();
        let mut results = fs.match_recipient(&recipient);
        results.override_fs(
            "rust_home_mixer_enable_ad_adjacent_served_filter".to_string(),
            "false",
        );
        results.into()
    }

    #[test]
    fn disable_when_fs_off() {
        let q = ScoredPostsQuery {
            params: params_with_filter_off(),
            is_top_request: true,
            request_context: "foreground".to_string(),
            ..Default::default()
        };
        assert!(!AdAdjacentServedFilter.enable(&q));
    }
}
