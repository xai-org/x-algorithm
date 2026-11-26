mod blender_comparison_test;
mod following_ad_blender_tests;
mod least_movement_blender_tests;
mod multi_risk_blender_tests;
mod partition_organic_blender_tests;
mod safe_gap_blender_tests;
mod slot_spread_blender_tests;
mod time_gap_blender_tests;

use crate::ads::util::{should_drop_bsr_low, should_drop_handle, should_drop_keyword};
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

fn post(tweet_id: u64, verdict: BrandSafetyVerdict) -> ScoredPost {
    ScoredPost {
        tweet_id,
        author_id: tweet_id,
        brand_safety_verdict: verdict.into(),
        ..Default::default()
    }
}

fn ad(post_id: i64, insert_position: i32, risk: BrandSafetyRiskLevel) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        insert_position,
        ad_adjacency_control: Some(AdAdjacencyControl {
            brand_safety_risk: risk.into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn mixed_feed() -> (Vec<ScoredPost>, Vec<AdIndexInfo>) {
    use BrandSafetyVerdict::*;
    let posts = (1..=35)
        .map(|id| {
            let verdict = match id {
                2 | 15 => LowRisk,
                8 | 22 => MediumRisk,
                12 | 27 => HighRisk,
                _ => Safe,
            };
            let mut post = post(id, verdict);
            if id == 17 {
                post.author_id = 999;
            }
            if id == 6 || id == 20 {
                post.tweet_text = "#tires on sale".to_string();
            }
            post
        })
        .collect();
    let mut ads: Vec<AdIndexInfo> = (0..10)
        .map(|i| {
            let risk = if i == 0 || i == 3 {
                BrandSafetyRiskLevel::BsrLow
            } else {
                BrandSafetyRiskLevel::BsrNormal
            };
            ad(100 + i, 1 + 4 * i as i32, risk)
        })
        .collect();
    ads[1].ad_adjacency_control.as_mut().unwrap().keywords = vec!["tires".to_string()];
    ads[2].ad_adjacency_control.as_mut().unwrap().handles = vec![999];
    (posts, ads)
}

fn unplaceable_ad_feed() -> (Vec<ScoredPost>, Vec<AdIndexInfo>) {
    let mut posts: Vec<ScoredPost> = (1..=12)
        .map(|id| post(id, BrandSafetyVerdict::Safe))
        .collect();
    for post in &mut posts[..11] {
        post.tweet_text = "new tires".to_string();
    }
    posts[2].brand_safety_verdict = BrandSafetyVerdict::LowRisk.into();
    let mut sensitive = ad(100, 1, BrandSafetyRiskLevel::BsrLow);
    sensitive.ad_adjacency_control.as_mut().unwrap().keywords = vec!["tires".to_string()];
    (
        posts,
        vec![sensitive, ad(101, 5, BrandSafetyRiskLevel::BsrNormal)],
    )
}

fn served_ads(items: &[FeedItem]) -> Vec<(usize, i64)> {
    items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| match &item.item {
            Some(feed_item::Item::Ad(ad)) => Some((i, ad.post_id)),
            _ => None,
        })
        .collect()
}

fn organic_moves(items: &[FeedItem]) -> u64 {
    items
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Post(post)) => Some(post.tweet_id),
            _ => None,
        })
        .enumerate()
        .map(|(slot, tweet_id)| (slot as u64 + 1).abs_diff(tweet_id))
        .sum()
}

fn drop_summary(drops: &[DroppedAd]) -> Vec<(u64, AdDropReason, u64)> {
    drops
        .iter()
        .map(|d| (d.tweet_id, d.drop_reason(), d.blocking_tweet_id))
        .collect()
}

fn assert_brand_safe(items: &[FeedItem]) {
    let as_post = |item: Option<&FeedItem>| match item.and_then(|i| i.item.as_ref()) {
        Some(feed_item::Item::Post(post)) => post.clone(),
        _ => panic!("an ad must sit between two posts"),
    };
    for (i, item) in items.iter().enumerate() {
        let Some(feed_item::Item::Ad(ad)) = &item.item else {
            continue;
        };
        let above = as_post(i.checked_sub(1).and_then(|j| items.get(j)));
        let below = as_post(items.get(i + 1));
        for neighbour in [&above, &below] {
            assert!(
                !matches!(
                    neighbour.brand_safety_verdict(),
                    BrandSafetyVerdict::HighRisk | BrandSafetyVerdict::MediumRisk
                ),
                "ad {} next to post {}",
                ad.post_id,
                neighbour.tweet_id
            );
        }
        let (above, below) = (Some(&above), Some(&below));
        assert!(!should_drop_bsr_low(ad, above, below), "ad {}", ad.post_id);
        assert!(!should_drop_handle(ad, above, below), "ad {}", ad.post_id);
        assert!(!should_drop_keyword(ad, above, below), "ad {}", ad.post_id);
    }
}
