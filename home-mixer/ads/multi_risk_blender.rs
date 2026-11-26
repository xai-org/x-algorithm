use super::drops::{adjacency_drop, dropped_ad, truncate_recording_drops};
use super::health_metrics::{serving_limitation, HealthMetrics, SlotRejections};
use super::util::*;
use super::AdsBlender;
use crate::params::RESULT_SIZE;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

const METRICS: HealthMetrics = HealthMetrics {
    enforcement: "MultiRisk.enforcement",
    slot_outcome: "MultiRisk.slot_outcome",
    serving_limitation: "MultiRisk.serving_limitation",
};

pub struct MultiRiskAdsBlender;

impl AdsBlender for MultiRiskAdsBlender {
    fn blend_inner_with_drops(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        blend_with_drops(scored_posts, ads, MIN_POSTS_FOR_ADS, drops)
    }
}

#[cfg(test)]
pub(crate) fn blend_impl(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
) -> Vec<FeedItem> {
    blend_with_drops(scored_posts, ads, min_posts, &mut Vec::new())
}

pub(crate) fn blend_with_drops(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let n = scored_posts.len();

    if ads.is_empty() || n < min_posts {
        METRICS.emit_serving_limitation(if ads.is_empty() {
            "no_ads"
        } else {
            "too_few_posts"
        });
        drops.extend(
            ads.iter()
                .map(|ad| dropped_ad(ad, AdDropReason::TooFewPosts)),
        );
        return posts_to_feed_items(scored_posts);
    }

    let spacing = compute_spacing(&ads);
    let spacing_cap = n
        .saturating_sub(1)
        .checked_div(spacing.requested)
        .unwrap_or(0);

    let safe_count = scored_posts
        .iter()
        .filter(|p| !is_medium_risk(p) && !is_high_risk(p))
        .count();
    let medium_count = scored_posts.iter().filter(|p| is_medium_risk(p)).count();
    let max_safe_slots = safe_count / 2;
    let max_medium_slots = medium_count / 2;
    let num_ads = ads.len();
    let safe_budget = num_ads.min(spacing_cap).min(max_safe_slots);
    METRICS.emit_serving_limitation(serving_limitation(num_ads, spacing_cap, max_safe_slots));

    let any_bsr_high = ads.iter().any(is_bsr_high_ad);
    if safe_budget == 0 && !(any_bsr_high && max_medium_slots > 0 && spacing_cap > 0) {
        let reason = if spacing_cap == 0 {
            AdDropReason::SpacingCap
        } else {
            AdDropReason::NoEligibleNeighbours
        };
        drops.extend(ads.iter().map(|ad| dropped_ad(ad, reason)));
        return posts_to_feed_items(scored_posts);
    }

    let mut safe: Vec<ScoredPost> = Vec::new();
    let mut medium: Vec<ScoredPost> = Vec::new();
    let mut high: Vec<ScoredPost> = Vec::new();
    for post in scored_posts {
        if is_high_risk(&post) {
            high.push(post);
        } else if is_medium_risk(&post) {
            medium.push(post);
        } else {
            safe.push(post);
        }
    }

    let num_safe = safe.len();
    let group_size = if safe_budget > 0 {
        num_safe / safe_budget
    } else {
        0
    };

    let mut safe_opts: Vec<Option<ScoredPost>> = safe.into_iter().map(Some).collect();
    let mut medium_opts: Vec<Option<ScoredPost>> = medium.into_iter().map(Some).collect();
    let mut triples: Vec<(AdIndexInfo, ScoredPost, ScoredPost)> = Vec::new();

    let mut slot_tokens: SlotTokens = None;
    let mut medium_slot_tokens: SlotTokens = None;

    let mut bsr_ok: u64 = 0;
    let mut rejections = SlotRejections::default();
    let mut slot_rejections = SlotRejections::default();

    let mut safe_group_idx = 0;
    let mut medium_pair_idx = 0;

    for ad in ads {
        if triples.len() >= spacing_cap {
            drops.push(dropped_ad(&ad, AdDropReason::SpacingCap));
            continue;
        }

        if is_bsr_high_ad(&ad) && medium_pair_idx < max_medium_slots {
            let start = medium_pair_idx * 2;
            let above_ref = medium_opts[start].as_ref();
            let below_ref = medium_opts[start + 1].as_ref();
            match handle_block(&ad, above_ref, below_ref)
                .or_else(|| keyword_block(&ad, above_ref, below_ref, &mut medium_slot_tokens))
            {
                None => {
                    let above = medium_opts[start].take().unwrap();
                    let below = medium_opts[start + 1].take().unwrap();
                    triples.push((ad, above, below));
                    medium_pair_idx += 1;
                    medium_slot_tokens = None;
                    slot_rejections = SlotRejections::default();
                    continue;
                }
                Some(block) if safe_group_idx >= safe_budget || group_size == 0 => {
                    rejections.count(&block.reason);
                    slot_rejections.count(&block.reason);
                    drops.push(adjacency_drop(&ad, block));
                    continue;
                }
                Some(_) => {}
            }
        }

        if safe_group_idx >= safe_budget || group_size == 0 {
            drops.push(dropped_ad(&ad, AdDropReason::NoEligibleNeighbours));
            continue;
        }
        let group_start = safe_group_idx * group_size;
        let above_ref = safe_opts[group_start].as_ref();
        let below_ref = safe_opts[group_start + 1].as_ref();

        let low_risk = low_risk_block(&ad, above_ref, below_ref);
        if is_bsr_low_ad(&ad) && low_risk.is_none() {
            bsr_ok += 1;
        }
        let block = low_risk
            .or_else(|| handle_block(&ad, above_ref, below_ref))
            .or_else(|| keyword_block(&ad, above_ref, below_ref, &mut slot_tokens));
        if let Some(block) = block {
            rejections.count(&block.reason);
            slot_rejections.count(&block.reason);
            drops.push(adjacency_drop(&ad, block));
            continue;
        }

        let above = safe_opts[group_start].take().unwrap();
        let below = safe_opts[group_start + 1].take().unwrap();
        triples.push((ad, above, below));
        safe_group_idx += 1;
        slot_tokens = None;
        slot_rejections = SlotRejections::default();
    }

    let placed_ads = triples.len();
    METRICS.emit_enforcement(
        rejections.bsr_low,
        bsr_ok,
        rejections.handle,
        rejections.keyword,
    );
    let offered_slots = num_ads
        .min(spacing_cap)
        .min(max_safe_slots.saturating_add(medium_pair_idx));
    METRICS.emit_slot_outcome(
        placed_ads as u64,
        offered_slots.saturating_sub(placed_ads) as u64,
        &slot_rejections,
    );

    if placed_ads == 0 {
        let mut all_posts: Vec<ScoredPost> = safe_opts.into_iter().flatten().collect();
        all_posts.extend(medium_opts.into_iter().flatten());
        all_posts.extend(high);
        all_posts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return posts_to_feed_items(all_posts);
    }

    let mut filler: Vec<ScoredPost> =
        Vec::with_capacity(num_safe + medium_count + high.len() - 2 * placed_ads);
    filler.extend(safe_opts.into_iter().flatten());
    filler.extend(medium_opts.into_iter().flatten());
    filler.extend(high);
    filler.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let inter_ad_gaps = placed_ads;
    let filler_per_gap = filler.len() / inter_ad_gaps;
    let remainder = filler.len() % inter_ad_gaps;
    let mut filler_iter = filler.into_iter();

    let mut items: Vec<FeedItem> = Vec::with_capacity(n + placed_ads);

    for (i, (ad, above, below)) in triples.into_iter().enumerate() {
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(above)),
        });
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(ad)),
        });
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(below)),
        });

        let count = filler_per_gap + if i >= inter_ad_gaps - remainder { 1 } else { 0 };
        for _ in 0..count {
            if let Some(post) = filler_iter.next() {
                items.push(FeedItem {
                    position: 0,
                    item: Some(feed_item::Item::Post(post)),
                });
            }
        }
    }

    truncate_recording_drops(&mut items, RESULT_SIZE, drops);
    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }

    items
}
