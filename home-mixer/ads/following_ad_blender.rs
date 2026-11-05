use super::drops::{adjacency_drop, dropped_ad, truncate_recording_drops};
use super::util::{
    handle_block, keyword_block, low_risk_block, posts_to_feed_items, Block, MIN_POSTS_FOR_ADS,
};
use super::AdsBlender;
use crate::params::FOLLOWING_MAX_RESULT_SIZE;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

const DEFAULT_SPACING: usize = 3;
const MIN_REQUESTED_GAP: usize = 3;
const SPACING_PROBE_ADS: usize = 4;

pub struct FollowingAdBlender {
    pub(crate) result_size: usize,
}

impl Default for FollowingAdBlender {
    fn default() -> Self {
        Self {
            result_size: FOLLOWING_MAX_RESULT_SIZE,
        }
    }
}

impl AdsBlender for FollowingAdBlender {
    fn blend_inner_with_drops(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        blend_with_drops(
            scored_posts,
            ads,
            MIN_POSTS_FOR_ADS,
            self.result_size,
            drops,
        )
    }
}

pub(crate) fn compute_spacing(ads: &[AdIndexInfo]) -> usize {
    if ads.len() < 2 {
        return DEFAULT_SPACING;
    }

    let mut positions: Vec<i32> = ads
        .iter()
        .take(SPACING_PROBE_ADS)
        .map(|a| a.insert_position)
        .collect();
    positions.sort_unstable();

    let min_diff = positions
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .filter(|&d| d > 0)
        .min();

    match min_diff {
        Some(requested) if requested >= MIN_REQUESTED_GAP => requested,
        _ => DEFAULT_SPACING,
    }
}

pub(crate) fn compute_first_position(ads: &[AdIndexInfo], spacing: usize) -> usize {
    let first = ads
        .first()
        .map(|a| a.insert_position.max(0) as usize)
        .unwrap_or(1);
    first.clamp(1, spacing)
}

fn same_conversation_unit(a: &ScoredPost, b: &ScoredPost) -> bool {
    a.ancestors.contains(&b.tweet_id)
        || b.ancestors.contains(&a.tweet_id)
        || a.ancestors.iter().any(|id| b.ancestors.contains(id))
}

pub(crate) fn blend_units(posts: &[ScoredPost]) -> Vec<std::ops::Range<usize>> {
    if posts.is_empty() {
        return Vec::new();
    }
    let mut units = Vec::new();
    let mut start = 0;
    for i in 1..posts.len() {
        if !same_conversation_unit(&posts[i - 1], &posts[i]) {
            units.push(start..i);
            start = i;
        }
    }
    units.push(start..posts.len());
    units
}

fn is_medium_risk(post: &ScoredPost) -> bool {
    !matches!(
        post.brand_safety_verdict(),
        BrandSafetyVerdict::Safe | BrandSafetyVerdict::LowRisk
    )
}

fn unit_medium_risk(unit: &[ScoredPost]) -> bool {
    unit.iter().any(is_medium_risk)
}

fn group_block(ad: &AdIndexInfo, above: &[ScoredPost], below: &[ScoredPost]) -> Option<Block> {
    let posts = || above.iter().chain(below);
    posts()
        .find_map(|p| handle_block(ad, Some(p), None))
        .or_else(|| posts().find_map(|p| keyword_block(ad, Some(p), None, &mut None)))
        .or_else(|| posts().find_map(|p| low_risk_block(ad, Some(p), None)))
}

#[cfg(test)]
pub(crate) fn blend_impl(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    result_size: usize,
) -> Vec<FeedItem> {
    blend_with_drops(scored_posts, ads, min_posts, result_size, &mut Vec::new())
}

pub(crate) fn blend_with_drops(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    result_size: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let spacing = compute_spacing(&ads);
    blend_with_spacing(scored_posts, ads, min_posts, result_size, spacing, drops)
}

pub(crate) fn blend_with_spacing(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    result_size: usize,
    spacing: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let units = blend_units(&scored_posts);
    let n_units = units.len();

    if ads.is_empty() || n_units < min_posts || spacing < 1 {
        drops.extend(
            ads.iter()
                .map(|ad| dropped_ad(ad, AdDropReason::TooFewPosts)),
        );
        return posts_to_feed_items(scored_posts);
    }

    let first_position = compute_first_position(&ads, spacing);
    let mut pool: Vec<(AdIndexInfo, Option<Block>)> =
        ads.into_iter().map(|ad| (ad, None)).collect();
    let mut ad_after: Vec<Option<AdIndexInfo>> = (0..scored_posts.len()).map(|_| None).collect();
    let mut gap = spacing - first_position;
    let mut medium_risk_openings = 0usize;

    for (unit_idx, unit) in units.iter().enumerate() {
        gap += 1;
        if gap >= spacing && !pool.is_empty() && unit_idx + 1 < n_units {
            let above = &scored_posts[unit.start..unit.end];
            let next = &units[unit_idx + 1];
            let below = &scored_posts[next.start..next.end];
            if unit_medium_risk(above) || unit_medium_risk(below) {
                medium_risk_openings += 1;
                continue;
            }

            let mut placed_idx = None;
            for (idx, (ad, last_block)) in pool.iter_mut().enumerate() {
                match group_block(ad, above, below) {
                    Some(block) => *last_block = Some(block),
                    None => {
                        placed_idx = Some(idx);
                        break;
                    }
                }
            }
            if let Some(idx) = placed_idx {
                let last = unit.end - 1;
                ad_after[last] = Some(pool.remove(idx).0);
                gap = 0;
            }
        }
    }

    let no_room = if medium_risk_openings > 0 {
        AdDropReason::NoEligibleNeighbours
    } else {
        AdDropReason::SpacingCap
    };
    drops.extend(pool.into_iter().map(|(ad, block)| match block {
        Some(block) => adjacency_drop(&ad, block),
        None => dropped_ad(&ad, no_room),
    }));

    let mut items: Vec<FeedItem> = Vec::with_capacity(scored_posts.len() + ad_after.len());
    for (index, post) in scored_posts.into_iter().enumerate() {
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(post)),
        });
        if let Some(ad) = ad_after[index].take() {
            items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ad)),
            });
        }
    }

    truncate_recording_drops(&mut items, result_size, drops);
    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }
    items
}
