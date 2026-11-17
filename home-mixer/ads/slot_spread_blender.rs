use super::drops::{adjacency_drop, dropped_ad};
use super::health_metrics::{serving_limitation, HealthMetrics};
use super::neighbours::{
    assemble, page_capacity, safe_slot_estimate, unplaceable_drop, AdRules, ClassCounts,
    NeighbourChecker, PostClass,
};
use super::user_load::UserAdLoad;
use super::util::{compute_spacing, posts_to_feed_items, Block, BlockReason, MIN_POSTS_FOR_ADS};
use super::AdsBlender;
use crate::params::RESULT_SIZE;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

const METRICS: HealthMetrics = HealthMetrics {
    enforcement: "SlotSpread.enforcement",
    slot_outcome: "SlotSpread.slot_outcome",
    serving_limitation: "SlotSpread.serving_limitation",
};

#[derive(Default)]
pub struct SlotSpreadAdsBlender {
    pub(crate) load: UserAdLoad,
}

impl AdsBlender for SlotSpreadAdsBlender {
    fn blend_inner_with_drops(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        blend_with_load(scored_posts, ads, MIN_POSTS_FOR_ADS, self.load, drops)
    }
}

#[cfg(test)]
pub(crate) fn blend_with_drops(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    blend_with_load(scored_posts, ads, min_posts, UserAdLoad::default(), drops)
}

pub(crate) fn blend_with_load(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    load: UserAdLoad,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let first_drop = drops.len();
    let (items, offered) = blend(scored_posts, ads, min_posts, load, drops);
    METRICS.emit_page(offered, &items, &drops[first_drop..]);
    items
}

fn blend(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    load: UserAdLoad,
    drops: &mut Vec<DroppedAd>,
) -> (Vec<FeedItem>, usize) {
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
        return (posts_to_feed_items(scored_posts), 0);
    }

    let spacing = load
        .spacing
        .unwrap_or_else(|| compute_spacing(&ads).requested);
    let page_slots = page_capacity(spacing, n, RESULT_SIZE);
    let safe_slots = safe_slot_estimate(&scored_posts, &ads);
    METRICS.emit_serving_limitation(serving_limitation(ads.len(), page_slots, safe_slots));
    let capacity = page_slots.min(ads.len());
    let slot_count = capacity.min(safe_slots);
    if slot_count == 0 {
        let reason = if capacity == 0 {
            AdDropReason::SpacingCap
        } else {
            AdDropReason::NoEligibleNeighbours
        };
        drops.extend(ads.iter().map(|ad| dropped_ad(ad, reason)));
        return (posts_to_feed_items(scored_posts), 0);
    }

    let shown = n.min(RESULT_SIZE - slot_count);
    let slots = load.thin(spread_slots(slot_count, spacing, shown));
    let overflow = if slot_count >= capacity || slots.len() < slot_count {
        AdDropReason::SpacingCap
    } else {
        AdDropReason::NoEligibleNeighbours
    };
    let (organic, placed) = assign(&scored_posts, &ads, &slots, overflow, drops);
    if placed.is_empty() {
        return (posts_to_feed_items(scored_posts), slots.len());
    }

    let ad_before: Vec<(usize, usize)> = placed
        .iter()
        .enumerate()
        .map(|(k, &ad)| (slots[k] + 1, ad))
        .collect();
    (
        assemble(scored_posts, ads, &organic, &ad_before, RESULT_SIZE, drops),
        slots.len(),
    )
}

fn spread_slots(slots: usize, spacing: usize, shown: usize) -> Vec<usize> {
    let surplus = shown.saturating_sub((slots - 1) * spacing + 2);
    let (per_gap, remainder) = (surplus / slots, surplus % slots);
    let mut above = Vec::with_capacity(slots);
    let mut index = 0;
    for k in 0..slots {
        above.push(index);
        index += spacing + per_gap + usize::from(k >= slots - remainder);
    }
    above
}

fn assign(
    posts: &[ScoredPost],
    ads: &[AdIndexInfo],
    slots: &[usize],
    overflow: AdDropReason,
    drops: &mut Vec<DroppedAd>,
) -> (Vec<usize>, Vec<usize>) {
    let n = posts.len();
    let mut checker = NeighbourChecker::new(posts);
    let rules: Vec<AdRules> = ads.iter().map(AdRules::new).collect();
    let mut sensitive_after = vec![0; rules.len() + 1];
    for j in (0..rules.len()).rev() {
        sensitive_after[j] = sensitive_after[j + 1] + usize::from(rules[j].sensitive);
    }
    let counts = ClassCounts::of(posts);
    let (mut safe_free, mut low_free) = (counts.safe, counts.low);
    let mut organic: Vec<usize> = (0..n).collect();
    let mut used = vec![false; n];
    let mut placed: Vec<usize> = Vec::new();

    for (j, rule) in rules.iter().enumerate() {
        if placed.len() >= slots.len() {
            drops.push(dropped_ad(rule.ad, overflow));
            continue;
        }
        if !rule.bsr_high && safe_free + low_free < 2 {
            drops.push(dropped_ad(rule.ad, AdDropReason::NoEligibleNeighbours));
            continue;
        }
        if rule.sensitive && safe_free < 2 {
            drops.push(low_risk_shortage_drop(rule.ad, &checker, &organic, &used));
            continue;
        }

        let above = slots[placed.len()];
        let below = above + 1;
        let slots_left = slots.len() - placed.len() - 1;
        let prefer_low = !rule.sensitive
            && low_free > 0
            && safe_free / 2 < slots_left.min(sensitive_after[j + 1]);
        let Some(first) = pick(
            &mut checker,
            rule,
            above,
            &organic,
            &used,
            &[below],
            prefer_low,
        ) else {
            drops.push(unpaired_drop(rule, &mut checker, &organic, &used));
            continue;
        };
        let Some(second) = pick(
            &mut checker,
            rule,
            below,
            &organic,
            &used,
            &[above, first],
            prefer_low,
        ) else {
            drops.push(unpaired_drop(rule, &mut checker, &organic, &used));
            continue;
        };

        organic.swap(above, first);
        organic.swap(below, second);
        used[above] = true;
        used[below] = true;
        for position in [above, below] {
            match checker.class(organic[position]) {
                PostClass::Safe => safe_free -= 1,
                PostClass::Low => low_free -= 1,
                PostClass::Medium | PostClass::High => {}
            }
        }
        placed.push(j);
    }
    (organic, placed)
}

fn pick(
    checker: &mut NeighbourChecker,
    rule: &AdRules,
    index: usize,
    organic: &[usize],
    used: &[bool],
    exclude: &[usize],
    prefer_low: bool,
) -> Option<usize> {
    let passes: &[bool] = if prefer_low { &[true, false] } else { &[false] };
    passes.iter().find_map(|&low_only| {
        let mut usable = |position: usize| {
            !used[position]
                && !exclude.contains(&position)
                && (!low_only || checker.class(organic[position]) == PostClass::Low)
                && checker.compatible(rule, organic[position])
        };
        nearest(index, organic.len(), &mut usable)
    })
}

fn nearest(index: usize, len: usize, usable: &mut impl FnMut(usize) -> bool) -> Option<usize> {
    if usable(index) {
        return Some(index);
    }
    for distance in 1..len {
        let lower = index.checked_sub(distance);
        let upper = index + distance;
        if lower.is_none() && upper >= len {
            break;
        }
        if let Some(lower) = lower.filter(|&position| usable(position)) {
            return Some(lower);
        }
        if upper < len && usable(upper) {
            return Some(upper);
        }
    }
    None
}

fn unpaired_drop(
    rule: &AdRules,
    checker: &mut NeighbourChecker,
    organic: &[usize],
    used: &[bool],
) -> DroppedAd {
    let fits = organic
        .iter()
        .zip(used)
        .filter(|&(_, &taken)| !taken)
        .map(|(&post, _)| checker.fit(rule, post));
    unplaceable_drop(rule.ad, fits)
}

fn low_risk_shortage_drop(
    ad: &AdIndexInfo,
    checker: &NeighbourChecker,
    organic: &[usize],
    used: &[bool],
) -> DroppedAd {
    organic
        .iter()
        .zip(used)
        .find(|&(&post, &taken)| !taken && checker.class(post) == PostClass::Low)
        .map(|(&post, _)| {
            adjacency_drop(
                ad,
                Block {
                    tweet_id: checker.tweet_id(post),
                    reason: BlockReason::LowRiskNeighbour,
                },
            )
        })
        .unwrap_or_else(|| dropped_ad(ad, AdDropReason::NoEligibleNeighbours))
}
