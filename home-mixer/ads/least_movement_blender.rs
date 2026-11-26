use super::drops::dropped_ad;
use super::health_metrics::{serving_limitation, HealthMetrics};
use super::neighbours::{
    assemble, page_capacity, safe_slot_estimate, unplaceable_drop, AdRules, Fit, NeighbourChecker,
};
use super::util::{compute_spacing, posts_to_feed_items, MIN_POSTS_FOR_ADS};
use super::AdsBlender;
use crate::params::RESULT_SIZE;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

const SLACK_WEIGHT: i64 = 2;
const FIRST_AD_SLACK_WEIGHT: i64 = 50;
const METRICS: HealthMetrics = HealthMetrics {
    enforcement: "LeastMovement.enforcement",
    slot_outcome: "LeastMovement.slot_outcome",
    serving_limitation: "LeastMovement.serving_limitation",
};

pub struct LeastMovementAdsBlender;

impl AdsBlender for LeastMovementAdsBlender {
    fn blend_inner_with_drops(
        &self,
        scored_posts: Vec<ScoredPost>,
        ads: Vec<AdIndexInfo>,
        drops: &mut Vec<DroppedAd>,
    ) -> Vec<FeedItem> {
        blend_with_drops(scored_posts, ads, MIN_POSTS_FOR_ADS, drops)
    }
}

pub(crate) fn blend_with_drops(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<FeedItem> {
    let first_drop = drops.len();
    let (items, offered) = blend(scored_posts, ads, min_posts, drops);
    METRICS.emit_page(offered, &items, &drops[first_drop..]);
    items
}

fn blend(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
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

    let spacing = compute_spacing(&ads).requested;
    let capacity = page_capacity(spacing, n, RESULT_SIZE);
    let safe_slots = safe_slot_estimate(&scored_posts, &ads);
    METRICS.emit_serving_limitation(serving_limitation(ads.len(), capacity, safe_slots));
    let offered = ads.len().min(capacity).min(safe_slots);
    let chosen = select(&scored_posts, &ads, capacity, drops);
    if chosen.is_empty() {
        return (posts_to_feed_items(scored_posts), offered);
    }

    let (chosen_ads, compatible): (Vec<usize>, Vec<Vec<bool>>) = chosen.into_iter().unzip();
    let plan = Layout::new(n, spacing, compatible).run();
    drops.extend(
        plan.skipped
            .iter()
            .map(|&k| dropped_ad(&ads[chosen_ads[k]], AdDropReason::NoEligibleNeighbours)),
    );
    let ad_before: Vec<(usize, usize)> = plan
        .placed
        .iter()
        .map(|&(k, index)| (index, chosen_ads[k]))
        .collect();
    (
        assemble(scored_posts, ads, &plan.out, &ad_before, RESULT_SIZE, drops),
        offered,
    )
}

fn select(
    posts: &[ScoredPost],
    ads: &[AdIndexInfo],
    capacity: usize,
    drops: &mut Vec<DroppedAd>,
) -> Vec<(usize, Vec<bool>)> {
    let mut checker = NeighbourChecker::new(posts);
    let all_posts = vec![true; posts.len()];
    let mut chosen: Vec<(usize, Vec<bool>)> = Vec::new();
    for (index, ad) in ads.iter().enumerate() {
        if chosen.len() >= capacity {
            drops.push(dropped_ad(ad, AdDropReason::SpacingCap));
            continue;
        }
        let rules = AdRules::new(ad);
        let fits: Vec<Fit> = (0..posts.len())
            .map(|post| checker.fit(&rules, post))
            .collect();
        let compatible: Vec<bool> = fits
            .iter()
            .map(|fit| matches!(fit, Fit::Compatible))
            .collect();
        let feasible = compatible.iter().filter(|&&c| c).count() >= 2 && {
            let mut sets: Vec<&[bool]> = chosen.iter().map(|(_, set)| set.as_slice()).collect();
            sets.push(&compatible);
            match_pairs(&sets, &all_posts).is_some()
        };
        if feasible {
            chosen.push((index, compatible));
        } else {
            drops.push(unplaceable_drop(ad, fits));
        }
    }
    chosen
}

fn match_pairs(sets: &[&[bool]], pool: &[bool]) -> Option<Vec<[usize; 2]>> {
    let mut owner: Vec<Option<usize>> = vec![None; pool.len()];
    let mut assigned = vec![0; sets.len() * 2];
    for slot in 0..sets.len() * 2 {
        let mut seen = vec![false; pool.len()];
        if !augment(slot, sets, pool, &mut owner, &mut assigned, &mut seen) {
            return None;
        }
    }
    Some(
        assigned
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect(),
    )
}

fn augment(
    slot: usize,
    sets: &[&[bool]],
    pool: &[bool],
    owner: &mut [Option<usize>],
    assigned: &mut [usize],
    seen: &mut [bool],
) -> bool {
    for post in 0..pool.len() {
        if !sets[slot / 2][post] || !pool[post] || seen[post] {
            continue;
        }
        seen[post] = true;
        let free = match owner[post] {
            None => true,
            Some(other) => augment(other, sets, pool, owner, assigned, seen),
        };
        if free {
            owner[post] = Some(slot);
            assigned[slot] = post;
            return true;
        }
    }
    false
}

fn is_strict_subset(set: &[bool], of: &[bool]) -> bool {
    set != of && set.iter().zip(of).all(|(&inside, &outer)| !inside || outer)
}

struct Plan {
    out: Vec<usize>,
    placed: Vec<(usize, usize)>,
    skipped: Vec<usize>,
}

struct Choice {
    key: (i64, usize, usize, usize),
    index: usize,
    emitted: Vec<usize>,
    pool: Vec<usize>,
}

struct Window {
    emitted: usize,
    lo: usize,
    hi: usize,
    weight: i64,
}

struct Layout {
    posts: usize,
    spacing: usize,
    page_posts: usize,
    compatible: Vec<Vec<bool>>,
    pool: Vec<usize>,
    out: Vec<usize>,
    placed: Vec<(usize, usize)>,
    skipped: Vec<usize>,
}

impl Layout {
    fn new(posts: usize, spacing: usize, compatible: Vec<Vec<bool>>) -> Self {
        Self {
            posts,
            spacing,
            page_posts: posts.min(RESULT_SIZE.saturating_sub(compatible.len())),
            compatible,
            pool: (0..posts).collect(),
            out: Vec::with_capacity(posts),
            placed: Vec::new(),
            skipped: Vec::new(),
        }
    }

    fn run(mut self) -> Plan {
        let mut remaining: Vec<usize> = (0..self.compatible.len()).collect();
        let mut previous: Option<usize> = None;
        while let Some(&ad) = remaining.first() {
            let emitted = self.out.len();
            let lo = previous
                .map_or(1, |index| index + self.spacing)
                .max(emitted + 1);
            let hi = self.page_posts as isize
                - 1
                - (remaining.len() as isize - 1) * self.spacing as isize;
            let weight = if self.placed.is_empty() {
                FIRST_AD_SLACK_WEIGHT
            } else {
                SLACK_WEIGHT
            };
            if let Some(choice) = self.best_choice(ad, &remaining[1..], emitted, lo, hi, weight) {
                self.out.extend(choice.emitted);
                self.pool = choice.pool;
                self.placed.push((ad, choice.index));
                previous = Some(choice.index);
                remaining.remove(0);
                continue;
            }
            if lo as isize <= hi && self.place_tight(&remaining, lo) {
                break;
            }
            self.skipped.push(remaining.remove(0));
        }
        let page_posts = self
            .posts
            .min(RESULT_SIZE.saturating_sub(self.placed.len()));
        let fill = page_posts
            .saturating_sub(self.out.len())
            .min(self.pool.len());
        self.out.extend(self.pool.drain(..fill));
        Plan {
            out: self.out,
            placed: self.placed,
            skipped: self.skipped,
        }
    }

    fn membership(&self, posts: &[usize]) -> Vec<bool> {
        let mut set = vec![false; self.posts];
        for &post in posts {
            set[post] = true;
        }
        set
    }

    fn place_tight(&mut self, remaining: &[usize], lo: usize) -> bool {
        let pairs = {
            let sets: Vec<&[bool]> = remaining
                .iter()
                .map(|&ad| self.compatible[ad].as_slice())
                .collect();
            match_pairs(&sets, &self.membership(&self.pool))
        };
        let Some(pairs) = pairs else {
            return false;
        };
        let paired = self.membership(pairs.as_flattened());
        self.pool.retain(|&post| !paired[post]);
        for (j, (&ad, pair)) in remaining.iter().zip(pairs).enumerate() {
            let index = lo + j * self.spacing;
            let take = (index - 1)
                .saturating_sub(self.out.len())
                .min(self.pool.len());
            self.out.extend(self.pool.drain(..take));
            self.out.push(pair[0].min(pair[1]));
            self.placed.push((ad, self.out.len()));
            self.out.push(pair[0].max(pair[1]));
        }
        true
    }

    fn best_choice(
        &self,
        ad: usize,
        rest: &[usize],
        emitted: usize,
        lo: usize,
        hi: isize,
        weight: i64,
    ) -> Option<Choice> {
        let mine = self.compatible[ad].as_slice();
        let compatible: Vec<usize> = (0..self.pool.len())
            .filter(|&i| mine[self.pool[i]])
            .collect();
        if lo as isize > hi || compatible.len() < 2 {
            return None;
        }
        let window = Window {
            emitted,
            lo,
            hi: hi as usize,
            weight,
        };
        let rest_sets: Vec<&[bool]> = rest
            .iter()
            .map(|&a| self.compatible[a].as_slice())
            .collect();
        let stricter: Vec<&[bool]> = rest_sets
            .iter()
            .copied()
            .filter(|set| is_strict_subset(set, mine))
            .collect();
        let needed: Vec<bool> = (0..self.posts)
            .map(|post| rest_sets.iter().any(|set| set[post]))
            .collect();
        let unclaimed: Vec<usize> = compatible
            .iter()
            .copied()
            .filter(|&i| !stricter.iter().any(|set| set[self.pool[i]]))
            .collect();
        let unclaimed = (!unclaimed.is_empty() && unclaimed.len() < compatible.len())
            .then_some(unclaimed.as_slice());
        [(None, None), (Some(needed.as_slice()), unclaimed)]
            .into_iter()
            .find_map(|(reserve, unclaimed)| {
                self.choices(&window, &compatible, reserve, unclaimed)
                    .into_iter()
                    .find(|choice| {
                        rest_sets.is_empty()
                            || match_pairs(&rest_sets, &self.membership(&choice.pool)).is_some()
                    })
            })
    }

    fn choices(
        &self,
        window: &Window,
        compatible: &[usize],
        reserve: Option<&[bool]>,
        unclaimed: Option<&[usize]>,
    ) -> Vec<Choice> {
        let mut choices = Vec::new();
        for index in window.lo..=window.hi {
            let before = index - 1 - window.emitted;
            if before + 2 > self.pool.len() {
                break;
            }
            let mut pairs = candidate_pairs(compatible, before);
            if let Some(unclaimed) = unclaimed {
                pairs.extend(candidate_pairs(unclaimed, before));
                let near_any = nearest_to(compatible, before);
                let near_unclaimed = nearest_to(unclaimed, before);
                if near_any != near_unclaimed {
                    pairs.push((near_any.min(near_unclaimed), near_any.max(near_unclaimed)));
                }
            }
            let mut unique: Vec<(usize, usize)> = Vec::with_capacity(pairs.len());
            for pair in pairs {
                if !unique.contains(&pair) {
                    unique.push(pair);
                }
            }
            for pair in unique {
                if let Some(choice) = self.choice(window, index, before, pair, reserve) {
                    choices.push(choice);
                }
            }
        }
        choices.sort_by_key(|choice| choice.key);
        choices
    }

    fn choice(
        &self,
        window: &Window,
        index: usize,
        before: usize,
        (a, b): (usize, usize),
        reserve: Option<&[bool]>,
    ) -> Option<Choice> {
        let end = b.max(before + 1);
        let region = &self.pool[..=end];
        let beyond = &self.pool[end + 1..];
        let rest: Vec<usize> = region
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != a && j != b)
            .map(|(_, &post)| post)
            .collect();
        let (mut sequence, tail) = match reserve {
            Some(reserve) => {
                let fill: Vec<usize> = rest
                    .iter()
                    .chain(beyond)
                    .copied()
                    .filter(|&post| !reserve[post])
                    .take(before)
                    .collect();
                if fill.len() < before {
                    return None;
                }
                let mut sequence = fill.clone();
                sequence.extend([self.pool[a], self.pool[b]]);
                sequence.extend(rest.iter().copied().filter(|post| !fill.contains(post)));
                let tail: Vec<usize> = beyond
                    .iter()
                    .copied()
                    .filter(|post| !fill.contains(post))
                    .collect();
                (sequence, tail)
            }
            None => {
                let mut sequence = rest[..before].to_vec();
                sequence.extend([self.pool[a], self.pool[b]]);
                sequence.extend_from_slice(&rest[before..]);
                (sequence, beyond.to_vec())
            }
        };
        let mut cost =
            self.movement(&sequence, window.emitted) - self.movement(region, window.emitted);
        if tail.len() != beyond.len() {
            cost += self.movement(&tail, window.emitted + sequence.len())
                - self.movement(beyond, window.emitted + end + 1);
        }
        let slack = index - window.lo;
        let mut pool = sequence.split_off(before + 2);
        pool.extend(tail);
        Some(Choice {
            key: (cost + window.weight * slack as i64, slack, index, a + b),
            index,
            emitted: sequence,
            pool,
        })
    }

    fn movement(&self, posts: &[usize], start: usize) -> i64 {
        posts
            .iter()
            .enumerate()
            .take_while(|&(j, _)| start + j < self.page_posts)
            .map(|(j, &post)| (start + j).abs_diff(post) as i64)
            .sum()
    }
}

fn candidate_pairs(compatible: &[usize], before: usize) -> Vec<(usize, usize)> {
    let split = compatible.partition_point(|&i| i <= before);
    let (at_or_before, after) = compatible.split_at(split);
    let mut pairs = Vec::with_capacity(3);
    if let (Some(&a), Some(&b)) = (at_or_before.last(), after.first()) {
        pairs.push((a, b));
    }
    if let [a, b, ..] = after {
        pairs.push((*a, *b));
    }
    if let [.., a, b] = at_or_before {
        pairs.push((*a, *b));
    }
    pairs
}

fn nearest_to(positions: &[usize], target: usize) -> usize {
    positions
        .iter()
        .copied()
        .min_by_key(|&i| (i.abs_diff(target), i))
        .unwrap_or(target)
}
