use super::util::MIN_REQUESTED_GAP;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    AdsPageScoreHighThreshold, AdsPageScoreKeepPct, AdsPageScoreLowThreshold, AdsPageScoreSpacing,
    AdsPageScoreTopK, AdsUserLoadEvenSpread, AdsUserStateKeepPct, AdsUserStateSpacing,
    ValueModelMode,
};
use xai_core_entities::entities::SubscriptionLevel;
use xai_home_mixer_proto::ScoredPost;
use xai_stats_receiver::global_stats_receiver;

const MIN_KEEP_PCT: u64 = 50;
const REQUESTS_METRIC: &str = "AdsUserLoad.requests";
const SLOTS_METRIC: &str = "AdsUserLoad.slots";
const PAGE_SCORE_METRIC: &str = "AdsPageScore.mean";
const PAGE_TIERS: [&str; 3] = ["page_low", "page_mid", "page_high"];
const PROD_VALUE_MODEL: &str = "weighted";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct UserAdLoad {
    pub(crate) keep_rate: f64,
    pub(crate) spacing: Option<usize>,
    pub(crate) even_spread: bool,
    seed: u64,
    state: Option<&'static str>,
    page: Option<PageScoreTiers>,
}

impl Default for UserAdLoad {
    fn default() -> Self {
        Self {
            keep_rate: 1.0,
            spacing: None,
            even_spread: false,
            seed: 0,
            state: None,
            page: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PageScoreTiers {
    low: f64,
    high: f64,
    keep_pct: [u64; 3],
    spacing: [u64; 3],
    top_k: usize,
}

impl PageScoreTiers {
    fn from_query(query: &ScoredPostsQuery) -> Option<Self> {
        if query.request_type != RequestType::ForYou {
            return None;
        }
        let keep_pct = query.params.get(AdsPageScoreKeepPct);
        let spacing = query.params.get(AdsPageScoreSpacing);
        if keep_pct.is_empty() && spacing.is_empty() {
            return None;
        }
        if query.params.get(ValueModelMode) != PROD_VALUE_MODEL {
            record_request("page_vm_mode", "skipped");
            return None;
        }
        let pick = |values: &[u64], tier: usize, default: u64| {
            values.get(tier).copied().unwrap_or(default)
        };
        Some(Self {
            low: query.params.get(AdsPageScoreLowThreshold),
            high: query.params.get(AdsPageScoreHighThreshold),
            keep_pct: std::array::from_fn(|tier| pick(&keep_pct, tier, 100)),
            spacing: std::array::from_fn(|tier| pick(&spacing, tier, 0)),
            top_k: query.params.get(AdsPageScoreTopK).max(1) as usize,
        })
    }

    fn tier(&self, posts: &[ScoredPost]) -> Option<(usize, f64)> {
        let scores: Vec<f64> = posts
            .iter()
            .take(self.top_k)
            .map(|post| post.score as f64)
            .filter(|score| score.is_finite())
            .collect();
        if scores.is_empty() {
            return None;
        }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let tier = if mean < self.low {
            0
        } else if mean < self.high {
            1
        } else {
            2
        };
        Some((tier, mean))
    }
}

pub(crate) fn is_enabled(query: &ScoredPostsQuery) -> bool {
    query.request_type == RequestType::ForYou
        && (!query.params.get(AdsUserStateKeepPct).is_empty()
            || !query.params.get(AdsUserStateSpacing).is_empty())
}

impl UserAdLoad {
    pub(crate) fn for_request(query: &ScoredPostsQuery) -> Self {
        let page = PageScoreTiers::from_query(query);
        if !is_enabled(query) && page.is_none() {
            return Self::default();
        }
        if matches!(
            query.subscription_level,
            Some(SubscriptionLevel::Premium | SubscriptionLevel::PremiumPlus)
        ) {
            record_request("premium", "skipped");
            return Self::default();
        }
        let base = Self {
            seed: query.request_id,
            even_spread: query.params.get(AdsUserLoadEvenSpread),
            page,
            ..Self::default()
        };
        if !is_enabled(query) {
            return base;
        }
        let Some(user_state) = query.user_demographics.as_ref().map(|d| d.user_state) else {
            record_request("missing", "skipped");
            return base;
        };
        let index = usize::try_from(user_state).unwrap_or(usize::MAX);
        let keep_pct = query
            .params
            .get(AdsUserStateKeepPct)
            .get(index)
            .copied()
            .unwrap_or(100)
            .clamp(MIN_KEEP_PCT, 100);
        let spacing = query
            .params
            .get(AdsUserStateSpacing)
            .get(index)
            .copied()
            .filter(|&s| s > 0)
            .map(|s| (s as usize).max(MIN_REQUESTED_GAP));
        let state = state_name(user_state);
        record_request(state, "active");
        Self {
            keep_rate: keep_pct as f64 / 100.0,
            spacing,
            state: Some(state),
            ..base
        }
    }

    pub(crate) fn for_page(self, posts: &[ScoredPost]) -> Self {
        let Some((page, (tier, mean))) = self.page.and_then(|p| p.tier(posts).map(|t| (p, t)))
        else {
            return self;
        };
        let label = PAGE_TIERS[tier];
        record_request(label, "active");
        if let Some(receiver) = global_stats_receiver() {
            receiver.observe_vm(PAGE_SCORE_METRIC, &[("tier", label)], mean);
        }
        let spacing = Some(page.spacing[tier])
            .filter(|&s| s > 0)
            .map(|s| (s as usize).max(MIN_REQUESTED_GAP));
        Self {
            keep_rate: page.keep_pct[tier].clamp(MIN_KEEP_PCT, 100) as f64 / 100.0,
            spacing,
            state: Some(label),
            ..self
        }
    }

    pub(crate) fn kept_count(&self, planned: usize) -> usize {
        let kept = if self.keep_rate >= 1.0 {
            planned
        } else {
            let expected = planned as f64 * self.keep_rate;
            let floor = expected.floor();
            let round_up = unit_hash(self.seed, u64::MAX) < expected - floor;
            (floor as usize + usize::from(round_up)).min(planned)
        };
        if let Some(state) = self.state {
            record_slots(state, kept, planned - kept);
        }
        kept
    }

    pub(crate) fn thin(&self, slots: Vec<usize>) -> Vec<usize> {
        let planned = slots.len();
        let kept: Vec<usize> = if self.keep_rate >= 1.0 {
            slots
        } else {
            slots
                .into_iter()
                .enumerate()
                .filter(|&(k, _)| unit_hash(self.seed, k as u64) < self.keep_rate)
                .map(|(_, slot)| slot)
                .collect()
        };
        if let Some(state) = self.state {
            record_slots(state, kept.len(), planned - kept.len());
        }
        kept
    }
}

fn unit_hash(seed: u64, slot: u64) -> f64 {
    let mut z = seed ^ slot.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn state_name(user_state: i32) -> &'static str {
    match user_state {
        0 => "new",
        1 => "near_zero",
        2 => "very_light",
        3 => "light",
        4 => "medium_tweeter",
        5 => "medium_non_tweeter",
        6 => "heavy_non_tweeter",
        7 => "heavy_tweeter",
        _ => "unknown",
    }
}

fn record_request(state: &'static str, outcome: &'static str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(
            REQUESTS_METRIC,
            &[("state", state), ("outcome", outcome)],
            1,
        );
    }
}

fn record_slots(state: &'static str, kept: usize, thinned: usize) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    for (outcome, count) in [("kept", kept), ("thinned", thinned)] {
        if count > 0 {
            receiver.incr(
                SLOTS_METRIC,
                &[("state", state), ("outcome", outcome)],
                count as u64,
            );
        }
    }
}
