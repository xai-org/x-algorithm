use super::util::MIN_REQUESTED_GAP;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{AdsUserStateKeepPct, AdsUserStateSpacing};
use xai_core_entities::entities::SubscriptionLevel;
use xai_stats_receiver::global_stats_receiver;

const MIN_KEEP_PCT: u64 = 50;
const REQUESTS_METRIC: &str = "AdsUserLoad.requests";
const SLOTS_METRIC: &str = "AdsUserLoad.slots";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct UserAdLoad {
    pub(crate) keep_rate: f64,
    pub(crate) spacing: Option<usize>,
    seed: u64,
    state: Option<&'static str>,
}

impl Default for UserAdLoad {
    fn default() -> Self {
        Self {
            keep_rate: 1.0,
            spacing: None,
            seed: 0,
            state: None,
        }
    }
}

pub(crate) fn is_enabled(query: &ScoredPostsQuery) -> bool {
    query.request_type == RequestType::ForYou
        && (!query.params.get(AdsUserStateKeepPct).is_empty()
            || !query.params.get(AdsUserStateSpacing).is_empty())
}

impl UserAdLoad {
    pub(crate) fn for_request(query: &ScoredPostsQuery) -> Self {
        if !is_enabled(query) {
            return Self::default();
        }
        if matches!(
            query.subscription_level,
            Some(SubscriptionLevel::Premium | SubscriptionLevel::PremiumPlus)
        ) {
            record_request("premium", "skipped");
            return Self::default();
        }
        let Some(user_state) = query.user_demographics.as_ref().map(|d| d.user_state) else {
            record_request("missing", "skipped");
            return Self::default();
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
            seed: query.request_id,
            state: Some(state),
        }
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
