//! Bounded, request-spanning author exposure counts. No service dependencies.
use std::collections::{HashMap, HashSet};

pub const MAX_EXPOSURES: usize = 20_000;
pub const MAX_RECENT_ITEMS: usize = 2_000;

#[derive(Clone, Copy, Debug)]
pub struct HistoryConfig {
    pub window_ms: i64,
    pub half_life_ms: f64,
    pub weight: f64,
    pub max_count: f64,
    pub max_items: usize,
}

impl HistoryConfig {
    pub fn is_valid(self) -> bool {
        self.window_ms > 0
            && self.window_ms <= 3_600_000
            && self.half_life_ms.is_finite()
            && self.half_life_ms > 0.0
            && self.weight.is_finite()
            && self.weight > 0.0
            && self.weight <= 1.0
            && self.max_count.is_finite()
            && self.max_count > 0.0
            && self.max_count <= MAX_RECENT_ITEMS as f64
            && self.max_items > 0
            && self.max_items <= MAX_RECENT_ITEMS
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExposureItem {
    Position(i64),
    Post(u64),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AuthorExposure {
    pub served_at_ms: i64,
    pub request_id: Option<i64>,
    pub item: ExposureItem,
    pub author_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Batch {
    Request(i64),
    Timestamp(i64),
}

pub fn content_author_id(author_id: u64, retweeted_user_id: Option<u64>) -> u64 {
    retweeted_user_id.filter(|id| *id != 0).unwrap_or(author_id)
}

/// Returns None for invalid configuration or oversized input (legacy fallback).
/// One (request, position, author) is one exposure, even if history expands a
/// conversation into several ancestor entries. Repeated serving in a DIFFERENT
/// request is a new exposure; replaying the same history batch is not.
pub fn recent_author_counts(
    exposures: &[AuthorExposure],
    now_ms: i64,
    config: HistoryConfig,
) -> Option<HashMap<u64, f64>> {
    if !config.is_valid() || now_ms <= 0 || exposures.len() > MAX_EXPOSURES {
        return None;
    }
    let mut recent: Vec<_> = exposures
        .iter()
        .copied()
        .filter(|e| {
            e.author_id != 0
                && e.served_at_ms > 0
                && now_ms
                    .checked_sub(e.served_at_ms)
                    .is_some_and(|age| (0..config.window_ms).contains(&age))
        })
        .collect();
    // Newest events win both the item budget and duplicate-batch timestamps.
    // Full ordering makes truncation independent of history response ordering.
    recent.sort_unstable_by(|a, b| b.cmp(a));
    let mut seen = HashSet::new();
    let mut counts = HashMap::<u64, f64>::new();
    for e in recent {
        let batch = match e.request_id.filter(|id| *id != 0) {
            Some(id) => Batch::Request(id),
            None => Batch::Timestamp(e.served_at_ms),
        };
        if !seen.insert((batch, e.item, e.author_id)) {
            continue;
        }
        let age_ms = (now_ms - e.served_at_ms) as f64;
        let count = counts.entry(e.author_id).or_default();
        *count = (*count + (-age_ms / config.half_life_ms).exp2()).min(config.max_count);
        if seen.len() == config.max_items {
            break;
        }
    }
    for count in counts.values_mut() {
        *count *= config.weight;
    }
    Some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> HistoryConfig {
        HistoryConfig {
            window_ms: 300_000,
            half_life_ms: 60_000.0,
            weight: 1.0,
            max_count: 5.0,
            max_items: 200,
        }
    }
    fn event(request: i64, position: i64, author: u64, time: i64) -> AuthorExposure {
        AuthorExposure {
            request_id: Some(request),
            item: ExposureItem::Position(position),
            author_id: author,
            served_at_ms: time,
        }
    }
    #[test]
    fn half_life_window_and_invalid_timestamps() {
        let now = 1_000_000;
        let es = [
            event(1, 0, 1, now),
            event(2, 0, 2, now - 60_000),
            event(3, 0, 3, now - 300_000),
            event(4, 0, 4, now + 1),
            event(5, 0, 5, i64::MIN),
            event(6, 0, 0, now),
        ];
        let c = recent_author_counts(&es, now, config()).unwrap();
        assert_eq!(c, HashMap::from([(1, 1.0), (2, 0.5)]));
    }
    #[test]
    fn retries_and_conversation_entries_count_once_per_slot_author() {
        let a = event(1, 0, 10, 1_000_000);
        let es = [
            a,
            a,
            event(1, 0, 10, 999_999),
            event(1, 1, 10, 1_000_000),
            event(2, 0, 10, 1_000_000),
            event(1, 0, 20, 1_000_000),
        ];
        assert_eq!(
            recent_author_counts(&es, 1_000_000, config()).unwrap(),
            HashMap::from([(10, 3.0), (20, 1.0)])
        );
    }
    #[test]
    fn fallback_identity_and_negative_request_ids() {
        let mut a = event(-5, 0, 10, 1_000_000);
        let b = AuthorExposure {
            served_at_ms: 999_999,
            ..a
        };
        assert_eq!(
            recent_author_counts(&[a, b], 1_000_000, config()).unwrap()[&10],
            1.0
        );
        a.request_id = None;
        assert_eq!(
            recent_author_counts(&[a, a], 1_000_000, config()).unwrap()[&10],
            1.0
        );
        let b = AuthorExposure {
            item: ExposureItem::Post(99),
            ..a
        };
        assert_eq!(
            recent_author_counts(&[a, b], 1_000_000, config()).unwrap()[&10],
            2.0
        );
    }
    #[test]
    fn newest_unique_events_get_the_budget_and_order_does_not_matter() {
        let mut es = vec![
            event(1, 0, 10, 900_000),
            event(2, 0, 20, 1_000_000),
            event(2, 0, 20, 1_000_000),
            event(3, 0, 30, 990_000),
        ];
        let cfg = HistoryConfig {
            max_items: 2,
            ..config()
        };
        let c = recent_author_counts(&es, 1_000_000, cfg).unwrap();
        assert!(!c.contains_key(&10));
        assert!(c.contains_key(&20));
        assert!(c.contains_key(&30));
        es.reverse();
        assert_eq!(recent_author_counts(&es, 1_000_000, cfg).unwrap(), c);
    }
    #[test]
    fn cap_then_weight() {
        let es: Vec<_> = (1..100).map(|id| event(id, 0, 10, 1_000_000)).collect();
        let cfg = HistoryConfig {
            weight: 0.25,
            max_count: 2.0,
            ..config()
        };
        assert_eq!(recent_author_counts(&es, 1_000_000, cfg).unwrap()[&10], 0.5);
    }
    #[test]
    fn bad_config_and_oversized_input_fail_open() {
        for cfg in [
            HistoryConfig {
                weight: f64::NAN,
                ..config()
            },
            HistoryConfig {
                weight: -1.0,
                ..config()
            },
            HistoryConfig {
                weight: 0.0,
                ..config()
            },
            HistoryConfig {
                weight: 1.01,
                ..config()
            },
            HistoryConfig {
                half_life_ms: 0.0,
                ..config()
            },
            HistoryConfig {
                half_life_ms: f64::INFINITY,
                ..config()
            },
            HistoryConfig {
                max_count: f64::NAN,
                ..config()
            },
            HistoryConfig {
                max_count: 0.0,
                ..config()
            },
            HistoryConfig {
                window_ms: 0,
                ..config()
            },
            HistoryConfig {
                window_ms: 3_600_001,
                ..config()
            },
            HistoryConfig {
                max_items: 0,
                ..config()
            },
            HistoryConfig {
                max_items: MAX_RECENT_ITEMS + 1,
                ..config()
            },
        ] {
            assert!(recent_author_counts(&[], 1_000_000, cfg).is_none());
        }
        assert!(recent_author_counts(&[], 0, config()).is_none());
        assert!(
            recent_author_counts(&vec![event(1, 0, 1, 1); MAX_EXPOSURES + 1], 100, config())
                .is_none()
        );
    }
    #[test]
    fn count_decreases_as_time_advances() {
        let es = [event(1, 0, 1, 1_000_000)];
        let mut prior = f64::INFINITY;
        for seconds in 0..=300 {
            let next = recent_author_counts(&es, 1_000_000 + seconds * 1000, config())
                .unwrap()
                .get(&1)
                .copied()
                .unwrap_or(0.0);
            assert!(next <= prior);
            prior = next;
        }
        assert_eq!(prior, 0.0);
    }
    #[test]
    fn original_author_key_handles_absent_and_zero_ids() {
        assert_eq!(content_author_id(10, Some(20)), 20);
        assert_eq!(content_author_id(10, Some(0)), 10);
        assert_eq!(content_author_id(10, None), 10);
    }
}
