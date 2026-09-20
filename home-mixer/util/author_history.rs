//! Adapter from existing served-history batches; no new storage or RPC.
use super::author_diversity::{self, AuthorExposure, ExposureItem, HistoryConfig, MAX_EXPOSURES};
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::*;
use std::collections::HashMap;
use xai_x_thrift::served_history::EntityIdType;

/// None preserves the legacy scoring path exactly. Some(empty) is a valid,
/// enabled treatment with no recent exposures (e.g. a first request).
pub fn counts_for_query(query: &ScoredPostsQuery) -> Option<HashMap<u64, f64>> {
    if query.request_type != RequestType::ForYou
        || !query.params.get(EnableCrossRequestAuthorDiversity)
        || !query.served_history_loaded
    {
        return None;
    }
    let config = HistoryConfig {
        window_ms: i64::from(query.params.get(AuthorDiversityHistoryWindowSeconds)) * 1000,
        half_life_ms: query.params.get(AuthorDiversityHistoryHalfLifeSeconds) * 1000.0,
        weight: query.params.get(AuthorDiversityHistoryWeight),
        max_count: query.params.get(AuthorDiversityHistoryMaxCount),
        max_items: query.params.get(AuthorDiversityHistoryMaxItems),
    };
    if !config.is_valid() {
        return None;
    }
    let mut exposures = Vec::new();
    let mut inspected = 0usize;
    // The client currently returns at most 100 batches. Bound malformed or
    // unexpectedly large input as well; never use an order-biased partial scan.
    for batch in &query.served_history {
        inspected += 1;
        if inspected > MAX_EXPOSURES {
            return None;
        }
        let Some(time) = batch.served_time_ms.filter(|&time| {
            time > 0
                && query
                    .request_time_ms
                    .checked_sub(time)
                    .is_some_and(|age| (0..config.window_ms).contains(&age))
        }) else {
            continue;
        };
        for entry in &batch.entries {
            inspected += 1;
            if inspected > MAX_EXPOSURES {
                return None;
            }
            if entry.entity_type != EntityIdType::TWEET {
                continue;
            }
            for ids in entry.item_ids.iter().flatten() {
                inspected += 1;
                if inspected > MAX_EXPOSURES {
                    return None;
                }
                let Some(author_id) = ids.source_author_id.filter(|id| *id > 0) else {
                    continue;
                };
                let Some(post_id) = ids
                    .source_tweet_id
                    .filter(|id| *id > 0)
                    .or_else(|| ids.tweet_id.filter(|id| *id > 0))
                else {
                    continue;
                };
                let item = match entry.sort_index.filter(|position| *position >= 0) {
                    Some(position) => ExposureItem::Position(position),
                    None => ExposureItem::Post(post_id as u64),
                };
                exposures.push(AuthorExposure {
                    served_at_ms: time,
                    request_id: batch.served_id,
                    item,
                    author_id: author_id as u64,
                });
            }
        }
    }
    author_diversity::recent_author_counts(&exposures, query.request_time_ms, config)
}
