//! PR-F5 ControlledSourceReexposure.
//!
//! Interaction with [`super::previously_served_posts_filter::PreviouslyServedPostsFilter`]:
//! - When `EnableControlledSourceReexposure` is false OR `MaxServesK <= 1`,
//!   this filter is disabled and PreviouslyServed keeps today's burn behavior.
//! - When Enable && K > 1, PreviouslyServed pass-throughs served hits and this
//!   filter owns: K cap, optional request gap, InNetworkOnly, prefer-original
//!   within slate, ≤1 card per `source_key` (quotes share bucket with RTs).
//! AgeFilter / OON-RT / VF are untouched.

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    ControlledSourceReexposureInNetworkOnly, ControlledSourceReexposureMaxServesK,
    ControlledSourceReexposureMinRequestGap, EnableControlledSourceReexposure,
};
use crate::util::candidates_util::{related_post_ids_iter, source_key};
use crate::util::serve_card_classify::record_serve_card_type;
use std::collections::{HashMap, HashSet};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct ControlledSourceReexposureFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for ControlledSourceReexposureFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableControlledSourceReexposure)
            && query.params.get(ControlledSourceReexposureMaxServesK) > 1
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let k = query.params.get(ControlledSourceReexposureMaxServesK);
        let min_gap = query.params.get(ControlledSourceReexposureMinRequestGap);
        let in_network_only = query.params.get(ControlledSourceReexposureInNetworkOnly);

        let served_ids: HashSet<u64> = query.served_ids.iter().copied().collect();
        let mut serve_count: HashMap<u64, u32> = HashMap::new();
        for &id in &query.served_ids {
            *serve_count.entry(id).or_default() += 1;
        }

        let history_len = query.served_history.len() as u32;
        let mut last_serve_request: HashMap<u64, u32> = HashMap::new();
        for (req_idx, sh) in query.served_history.iter().enumerate() {
            for entry in &sh.entries {
                for ids in entry.item_ids.iter().flatten() {
                    for opt in [ids.tweet_id, ids.source_tweet_id] {
                        if let Some(id) = opt {
                            last_serve_request.insert(id as u64, req_idx as u32);
                        }
                    }
                }
            }
        }

        let mut groups: HashMap<u64, Vec<PostCandidate>> = HashMap::new();
        let mut order: Vec<u64> = Vec::new();
        for c in candidates {
            let sk = source_key(&c);
            groups.entry(sk).or_insert_with(|| {
                order.push(sk);
                Vec::new()
            });
            groups.get_mut(&sk).unwrap().push(c);
        }

        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for sk in order {
            let group = groups.remove(&sk).unwrap_or_default();
            let already_served = served_ids.contains(&sk)
                || group
                    .iter()
                    .any(|c| related_post_ids_iter(c).any(|id| served_ids.contains(&id)));

            if already_served {
                let count = *serve_count.get(&sk).unwrap_or(&0);
                if count >= k {
                    removed.extend(group);
                    continue;
                }
                if min_gap > 0 {
                    if let Some(&last_req) = last_serve_request.get(&sk) {
                        if history_len.saturating_sub(last_req) < min_gap {
                            removed.extend(group);
                            continue;
                        }
                    }
                }

                let (eligible, ineligible): (Vec<_>, Vec<_>) =
                    group.into_iter().partition(|c| {
                        if !in_network_only {
                            return true;
                        }
                        let is_amp =
                            c.retweeted_tweet_id.is_some() || c.quoted_tweet_id.is_some();
                        if is_amp {
                            c.in_network == Some(true)
                        } else {
                            c.in_network != Some(false)
                        }
                    });
                removed.extend(ineligible);
                if eligible.is_empty() {
                    continue;
                }
                let (chosen, leftovers) = pick_prefer_original(eligible);
                removed.extend(leftovers);
                if let Some(chosen) = chosen {
                    let _ = record_serve_card_type(&chosen);
                    kept.push(chosen);
                }
            } else {
                let (chosen, leftovers) = pick_prefer_original(group);
                removed.extend(leftovers);
                if let Some(chosen) = chosen {
                    let _ = record_serve_card_type(&chosen);
                    kept.push(chosen);
                }
            }
        }

        FilterResult { kept, removed }
    }
}

/// Prefer native/original over RT over quote; return ≤1 chosen + leftovers.
fn pick_prefer_original(
    group: Vec<PostCandidate>,
) -> (Option<PostCandidate>, Vec<PostCandidate>) {
    if group.is_empty() {
        return (None, Vec::new());
    }
    let natives: Vec<_> = group
        .iter()
        .filter(|c| c.retweeted_tweet_id.is_none() && c.quoted_tweet_id.is_none())
        .collect();
    let chosen_id = if let Some(n) = natives.first() {
        n.tweet_id
    } else {
        let rts: Vec<_> = group
            .iter()
            .filter(|c| c.retweeted_tweet_id.is_some())
            .collect();
        rts.first()
            .map(|c| c.tweet_id)
            .unwrap_or(group[0].tweet_id)
    };
    let mut chosen = None;
    let mut leftovers = Vec::new();
    for c in group {
        if c.tweet_id == chosen_id && chosen.is_none() {
            chosen = Some(c);
        } else {
            leftovers.push(c);
        }
    }
    (chosen, leftovers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    fn cand(
        tweet_id: u64,
        retweeted: Option<u64>,
        quoted: Option<u64>,
        in_network: Option<bool>,
    ) -> PostCandidate {
        PostCandidate {
            tweet_id,
            retweeted_tweet_id: retweeted,
            quoted_tweet_id: quoted,
            in_network,
            ..Default::default()
        }
    }

    fn params(enable: bool, k: u32, gap: u32, in_only: bool) -> Params {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_controlled_source_reexposure".to_string(),
            if enable { "true" } else { "false" },
        );
        results.override_fs(
            "rust_home_mixer_controlled_source_reexposure_max_serves_k".to_string(),
            &k.to_string(),
        );
        results.override_fs(
            "rust_home_mixer_controlled_source_reexposure_min_request_gap".to_string(),
            &gap.to_string(),
        );
        results.override_fs(
            "rust_home_mixer_controlled_source_reexposure_in_network_only".to_string(),
            if in_only { "true" } else { "false" },
        );
        results.into()
    }

    fn query(served: Vec<u64>, enable: bool, k: u32, gap: u32, in_only: bool) -> ScoredPostsQuery {
        ScoredPostsQuery {
            served_ids: served,
            params: params(enable, k, gap, in_only),
            ..Default::default()
        }
    }

    #[test]
    fn enable_false_when_flag_off_or_k_one() {
        let f = ControlledSourceReexposureFilter;
        assert!(!f.enable(&query(vec![], false, 2, 0, true)));
        assert!(!f.enable(&query(vec![], true, 1, 0, true)));
        assert!(f.enable(&query(vec![], true, 2, 0, true)));
    }

    #[test]
    fn enable_true_k2_allows_one_re_serve() {
        let f = ControlledSourceReexposureFilter;
        let q = query(vec![100], true, 2, 0, true);
        let candidates = vec![
            cand(201, Some(100), None, Some(true)),
            cand(202, Some(100), None, Some(true)),
        ];
        let result = f.filter(&q, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].retweeted_tweet_id, Some(100));
    }

    #[test]
    fn quotes_share_source_key_with_rt() {
        let f = ControlledSourceReexposureFilter;
        let q = query(vec![], true, 2, 0, true);
        let candidates = vec![
            cand(201, Some(100), None, Some(true)),
            cand(301, None, Some(100), Some(true)),
        ];
        let result = f.filter(&q, candidates);
        assert_eq!(result.kept.len(), 1, "≤1 per source across RT+quote");
    }

    #[test]
    fn prefer_original_in_slate() {
        let f = ControlledSourceReexposureFilter;
        let q = query(vec![], true, 2, 0, true);
        let candidates = vec![
            cand(201, Some(100), None, Some(true)),
            cand(100, None, None, Some(true)),
        ];
        let result = f.filter(&q, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 100);
    }

    #[test]
    fn in_network_only_drops_oon_amp_on_reexposure() {
        let f = ControlledSourceReexposureFilter;
        let q = query(vec![100], true, 2, 0, true);
        let candidates = vec![cand(201, Some(100), None, Some(false))];
        let result = f.filter(&q, candidates);
        assert!(result.kept.is_empty());
    }

    #[test]
    fn k_exhausted_drops_re_serve() {
        let f = ControlledSourceReexposureFilter;
        let q = query(vec![100, 100], true, 2, 0, true);
        let candidates = vec![cand(201, Some(100), None, Some(true))];
        let result = f.filter(&q, candidates);
        assert!(result.kept.is_empty());
    }
}
