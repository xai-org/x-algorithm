use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{
    EnableResponseDiversityStatsExperimentBucket, EnableSemanticIdHydration, SidServerEndpoint,
};
use rand::random;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SidClient;
use xai_candidate_pipeline::component_library::utils::Composition;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::ExperimentBucket;
use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt, global_stats_receiver};
use xai_x_thrift::served_history::{EntityIdType, ItemIds, ServedHistory};

const METRIC_PREFIX: &str = "ResponseDiversity";
const DEFAULT_SAMPLING_RATE: f64 = 0.05;
const TOP_POSITIONS: usize = 10;
const TOP3_POSITIONS: usize = 3;
const CROSS_SESSION_WINDOWS: [(&str, i64); 3] =
    [("3", 3 * 60_000), ("10", 10 * 60_000), ("30", 30 * 60_000)];
const CROSS_PAGE_WINDOWS: [(&str, usize); 3] = [("3", 3), ("10", 10), ("35", 35)];

const STAGE_FINAL: &str = "final";
const STAGE_TOP10: &str = "top10";
const STAGE_TOP3: &str = "top3";
const STAGE_PRE_HEURISTIC: &str = "pre_heuristic";

pub struct ResponseDiversityStatsSideEffect {
    pub sid_client: Arc<dyn SidClient>,
}

struct HistoryPost {
    tweet_id: u64,
    author_id: Option<u64>,
    served_time_ms: i64,
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for ResponseDiversityStatsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };
        let selected = &input.selected_candidates;
        if selected.is_empty() {
            return Ok(());
        }

        let experiment_buckets = input
            .query
            .params
            .experiment_buckets(EnableResponseDiversityStatsExperimentBucket);
        if experiment_buckets.is_empty() && random::<f64>() >= DEFAULT_SAMPLING_RATE {
            return Ok(());
        }
        let unbucketed = ExperimentBucket::new("", "");
        let buckets: Vec<&ExperimentBucket> = if experiment_buckets.is_empty() {
            vec![&unbucketed]
        } else {
            experiment_buckets
        };

        let final_order = sorted_desc(selected.iter(), |c| c.score);
        let top = &final_order[..TOP_POSITIONS.min(final_order.len())];
        let top3 = &final_order[..TOP3_POSITIONS.min(final_order.len())];
        let pre_heuristic =
            pre_heuristic_top_k(selected, &input.non_selected_candidates, selected.len());

        for bucket in &buckets {
            record_stage(receiver.as_ref(), STAGE_FINAL, &final_order, bucket);
            record_stage(receiver.as_ref(), STAGE_TOP10, top, bucket);
            record_stage(receiver.as_ref(), STAGE_TOP3, top3, bucket);
            record_stage(
                receiver.as_ref(),
                STAGE_PRE_HEURISTIC,
                &pre_heuristic,
                bucket,
            );
        }

        let query = &input.query;
        if query.request_type == RequestType::ForYou {
            let history = history_posts(&query.served_history, query.request_time_ms);
            let sids = self.history_sids(query, &history, receiver.as_ref()).await;
            record_cross_session(
                receiver.as_ref(),
                selected,
                &history,
                &sids,
                query.request_time_ms,
                &buckets,
            );
            record_cross_page(receiver.as_ref(), &final_order, &history, &sids, &buckets);
        }

        Ok(())
    }
}

impl ResponseDiversityStatsSideEffect {
    async fn history_sids(
        &self,
        query: &ScoredPostsQuery,
        history: &[HistoryPost],
        receiver: &dyn StatsReceiverExt,
    ) -> FxHashMap<u64, Vec<i32>> {
        let wanted: FxHashSet<u64> = history.iter().map(|p| p.tweet_id).collect();
        let mut sids: FxHashMap<u64, Vec<i32>> = query
            .cached_posts
            .iter()
            .filter(|c| wanted.contains(&c.get_original_tweet_id()))
            .filter_map(|c| Some((c.get_original_tweet_id(), c.semantic_ids.clone()?)))
            .collect();
        let from_cached_posts = sids.len();

        let endpoint = query.params.get(SidServerEndpoint);
        let mut from_lookup = 0;
        if query.params.get(EnableSemanticIdHydration) && !endpoint.is_empty() {
            let mut missing: Vec<u64> = wanted
                .iter()
                .filter(|id| !sids.contains_key(id))
                .copied()
                .collect();
            missing.sort_unstable();
            let post_ids: Vec<i64> = missing.iter().map(|&id| id as i64).collect();
            if !post_ids.is_empty()
                && let Some(looked_up) = self.sid_client.lookup(&endpoint, &post_ids).await
            {
                for (id, codes) in missing.into_iter().zip(looked_up) {
                    if let Some(codes) = codes {
                        sids.insert(id, codes);
                        from_lookup += 1;
                    }
                }
            }
        }

        let metric = format!("{METRIC_PREFIX}.CrossSessionHistorySids");
        for (source, count) in [
            ("cached_posts", from_cached_posts),
            ("lookup", from_lookup),
            ("missing", wanted.len() - sids.len()),
        ] {
            receiver.incr(&metric, &[("source", source)], count as u64);
        }
        sids
    }
}

fn history_posts(served_history: &[ServedHistory], now_ms: i64) -> Vec<HistoryPost> {
    let since_ms = now_ms - CROSS_SESSION_WINDOWS[CROSS_SESSION_WINDOWS.len() - 1].1;
    let mut posts = Vec::new();
    for response in served_history {
        let Some(served_time_ms) = response.served_time_ms.filter(|&t| t >= since_ms) else {
            continue;
        };
        let mut main_entries: BTreeMap<i64, &ItemIds> = BTreeMap::new();
        for entry in &response.entries {
            if entry.entity_type != EntityIdType::TWEET {
                continue;
            }
            let Some(sort_index) = entry.sort_index else {
                continue;
            };
            for ids in entry.item_ids.iter().flatten() {
                let main = main_entries.entry(sort_index).or_insert(ids);
                if ids.tweet_id > main.tweet_id {
                    *main = ids;
                }
            }
        }
        posts.extend(main_entries.into_values().filter_map(|ids| {
            Some(HistoryPost {
                tweet_id: u64::try_from(ids.source_tweet_id.or(ids.tweet_id)?).ok()?,
                author_id: ids.source_author_id.and_then(|id| u64::try_from(id).ok()),
                served_time_ms,
            })
        }));
    }
    posts
}

fn sorted_desc<'a>(
    candidates: impl Iterator<Item = &'a PostCandidate>,
    key: impl Fn(&PostCandidate) -> Option<f64>,
) -> Vec<&'a PostCandidate> {
    let mut sorted: Vec<&PostCandidate> = candidates.collect();
    sorted.sort_by(|a, b| {
        key(b)
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&key(a).unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(Ordering::Equal)
    });
    sorted
}

fn pre_heuristic_top_k<'a>(
    selected: &'a [PostCandidate],
    non_selected: &'a [PostCandidate],
    k: usize,
) -> Vec<&'a PostCandidate> {
    let mut ranked = sorted_desc(selected.iter().chain(non_selected.iter()), |c| {
        c.weighted_score
    });
    ranked.truncate(k);
    ranked
}

fn record_stage(
    receiver: &dyn StatsReceiverExt,
    stage: &str,
    posts: &[&PostCandidate],
    bucket: &ExperimentBucket,
) {
    if posts.is_empty() {
        return;
    }
    let scopes: [(&str, &str); 3] = [
        ("stage", stage),
        ("ddg", &bucket.experiment),
        ("bucket", &bucket.bucket),
    ];

    let authors = Composition::from_keys(posts.iter().map(|c| c.author_id));
    let sources = Composition::from_keys(posts.iter().map(|c| c.served_type));
    let sid_l1 = Composition::from_keys(posts.iter().filter_map(|c| c.semantic_id_prefix(1)));
    let sid_l2 = Composition::from_keys(posts.iter().filter_map(|c| c.semantic_id_prefix(2)));
    let in_network =
        posts.iter().filter(|c| c.in_network == Some(true)).count() as f64 / posts.len() as f64;
    let sid_coverage = sid_l1.size as f64 / posts.len() as f64;

    let observe = |name: &str, value: f64, buckets: HistogramBuckets| {
        receiver.observe(&format!("{METRIC_PREFIX}.{name}"), &scopes, value, buckets);
    };
    observe("Size", posts.len() as f64, HistogramBuckets::Bucket0To50);
    observe(
        "UniqueAuthors",
        authors.unique as f64,
        HistogramBuckets::Bucket0To50,
    );
    observe(
        "UniqueAuthorRatio",
        authors.unique_ratio(),
        HistogramBuckets::Bucket0To1,
    );
    observe(
        "MaxAuthorShare",
        authors.max_share(),
        HistogramBuckets::Bucket0To1,
    );
    observe("AuthorHhi", authors.hhi, HistogramBuckets::Bucket0To1);
    observe(
        "AuthorEntropyNorm",
        authors.entropy_norm,
        HistogramBuckets::Bucket0To1,
    );
    observe(
        "UniqueSources",
        sources.unique as f64,
        HistogramBuckets::Bucket0To50,
    );
    observe(
        "SourceEntropyNorm",
        sources.entropy_norm,
        HistogramBuckets::Bucket0To1,
    );
    observe("InNetworkShare", in_network, HistogramBuckets::Bucket0To1);
    observe("SidCoverage", sid_coverage, HistogramBuckets::Bucket0To1);
    if sid_l1.size > 0 {
        observe(
            "UniqueSidL1",
            sid_l1.unique as f64,
            HistogramBuckets::Bucket0To50,
        );
        observe(
            "MaxSidL1Share",
            sid_l1.max_share(),
            HistogramBuckets::Bucket0To1,
        );
        observe(
            "SidL1EntropyNorm",
            sid_l1.entropy_norm,
            HistogramBuckets::Bucket0To1,
        );
    }
    if sid_l2.size > 0 {
        observe(
            "UniqueSidL2",
            sid_l2.unique as f64,
            HistogramBuckets::Bucket0To50,
        );
        observe(
            "SidL2EntropyNorm",
            sid_l2.entropy_norm,
            HistogramBuckets::Bucket0To1,
        );
    }
}

fn record_cross_session(
    receiver: &dyn StatsReceiverExt,
    page: &[PostCandidate],
    history: &[HistoryPost],
    sids: &FxHashMap<u64, Vec<i32>>,
    now_ms: i64,
    buckets: &[&ExperimentBucket],
) {
    for (window, window_ms) in CROSS_SESSION_WINDOWS {
        let in_window: Vec<&HistoryPost> = history
            .iter()
            .filter(|p| p.served_time_ms >= now_ms - window_ms)
            .collect();
        let history_codes: Vec<&[i32]> = in_window
            .iter()
            .filter_map(|p| sids.get(&p.tweet_id).map(Vec::as_slice))
            .collect();
        let sid_l1 = Composition::from_keys(
            page.iter()
                .filter_map(|c| c.semantic_id_prefix(1))
                .chain(history_codes.iter().filter_map(|codes| codes.get(..1))),
        );
        let sid_l2 = Composition::from_keys(
            page.iter()
                .filter_map(|c| c.semantic_id_prefix(2))
                .chain(history_codes.iter().filter_map(|codes| codes.get(..2))),
        );
        let authors = Composition::from_keys(
            page.iter()
                .map(|c| c.get_original_author_id())
                .chain(in_window.iter().filter_map(|p| p.author_id)),
        );
        let entropies = [
            ("CrossSessionSidL1EntropyNorm", sid_l1),
            ("CrossSessionSidL2EntropyNorm", sid_l2),
            ("CrossSessionAuthorEntropyNorm", authors),
        ];

        for bucket in buckets {
            let scopes = [
                ("window", window),
                ("ddg", &bucket.experiment),
                ("bucket", &bucket.bucket),
            ];
            receiver.observe_expo(
                &format!("{METRIC_PREFIX}.CrossSessionHistoryPosts"),
                &scopes,
                in_window.len() as f64,
            );
            for (name, composition) in &entropies {
                if composition.size > 0 {
                    receiver.observe(
                        &format!("{METRIC_PREFIX}.{name}"),
                        &scopes,
                        composition.entropy_norm,
                        HistogramBuckets::Bucket0To1,
                    );
                }
            }
        }
    }
}

fn record_cross_page(
    receiver: &dyn StatsReceiverExt,
    page: &[&PostCandidate],
    history: &[HistoryPost],
    sids: &FxHashMap<u64, Vec<i32>>,
    buckets: &[&ExperimentBucket],
) {
    let mut served: Vec<&HistoryPost> = history.iter().collect();
    served.sort_by_key(|p| p.served_time_ms);
    let reach = CROSS_PAGE_WINDOWS[CROSS_PAGE_WINDOWS.len() - 1].1 - 1;
    let sequence: Vec<[Option<&[i32]>; 2]> = served[served.len().saturating_sub(reach)..]
        .iter()
        .map(|p| {
            let codes = sids.get(&p.tweet_id);
            [1, 2].map(|level| codes.and_then(|codes| codes.get(..level)))
        })
        .chain(
            page.iter()
                .map(|c| [1, 2].map(|level| c.semantic_id_prefix(level))),
        )
        .collect();
    let first = sequence.len() - page.len();

    for (window, size) in CROSS_PAGE_WINDOWS {
        let (mut entropy, mut distinct, mut max_share, mut positions) =
            ([0.0; 2], [0.0; 2], [0.0; 2], [0; 2]);
        for end in first + 1..=sequence.len() {
            let recent = &sequence[end.saturating_sub(size)..end];
            for level in 0..2 {
                let topics = Composition::from_keys(recent.iter().filter_map(|codes| codes[level]));
                if topics.size > 0 {
                    entropy[level] += topics.entropy_norm;
                    distinct[level] += topics.unique_ratio();
                    max_share[level] += topics.max_share();
                    positions[level] += 1;
                }
            }
        }
        let means = [
            ("CrossPageSidL1EntropyNorm", entropy[0], positions[0]),
            ("CrossPageSidL1DistinctRatio", distinct[0], positions[0]),
            ("CrossPageSidL1MaxShare", max_share[0], positions[0]),
            ("CrossPageSidL2DistinctRatio", distinct[1], positions[1]),
            ("CrossPageSidL2MaxShare", max_share[1], positions[1]),
        ];
        for bucket in buckets {
            let scopes = [
                ("window", window),
                ("ddg", &bucket.experiment),
                ("bucket", &bucket.bucket),
            ];
            for (name, total, positions) in means {
                if positions > 0 {
                    receiver.observe(
                        &format!("{METRIC_PREFIX}.{name}"),
                        &scopes,
                        total / positions as f64,
                        HistogramBuckets::Bucket0To1,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};
    use xai_x_thrift::served_history::{EntryWithItemIds, RequestType as ServedRequestType};

    const NOW_MS: i64 = 1_800_000_000_000;
    const ENDPOINT: &str = "xds://sid";

    type Recorded = (String, Vec<(String, String)>, f64);

    #[derive(Default)]
    struct Recorder(Mutex<Vec<Recorded>>);

    impl Recorder {
        fn push(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
            let scopes = scopes
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            self.0
                .lock()
                .unwrap()
                .push((name.to_string(), scopes, value));
        }

        fn value(&self, name: &str, scopes: &[(&str, &str)]) -> Option<f64> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .find(|(n, s, _)| {
                    n == name
                        && scopes
                            .iter()
                            .all(|(k, v)| s.iter().any(|(sk, sv)| sk == k && sv == v))
                })
                .map(|(_, _, value)| *value)
        }
    }

    impl StatsReceiverExt for Recorder {
        fn incr(&self, name: &str, scopes: &[(&str, &str)], value: u64) {
            self.push(name, scopes, value as f64);
        }
        fn observe(&self, name: &str, scopes: &[(&str, &str)], value: f64, _: HistogramBuckets) {
            self.push(name, scopes, value);
        }
        fn observe_expo(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
            self.push(name, scopes, value);
        }
        fn observe_vm(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
            self.push(name, scopes, value);
        }
        fn gauge(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
            self.push(name, scopes, value);
        }
    }

    #[derive(Default)]
    struct CountingSidClient {
        calls: Mutex<Vec<Vec<i64>>>,
    }

    #[async_trait]
    impl SidClient for CountingSidClient {
        async fn lookup(&self, _endpoint: &str, post_ids: &[i64]) -> Option<Vec<Option<Vec<i32>>>> {
            self.calls.lock().unwrap().push(post_ids.to_vec());
            Some(
                post_ids
                    .iter()
                    .map(|&id| (id == 3).then(|| vec![30, 31]))
                    .collect(),
            )
        }
    }

    fn post(sort_index: i64, tweet_id: i64, author_id: i64) -> EntryWithItemIds {
        EntryWithItemIds {
            entity_type: EntityIdType::TWEET,
            sort_index: Some(sort_index),
            size: None,
            item_ids: Some(vec![ItemIds {
                tweet_id: Some(tweet_id),
                source_author_id: Some(author_id),
                ..Default::default()
            }]),
        }
    }

    fn response(
        age_secs: i64,
        request_type: ServedRequestType,
        entries: Vec<EntryWithItemIds>,
    ) -> ServedHistory {
        ServedHistory {
            request_type,
            entries,
            served_id: None,
            served_time_ms: Some(NOW_MS - age_secs * 1_000),
        }
    }

    fn candidate(score: f64, author_id: u64, semantic_ids: Vec<i32>) -> PostCandidate {
        PostCandidate {
            score: Some(score),
            author_id,
            semantic_ids: Some(semantic_ids),
            ..Default::default()
        }
    }

    #[test]
    fn entropies_cover_the_page_and_each_window_including_polling() {
        let served_history = vec![
            response(30, ServedRequestType::POLLING, vec![post(0, 4, 103)]),
            response(
                5 * 60,
                ServedRequestType::INITIAL,
                vec![post(0, 1, 100), post(1, 2, 105)],
            ),
            response(20 * 60, ServedRequestType::OLDER, vec![post(0, 3, 102)]),
        ];
        let sids: FxHashMap<u64, Vec<i32>> = [
            (1, vec![1, 10]),
            (2, vec![2, 20]),
            (3, vec![3, 30]),
            (4, vec![4, 40]),
        ]
        .into_iter()
        .collect();
        let page = vec![
            candidate(4.0, 100, vec![1, 10]),
            candidate(3.0, 105, vec![3, 99]),
            candidate(2.0, 103, vec![4, 40]),
            candidate(1.0, 106, vec![5, 50]),
        ];
        let bucket = ExperimentBucket::new("20140", "treatment");
        let recorder = Recorder::default();
        let history = history_posts(&served_history, NOW_MS);

        record_cross_session(&recorder, &page, &history, &sids, NOW_MS, &[&bucket]);
        record_cross_page(
            &recorder,
            &page.iter().collect::<Vec<_>>(),
            &history,
            &sids,
            &[&bucket],
        );

        let value = |name: &str, window: &str| {
            recorder
                .value(
                    &format!("ResponseDiversity.CrossSession{name}"),
                    &[
                        ("window", window),
                        ("ddg", "20140"),
                        ("bucket", "treatment"),
                    ],
                )
                .unwrap()
        };
        for (window, posts, expected) in [
            ("3", 1.0, [0.827729, 0.827729, 0.827729]),
            ("10", 3.0, [0.796453, 0.796453, 0.694680]),
            ("30", 4.0, [0.75, 0.833333, 0.75]),
        ] {
            assert_eq!(value("HistoryPosts", window), posts, "window {window}");
            for (name, expected) in ["SidL1EntropyNorm", "SidL2EntropyNorm", "AuthorEntropyNorm"]
                .into_iter()
                .zip(expected)
            {
                let got = value(name, window);
                assert!(
                    (got - expected).abs() < 1e-6,
                    "{name} window {window}: {got}"
                );
            }
        }
        for (window, expected) in [
            ("3", [1.0, 1.0, 0.333333, 1.0, 0.333333]),
            ("10", [0.753627, 0.665774, 0.317262, 0.774405, 0.317262]),
            ("35", [0.753627, 0.665774, 0.317262, 0.774405, 0.317262]),
        ] {
            for (name, expected) in [
                "SidL1EntropyNorm",
                "SidL1DistinctRatio",
                "SidL1MaxShare",
                "SidL2DistinctRatio",
                "SidL2MaxShare",
            ]
            .into_iter()
            .zip(expected)
            {
                let got = recorder
                    .value(
                        &format!("ResponseDiversity.CrossPage{name}"),
                        &[
                            ("window", window),
                            ("ddg", "20140"),
                            ("bucket", "treatment"),
                        ],
                    )
                    .unwrap();
                assert!(
                    (got - expected).abs() < 1e-6,
                    "{name} window {window}: {got}"
                );
            }
        }
    }

    #[tokio::test]
    async fn history_sids_read_cached_posts_then_one_lookup() {
        let fs = FeatureSwitches::new(vec![]).unwrap();
        let mut results = fs.match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_semantic_id_hydration".to_string(),
            "true",
        );
        results.override_fs("rust_home_mixer_sid_server_endpoint".to_string(), ENDPOINT);
        let query = ScoredPostsQuery {
            params: results.into(),
            cached_posts: vec![PostCandidate {
                tweet_id: 1,
                semantic_ids: Some(vec![10, 11]),
                ..Default::default()
            }],
            ..Default::default()
        };
        let sid_client = Arc::new(CountingSidClient::default());
        let side_effect = ResponseDiversityStatsSideEffect {
            sid_client: sid_client.clone(),
        };
        let history: Vec<HistoryPost> = (1..=3)
            .map(|tweet_id| HistoryPost {
                tweet_id,
                author_id: None,
                served_time_ms: NOW_MS,
            })
            .collect();
        let recorder = Recorder::default();

        let sids = side_effect.history_sids(&query, &history, &recorder).await;

        assert_eq!(*sid_client.calls.lock().unwrap(), vec![vec![2, 3]]);
        assert_eq!(
            sids,
            [(1, vec![10, 11]), (3, vec![30, 31])].into_iter().collect()
        );
        for source in ["cached_posts", "lookup", "missing"] {
            assert_eq!(
                recorder.value(
                    "ResponseDiversity.CrossSessionHistorySids",
                    &[("source", source)]
                ),
                Some(1.0)
            );
        }
    }
}
