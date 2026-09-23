// run.py injects verbatim production bodies. Everything outside the injection
// points/modules is a test double. This is NOT a production service build.
#![allow(dead_code, unused_imports, private_interfaces, non_camel_case_types)]
extern crate self as xai_x_thrift;
#[cfg(test)]
#[path = "@SCHEDULING@"]
mod scheduling;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
type FxHashMap<K, V> = HashMap<K, V>;
const NEGATIVE_SCORES_OFFSET: f64 = 0.001;

trait Param {
    type Value: std::str::FromStr;
    const KEY: &'static str;
    fn default_value() -> Self::Value;
}
#[derive(Clone, Default)]
struct Params(HashMap<String, String>);
impl Params {
    fn get<P: Param>(&self, _: P) -> P::Value {
        self.0
            .get(P::KEY)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(P::default_value)
    }
    fn set<P: Param>(&mut self, _: P, value: impl ToString) {
        self.0.insert(P::KEY.to_string(), value.to_string());
    }
}
/* PARAMS */
mod params {
    pub(super) use super::*;
}

mod served_history {
    #[derive(Clone, Copy, PartialEq)]
    pub enum EntityIdType {
        TWEET,
        PROMOTED_TWEET,
        WHO_TO_FOLLOW,
        ANNOTATION,
    }
    #[derive(Clone, Copy)]
    pub enum RequestType {
        INITIAL,
    }
    #[derive(Clone)]
    pub struct ServedHistory {
        pub served_time_ms: Option<i64>,
        pub served_id: Option<i64>,
        pub entries: Vec<EntryWithItemIds>,
        pub request_type: RequestType,
    }
    #[derive(Clone)]
    pub struct EntryWithItemIds {
        pub entity_type: EntityIdType,
        pub sort_index: Option<i64>,
        pub size: Option<i64>,
        pub item_ids: Option<Vec<ItemIds>>,
    }
    #[derive(Clone, Default)]
    pub struct ItemIds {
        pub tweet_id: Option<i64>,
        pub source_tweet_id: Option<i64>,
        pub source_author_id: Option<i64>,
        pub quote_tweet_id: Option<i64>,
        pub quote_author_id: Option<i64>,
        pub in_reply_to_tweet_id: Option<i64>,
        pub in_reply_to_author_id: Option<i64>,
        pub article_id: Option<i64>,
        pub tweet_score: Option<TweetScore>,
        pub entry_id_to_replace: Option<String>,
        pub user_id: Option<i64>,
        pub impression_id: Option<String>,
    }
    #[derive(Clone)]
    pub enum TweetScore {
        TweetScoreV1(TweetScoreV1),
    }
    #[derive(Clone)]
    pub struct TweetScoreV1 {
        pub score: super::OrderedFloat,
        pub served_type: Option<String>,
        pub debug_info: Option<()>,
        pub prediction_request_id: Option<i64>,
        pub topics: Option<()>,
        pub tags: Option<()>,
        pub predicted_scores: Option<()>,
    }
}
use served_history::{
    EntityIdType, EntryWithItemIds, ItemIds, ServedHistory, TweetScore, TweetScoreV1,
};
#[derive(Clone)]
struct OrderedFloat(f64);
impl From<f64> for OrderedFloat {
    fn from(x: f64) -> Self {
        Self(x)
    }
}
enum ServedType {
    Undefined,
}
impl TryFrom<i32> for ServedType {
    type Error = ();
    fn try_from(_: i32) -> Result<Self, ()> {
        Ok(Self::Undefined)
    }
}
impl ServedType {
    fn as_str_name(&self) -> &'static str {
        "UNDEFINED"
    }
}
#[derive(Default)]
struct ScoredPost {
    tweet_id: u64,
    author_id: u64,
    retweeted_tweet_id: u64,
    retweeted_user_id: u64,
    in_reply_to_tweet_id: u64,
    prediction_request_id: u64,
    served_type: i32,
    score: f32,
    ancestors: Vec<u64>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SlateContext {
    k: u32,
    fatigue: f64,
}
#[derive(Clone, Debug, Default)]
struct PhoenixScores {
    favorite_score: Option<f64>,
}
#[derive(Clone, Debug, Default)]
struct PostCandidate {
    tweet_id: u64,
    author_id: u64,
    retweeted_tweet_id: Option<u64>,
    retweeted_user_id: Option<u64>,
    in_reply_to_tweet_id: Option<u64>,
    slate_context: Option<SlateContext>,
    served_slate_context: Option<SlateContext>,
    in_network: Option<bool>,
    weighted_score: Option<f64>,
    score: Option<f64>,
    base_positive: f64,
    base_negative: f64,
    phoenix_scores: PhoenixScores,
}
#[derive(Clone, Copy, Default, PartialEq)]
enum RequestType {
    #[default]
    ForYou,
    RankedFollowing,
}
#[derive(Clone, Default)]
struct ScoredPostsQuery {
    user_id: u64,
    client_app_id: i32,
    request_type: RequestType,
    params: Params,
    request_time_ms: i64,
    served_history: Vec<ServedHistory>,
    served_history_loaded: bool,
    served_ids: Vec<u64>,
    has_cached_posts: bool,
    who_to_follow_eligible: bool,
    feed_survey_eligible: bool,
}
mod models {
    pub mod query {
        pub(crate) use crate::{RequestType, ScoredPostsQuery};
    }
}
mod util {
    #[path = "@AUTHOR_DIVERSITY@"]
    pub mod author_diversity;
    #[path = "@AUTHOR_HISTORY@"]
    pub mod author_history;
}
struct ScoringWeights {
    total_sum: f64,
    negative_sum: f64,
}
impl ScoringWeights {
    fn from_params(_: &Params) -> Self {
        Self {
            total_sum: 1.0,
            negative_sum: 1.0,
        }
    }
    fn perturbed(self, _: &ScoredPostsQuery) -> Self {
        self
    }
}
struct AuthorColdStart;
impl AuthorColdStart {
    fn apply(&self, _: &ScoredPostsQuery, _: &[PostCandidate], scores: &[f64]) -> Vec<f64> {
        scores.to_vec()
    }
}
struct RankingScorer {
    author_cold_start: AuthorColdStart,
}
impl RankingScorer {
    // Explicit controlled inputs, replacing ML/weighted action predictions.
    fn compute_weighted_parts(
        _: &ScoringWeights,
        _: &ScoredPostsQuery,
        c: &PostCandidate,
    ) -> (f64, f64) {
        (
            c.base_positive + c.phoenix_scores.favorite_score.unwrap_or(0.0),
            c.base_negative,
        )
    }
    fn effective_oon_weight(_: &ScoredPostsQuery) -> f64 {
        1.0
    }
    /* SCORER */
}
struct FilterResult<C> {
    kept: Vec<C>,
    removed: Vec<C>,
}
struct PreviouslyServedPostsFilter;
impl PreviouslyServedPostsFilter {
    /* FILTER */
}
/* RELATED_IDS */
/* WRITER */
/* STRING_CASE */
/* HISTORY_FUNCTIONS */
struct TimelineType;
impl TimelineType {
    fn for_request(_: RequestType) -> Self {
        Self
    }
}
fn app_id_to_served_history_id(id: i32) -> i32 {
    id
}
#[derive(Default)]
struct MockHistory {
    batches: Mutex<HashMap<u64, Vec<ServedHistory>>>,
    fail: bool,
}
impl MockHistory {
    async fn get_recent(
        &self,
        user: u64,
        _: TimelineType,
        _: i32,
    ) -> Result<Vec<ServedHistory>, String> {
        if self.fail {
            return Err("injected failure".into());
        }
        Ok(self
            .batches
            .lock()
            .unwrap()
            .get(&user)
            .cloned()
            .unwrap_or_default())
    }
}
struct ServedHistoryQueryHydrator {
    client: Arc<MockHistory>,
}
impl ServedHistoryQueryHydrator {
    /* HYDRATOR */
}
fn ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(v) => v,
        _ => panic!("fixture futures must be ready"),
    }
}

#[cfg(test)]
mod upstream_regressions {
    use super::*;
    fn test_scorer() -> RankingScorer {
        RankingScorer {
            author_cold_start: AuthorColdStart,
        }
    }
    fn query_with_flags(flags: &[(&str, &str)]) -> ScoredPostsQuery {
        let mut q = ScoredPostsQuery::default();
        for &(k, v) in flags {
            q.params.0.insert(k.to_string(), v.to_string());
        }
        q
    }
    /* UPSTREAM_TESTS */
    #[test]
    fn scorer_regression() {
        ready(recent_author_is_penalized_in_both_scoring_paths());
    }
    #[test]
    fn original_key_regression() {
        history_and_pool_use_the_same_original_author_key();
    }
}

#[cfg(test)]
mod contracts {
    use super::*;
    fn query() -> ScoredPostsQuery {
        let mut q = ScoredPostsQuery {
            user_id: 7,
            request_time_ms: 1_000_000,
            served_history_loaded: true,
            ..Default::default()
        };
        q.params.set(EnableCrossRequestAuthorDiversity, true);
        q.params.set(AuthorDiversityHistoryWeight, 1.0);
        q
    }
    #[test]
    fn production_default_remains_control() {
        assert!(!ScoredPostsQuery::default()
            .params
            .get(EnableCrossRequestAuthorDiversity));
    }
    fn candidate(id: u64, author: u64) -> PostCandidate {
        PostCandidate {
            tweet_id: id,
            author_id: author,
            base_positive: 100.0,
            in_network: Some(true),
            ..Default::default()
        }
    }
    fn batch(request: i64, time: i64, post: &ScoredPost) -> ServedHistory {
        ServedHistory {
            request_type: served_history::RequestType::INITIAL,
            served_id: Some(request),
            served_time_ms: Some(time),
            entries: build_post_entries(post, 0),
        }
    }
    fn post(id: u64, author: u64) -> ScoredPost {
        ScoredPost {
            tweet_id: id,
            author_id: author,
            ..Default::default()
        }
    }
    fn coefficients(q: &ScoredPostsQuery, c: &[PostCandidate]) -> Vec<f64> {
        RankingScorer::author_diversity_multipliers(q, c, &vec![100.0; c.len()])
    }
    fn score(q: &ScoredPostsQuery, c: &[PostCandidate]) -> Vec<f64> {
        ready(
            RankingScorer {
                author_cold_start: AuthorColdStart,
            }
            .score(q, c),
        )
        .into_iter()
        .map(|r| r.unwrap().score.unwrap())
        .collect()
    }
    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}")
    }

    #[test]
    fn served_writer_reader_and_scorer_preserve_author_history_across_requests() {
        let client = Arc::new(MockHistory::default());
        client
            .batches
            .lock()
            .unwrap()
            .insert(7, vec![batch(1, 1_000_000, &post(10, 1))]);
        let h = ServedHistoryQueryHydrator { client };
        let mut q = query();
        q.served_history_loaded = false;
        let hydrated = ready(h.hydrate(&q)).unwrap();
        h.update(&mut q, hydrated);
        assert!(q.served_history_loaded);
        assert_eq!(q.served_ids, vec![10]);
        let filtered = PreviouslyServedPostsFilter.filter(
            &q,
            vec![candidate(10, 1), candidate(11, 1), candidate(12, 2)],
        );
        assert_eq!(filtered.removed.len(), 1);
        let q_passed_to_inner_pipeline = q.clone(); // ScoredPostsSource clones the hydrated query.
        assert_eq!(
            coefficients(&q_passed_to_inner_pipeline, &filtered.kept),
            vec![0.625, 1.0]
        );
        for pre_offset in [false, true] {
            q.params.set(MultiplierPreOffset, pre_offset);
            let s = score(&q, &filtered.kept);
            assert!(s[0] < s[1], "both actual scorer branches must use history");
        }
    }
    #[test]
    fn split_and_combined_requests_have_equal_author_exponents_at_equal_time() {
        let mut q = query();
        let c = [candidate(10, 1), candidate(11, 1)];
        let together = coefficients(&q, &c);
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1))];
        assert_eq!(coefficients(&q, &c[1..]), vec![together[1]]);
        assert_eq!(together, vec![1.0, 0.625]);
    }
    #[test]
    fn replayed_history_and_ancestor_expansion_do_not_multiply_penalties() {
        let mut q = query();
        let p = ScoredPost {
            ancestors: vec![8, 9],
            ..post(10, 1)
        };
        let h = batch(1, q.request_time_ms, &p);
        assert_eq!(h.entries.len(), 3);
        q.served_history = vec![h.clone(), h];
        assert_eq!(coefficients(&q, &[candidate(11, 1)]), vec![0.625]);
    }
    #[test]
    fn original_author_is_consistent_for_retweets_in_history_and_current_pool() {
        let mut q = query();
        q.served_history = vec![batch(
            1,
            q.request_time_ms,
            &ScoredPost {
                retweeted_tweet_id: 50,
                retweeted_user_id: 99,
                ..post(10, 1)
            },
        )];
        let c = [
            PostCandidate {
                retweeted_user_id: Some(99),
                ..candidate(11, 2)
            },
            PostCandidate {
                retweeted_user_id: Some(99),
                ..candidate(12, 3)
            },
            candidate(13, 2),
        ];
        assert_eq!(coefficients(&q, &c), vec![0.625, 0.4375, 1.0]);
        q.params.set(EnableCrossRequestAuthorDiversity, false);
        assert_eq!(coefficients(&q, &c), vec![1.0, 1.0, 0.625]); // legacy serving-author keys
    }
    #[test]
    fn zero_retweet_author_falls_back_to_serving_author() {
        let mut q = query();
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1))];
        assert_eq!(
            coefficients(
                &q,
                &[PostCandidate {
                    retweeted_user_id: Some(0),
                    ..candidate(11, 1)
                }]
            ),
            vec![0.625]
        );
    }
    #[test]
    fn advertisements_and_invalid_metadata_do_not_become_author_exposures() {
        let mut q = query();
        let mut ads = batch(1, q.request_time_ms, &post(10, 1));
        ads.entries[0].entity_type = EntityIdType::PROMOTED_TWEET;
        let mut unknown = batch(2, q.request_time_ms, &post(11, 1));
        unknown.entries[0].item_ids.as_mut().unwrap()[0].source_author_id = None;
        let mut bad_id = batch(3, q.request_time_ms, &post(12, 1));
        bad_id.entries[0].item_ids.as_mut().unwrap()[0].tweet_id = Some(-1);
        q.served_history = vec![ads, unknown, bad_id];
        assert_eq!(coefficients(&q, &[candidate(13, 1)]), vec![1.0]);
    }
    #[test]
    fn expiry_future_missing_timestamps_and_empty_history_are_safe() {
        let mut q = query();
        let mut missing = batch(1, 1, &post(10, 1));
        missing.served_time_ms = None;
        q.served_history = vec![
            missing,
            batch(2, q.request_time_ms + 1, &post(11, 1)),
            batch(3, q.request_time_ms - 300_000, &post(12, 1)),
        ];
        assert_eq!(coefficients(&q, &[candidate(13, 1)]), vec![1.0]);
        q.served_history.clear();
        assert_eq!(coefficients(&q, &[candidate(13, 1)]), vec![1.0]);
    }
    #[test]
    fn missing_position_uses_original_post_id_to_deduplicate_reposts() {
        let mut q = query();
        let mut h = batch(1, q.request_time_ms, &post(10, 1));
        h.entries[0].sort_index = None;
        h.entries.push(h.entries[0].clone());
        q.served_history = vec![h];
        assert_eq!(coefficients(&q, &[candidate(11, 1)]), vec![0.625]);
    }
    #[test]
    fn every_off_or_failure_path_preserves_legacy_scores() {
        let mut q = query();
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1))];
        let c = [
            PostCandidate {
                retweeted_user_id: Some(1),
                ..candidate(11, 2)
            },
            candidate(12, 3),
        ];
        for pre_offset in [false, true] {
            let mut baseline = q.clone();
            baseline.params.set(MultiplierPreOffset, pre_offset);
            baseline
                .params
                .set(EnableCrossRequestAuthorDiversity, false);
            let expected = score(&baseline, &c);
            for kind in 0..6 {
                let mut disabled = q.clone();
                disabled.params.set(MultiplierPreOffset, pre_offset);
                match kind {
                    0 => disabled
                        .params
                        .set(EnableCrossRequestAuthorDiversity, false),
                    1 => disabled.served_history_loaded = false,
                    2 => disabled.request_type = RequestType::RankedFollowing,
                    3 => disabled.params.set(AuthorDiversityHistoryWeight, 0.0),
                    4 => disabled
                        .params
                        .set(AuthorDiversityHistoryHalfLifeSeconds, 0.0),
                    _ => disabled.params.set(AuthorDiversityHistoryMaxItems, 0usize),
                }
                assert_eq!(score(&disabled, &c), expected);
            }
        }
    }
    #[test]
    fn master_author_diversity_switch_disables_history_in_both_score_branches() {
        for pre in [false, true] {
            let mut q = query();
            q.params.set(EnableAuthorDiversity, false);
            q.params.set(MultiplierPreOffset, pre);
            let c = [candidate(10, 1), candidate(11, 1)];
            let expected = score(&q, &c);
            q.served_history = vec![batch(1, q.request_time_ms, &post(9, 1))];
            assert_eq!(score(&q, &c), expected);
        }
    }
    #[test]
    fn cached_slate_context_does_not_override_fresh_history() {
        let mut q = query();
        q.has_cached_posts = true;
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1))];
        let context = SlateContext {
            k: 50,
            fatigue: 99.0,
        };
        let c = [
            PostCandidate {
                slate_context: Some(context),
                served_slate_context: Some(context),
                ..candidate(11, 1)
            },
            candidate(12, 2),
        ];
        for pre in [false, true] {
            q.params.set(MultiplierPreOffset, pre);
            let s = score(&q, &c);
            assert!(s[0] < s[1]);
        }
    }
    #[test]
    fn failed_history_read_fails_open_and_empty_success_is_distinguishable() {
        let mut q = query();
        q.served_history_loaded = false;
        let failed = ServedHistoryQueryHydrator {
            client: Arc::new(MockHistory {
                fail: true,
                ..Default::default()
            }),
        };
        assert!(ready(failed.hydrate(&q)).is_err());
        assert!(!q.served_history_loaded);
        assert!(util::author_history::counts_for_query(&q).is_none());
        let h = ServedHistoryQueryHydrator {
            client: Arc::new(MockHistory::default()),
        };
        let r = ready(h.hydrate(&q)).unwrap();
        h.update(&mut q, r);
        assert!(q.served_history_loaded);
        assert_eq!(
            util::author_history::counts_for_query(&q),
            Some(HashMap::new())
        );
    }
    #[test]
    fn history_is_scoped_to_the_viewer_by_existing_reader() {
        let client = Arc::new(MockHistory::default());
        client
            .batches
            .lock()
            .unwrap()
            .insert(7, vec![batch(1, 1_000_000, &post(10, 1))]);
        let h = ServedHistoryQueryHydrator { client };
        let mut q = query();
        q.user_id = 8;
        q.served_history_loaded = false;
        let r = ready(h.hydrate(&q)).unwrap();
        h.update(&mut q, r);
        assert!(q.served_history.is_empty());
        assert_eq!(coefficients(&q, &[candidate(11, 1)]), vec![1.0]);
    }
    #[test]
    fn large_history_falls_back_instead_of_partially_biasing_the_count() {
        let mut q = query();
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1)); 20_001];
        assert!(util::author_history::counts_for_query(&q).is_none());
        assert_eq!(coefficients(&q, &[candidate(11, 1)]), vec![1.0]);
    }
    #[test]
    fn history_decay_does_not_accidentally_rescale_negative_net_pre_offset_scores() {
        let mut q = query();
        q.params.set(MultiplierPreOffset, true);
        let c = [PostCandidate {
            base_positive: 0.0,
            base_negative: 0.5,
            ..candidate(11, 1)
        }];
        let before = score(&q, &c);
        q.served_history = vec![batch(1, q.request_time_ms, &post(10, 1))];
        assert_eq!(score(&q, &c), before);
    }
    #[test]
    fn deterministic_sessions_use_actual_fixed_functions() {
        for weight in [0.0, 0.25, 1.0] {
            let mut q = query();
            q.params.set(AuthorDiversityHistoryWeight, weight);
            let mut a_count = 0;
            let mut unique = HashSet::new();
            for page in 0..12u64 {
                q.request_time_ms = 1_000_000 + page as i64 * 10_000;
                let c = [
                    candidate(page * 10 + 1, 1),
                    candidate(page * 10 + 2, 1),
                    candidate(page * 10 + 3, 100 + page * 2),
                    candidate(page * 10 + 4, 101 + page * 2),
                ];
                let s = RankingScorer::apply_author_diversity(&q, &c, &[100.0, 99.0, 70.0, 69.0]);
                let mut indices = vec![0, 1, 2, 3];
                indices.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
                let mut entries = vec![];
                for (slot, &i) in indices[..2].iter().enumerate() {
                    if c[i].author_id == 1 {
                        a_count += 1;
                    }
                    unique.insert(c[i].author_id);
                    entries.extend(build_post_entries(
                        &post(c[i].tweet_id, c[i].author_id),
                        slot as i64,
                    ));
                }
                q.served_history.push(ServedHistory {
                    request_type: served_history::RequestType::INITIAL,
                    served_id: Some(page as i64 + 1),
                    served_time_ms: Some(q.request_time_ms),
                    entries,
                });
            }
            let expected = if weight == 0.0 {
                12
            } else if weight == 0.25 {
                7
            } else {
                3
            };
            assert_eq!(a_count, expected);
            println!(
                "weight={weight}: A {a_count}/24, unique authors={}",
                unique.len()
            );
        }
    }
}
